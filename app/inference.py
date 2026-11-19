"""Inference wrapper for the finalized PLFS wage quantile model.

The model and preprocessing bundle are loaded read-only from ``models/wage``.
No retraining, refitting or calibration happens here.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MODEL_DIR = ROOT / "models" / "wage"

WEIGHTS_NAME = "wage_quantile_mlp.pt"
BUNDLE_NAME = "wage_preprocessing.pkl"


class ResidualBlock(nn.Module):
    def __init__(self, size: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.LayerNorm(size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(size, size),
            nn.LayerNorm(size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.GELU()(x + self.block(x))


class TabularEncoder(nn.Module):
    def __init__(self, embedding_sizes: list[tuple[int, int]], numeric_size: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n, d) for n, d in embedding_sizes])
        total_emb_size = sum(d for _, d in embedding_sizes)
        self.input_layer = nn.Sequential(
            nn.Linear(total_emb_size + numeric_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(256), ResidualBlock(256), ResidualBlock(256), ResidualBlock(256)
        )

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        embs = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [numeric], dim=1)
        return self.residual_blocks(self.input_layer(x))


class QuantileRegressor(nn.Module):
    def __init__(self, embedding_sizes: list[tuple[int, int]], numeric_size: int) -> None:
        super().__init__()
        self.encoder = TabularEncoder(embedding_sizes, numeric_size)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(categorical, numeric))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WageModel:
    """Load the frozen model once and score single-person input rows."""

    def __init__(self, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.weights_path = MODEL_DIR / WEIGHTS_NAME
        self.bundle_path = MODEL_DIR / BUNDLE_NAME

        if not self.weights_path.exists():
            raise FileNotFoundError(self.weights_path)
        if not self.bundle_path.exists():
            raise FileNotFoundError(self.bundle_path)

        self.bundle = pickle.loads(self.bundle_path.read_bytes())
        self.numeric_features: list[str] = list(self.bundle["numeric_features"])
        self.categorical_features: list[str] = list(self.bundle["categorical_features"])
        self.quantiles: list[float] = list(self.bundle["quantiles"])
        self.embedding_sizes = list(self.bundle["embedding_sizes"])

        self.model = QuantileRegressor(self.embedding_sizes, len(self.numeric_features))
        state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        info_path = MODEL_DIR / "model_info.json"
        metrics_path = MODEL_DIR / "metrics.json"
        self.metadata = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        self.metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        self.schema = json.loads((APP_DIR / "labels.json").read_text(encoding="utf-8"))
        self.defaults: dict[str, Any] = self.schema["defaults"]
        self.numeric_defaults: dict[str, float] = self.schema["numeric_defaults"]
        self.numeric_specs: dict[str, dict[str, Any]] = self.schema["numeric_specs"]
        self.allowed: dict[str, set[str]] = {
            field: {opt["value"] for opt in spec["options"]}
            for field, spec in self.schema["fields"].items()
        }

        self.weights_sha256 = sha256(self.weights_path)
        self.preprocessing_sha256 = sha256(self.bundle_path)
        self.version = "PLFS wage quantile MLP"

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------
    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out):
            return None
        return out

    @staticmethod
    def _as_cat(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value in {"", "__missing__", "null", "None"}:
                return None
        return str(value)

    def prepare(self, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Return a model-ready input row plus human-readable warnings."""
        warnings: list[str] = []

        row: dict[str, Any] = {}
        for feature in self.numeric_features:
            row[feature] = self._as_float(payload.get(feature, self.numeric_defaults.get(feature)))

        age = row["age"]
        education = row["years_formal_education"]
        if age is None:
            raise ValueError("Age is required.")
        if education is None:
            row["potential_experience"] = None
            row["potential_experience_sq"] = None
            warnings.append(
                "Education is missing: the model uses the training median for education "
                "and potential experience."
            )
        else:
            experience = max(0.0, age - education - 6.0)
            row["potential_experience"] = experience
            row["potential_experience_sq"] = experience**2

        total_hours = row["tothrs_wrk"]
        if total_hours is None:
            raise ValueError("Total hours in the reference week is required.")
        if total_hours < 0:
            raise ValueError("Total hours cannot be negative.")
        if row["days_worked"] is None:
            raise ValueError("Days worked in the reference week is required.")
        if row["days_worked"] < 0:
            raise ValueError("Days worked cannot be negative.")
        row["avg_daily_hours"] = total_hours / 7.0

        for feature in self.numeric_features:
            spec = self.numeric_specs.get(feature, {})
            value = row[feature]
            if value is None:
                continue
            lo = spec.get("min")
            hi = spec.get("max")
            if lo is not None and value < lo:
                warnings.append(f"{spec.get('label', feature)} is below the training range ({lo}).")
            if hi is not None and value > hi:
                warnings.append(f"{spec.get('label', feature)} is above the training range ({hi}).")

        for feature in self.categorical_features:
            if feature == "state_district":
                continue
            if feature in payload:
                value = self._as_cat(payload[feature])
            else:
                value = self._as_cat(self.defaults.get(feature))
            if value is not None and value not in self.allowed.get(feature, set()):
                warnings.append(
                    f"{self.schema['fields'][feature]['label']} value {value!r} is not a "
                    "training category; the model will use its unknown-category embedding."
                )
            row[feature] = value

        st = row.get("st")
        dc = row.get("dc")
        state_district = f"{st}_{dc}" if st and dc else None
        if state_district is not None and state_district not in self.allowed.get("state_district", set()):
            warnings.append(
                "This state/district combination was not in the training data; the model "
                "will use its unknown-category embedding."
            )
        row["state_district"] = state_district

        if row.get("acws") != "31":
            warnings.append(
                "The model was trained on regular salaried/wage workers (current weekly "
                "activity status 31). Other statuses are outside the target population."
            )
        if row.get("days_worked") == 0 or total_hours == 0:
            warnings.append(
                "Zero days or zero hours is unusual for a regular wage worker; the estimate "
                "should be treated as an out-of-range scenario."
            )

        return row, warnings

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        row, warnings = self.prepare(payload)

        numeric = pd.DataFrame(
            [[row[f] for f in self.numeric_features]], columns=self.numeric_features
        )
        numeric = self.bundle["numeric_imputer"].transform(numeric)
        numeric = self.bundle["numeric_scaler"].transform(numeric)

        categorical = pd.DataFrame(
            [[row[f] if row[f] is not None else np.nan for f in self.categorical_features]],
            columns=self.categorical_features,
            dtype=object,
        )
        categorical = self.bundle["categorical_imputer"].transform(categorical)
        categorical = self.bundle["categorical_encoder"].transform(categorical)
        categorical = torch.tensor(categorical.astype(np.int64) + 1, dtype=torch.long)
        numeric_t = torch.tensor(numeric, dtype=torch.float32)

        categorical = categorical.to(self.device)
        numeric_t = numeric_t.to(self.device)
        raw = self.model(categorical, numeric_t).detach().cpu().numpy()[0]
        raw = np.maximum(np.expm1(raw), 0.0)
        q05, q50, q95 = (float(v) for v in raw)

        crossed = not (q05 <= q50 <= q95)
        if crossed:
            q05, q50, q95 = sorted((q05, q50, q95))
            warnings.append(
                "The three raw quantiles crossed; they were sorted for display. This is a "
                "single-row post-processing step only."
            )

        if q95 > 100_000:
            tail = self.metrics.get("tail_100k", {})
            coverage = float(tail.get("coverage_90", 0.8408))
            mae = float(tail.get("mae", 50303.0))
            warnings.append(
                "This estimate reaches the high-wage tail: on test rows above Rs 100,000, "
                f"5-95 coverage was {100 * coverage:.1f}% and MAE was "
                f"Rs {mae:,.0f}. Treat the upper tail as less certain."
            )

        return {
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "interval_width": q95 - q05,
            "crossed_raw": crossed,
            "warnings": warnings,
            "inputs_used": row,
            "model": {
                "version": self.version,
                "target": self.bundle["target"],
                "quantiles": self.quantiles,
                "weights_sha256": self.weights_sha256,
                "preprocessing_sha256": self.preprocessing_sha256,
            },
        }
