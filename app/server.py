"""Small local web app for the finalized PLFS regular-wage model.

Run from the bundle root:

    python app/server.py

Then open http://127.0.0.1:8000

Only the Python standard library is used for the web server and SQLite storage;
model inference uses the same torch / pandas / scikit-learn stack as the
notebooks.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import sqlite3
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
DB_PATH = APP_DIR / "data" / "predictions.db"

sys.path.insert(0, str(APP_DIR))
from inference import WageModel  # noqa: E402

SCHEMA = json.loads((APP_DIR / "labels.json").read_text(encoding="utf-8"))
MODEL = WageModel()

# Replace any frozen-model facts in labels.json with the current artifacts
# produced by 02_wage_model_end_to_end.ipynb.
_info = MODEL.metadata or {}
_metrics = MODEL.metrics or {}
_baseline = _metrics.get("baseline_test", {}) if isinstance(_metrics, dict) else {}
SCHEMA["model"] = {
    "model_name": "PLFS regular-wage quantile MLP",
    "model_version": MODEL.version,
    "target": MODEL.bundle.get("target", "ern_reg_current"),
    "target_description": "Monthly regular wage of a regular salaried/wage worker, 2025 prices.",
    "quantiles": MODEL.quantiles,
    "weights_sha256": MODEL.weights_sha256,
    "preprocessing_sha256": MODEL.preprocessing_sha256,
    "test_metrics": {
        "n": int(_info.get("dataset", {}).get("test_rows", _baseline.get("n", 0))),
        "coverage_90": float(_info.get("test_coverage", _baseline.get("coverage_90", 0.0))),
        "mae": float(_info.get("test_median_prediction_mae", _baseline.get("mae", 0.0))),
        "median_width": float(_info.get("test_median_interval_width", _baseline.get("median_width", 0.0))),
        "crossings": int(_info.get("test_quantile_crossings", _baseline.get("crossings", 0))),
    },
}

LABEL_LOOKUP: dict[str, dict[str, str]] = {
    field: {opt["value"]: opt["label"] for opt in spec["options"]}
    for field, spec in SCHEMA["fields"].items()
}
DISTRICT_LOOKUP: dict[tuple[str, str], str] = {}
for state, districts in SCHEMA["districts_by_state"].items():
    for district in districts:
        DISTRICT_LOOKUP[(state, district["value"])] = district["label"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model_version TEXT NOT NULL,
                q05 REAL NOT NULL,
                q50 REAL NOT NULL,
                q95 REAL NOT NULL,
                interval_width REAL NOT NULL,
                inputs_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                summary_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def make_summary(inputs: dict[str, Any]) -> dict[str, Any]:
    def label(field: str, value: Any) -> str:
        if value is None:
            return "Not specified"
        return LABEL_LOOKUP.get(field, {}).get(str(value), str(value))

    state = inputs.get("st")
    district = inputs.get("dc")
    district_label = DISTRICT_LOOKUP.get((str(state), str(district)))
    if district_label is None and district is not None:
        district_label = f"District {district}"

    return {
        "age": inputs.get("age"),
        "sex": label("sex", inputs.get("sex")),
        "education": label("gedu_lvl", inputs.get("gedu_lvl")),
        "state": label("st", state),
        "district": district_label or "Not specified",
        "sector": label("sec", inputs.get("sec")),
        "occupation": label("ocu_pas", inputs.get("ocu_pas")),
        "industry": label("ind_pas", inputs.get("ind_pas")),
        "acws": label("acws", inputs.get("acws")),
    }


def insert_prediction(inputs: dict[str, Any], result: dict[str, Any]) -> int:
    summary = make_summary(inputs)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (created_at, model_version, q05, q50, q95, interval_width,
                 inputs_json, result_json, summary_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                MODEL.version,
                float(result["q05"]),
                float(result["q50"]),
                float(result["q95"]),
                float(result["interval_width"]),
                json.dumps(inputs, ensure_ascii=False),
                json.dumps(
                    {
                        "q05": result["q05"],
                        "q50": result["q50"],
                        "q95": result["q95"],
                        "interval_width": result["interval_width"],
                        "warnings": result["warnings"],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(summary, ensure_ascii=False),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_predictions(limit: int = 20) -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, model_version, q05, q50, q95,
                   interval_width, summary_json
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "model_version": row["model_version"],
            "q05": row["q05"],
            "q50": row["q50"],
            "q95": row["q95"],
            "interval_width": row["interval_width"],
            "summary": json.loads(row["summary_json"]),
        }
        for row in rows
    ]


def count_predictions() -> int:
    with sqlite3.connect(DB_PATH) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])


def predictions_csv() -> str:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, model_version, q05, q50, q95,
                   interval_width, inputs_json, summary_json
            FROM predictions
            ORDER BY id
            """
        ).fetchall()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "id", "created_at", "model_version", "q05", "q50", "q95",
            "interval_width", "age", "sex", "education", "state", "district",
            "sector", "occupation", "industry", "inputs_json",
        ]
    )
    for row in rows:
        summary = json.loads(row["summary_json"])
        writer.writerow(
            [
                row["id"], row["created_at"], row["model_version"],
                f"{row['q05']:.2f}", f"{row['q50']:.2f}", f"{row['q95']:.2f}",
                f"{row['interval_width']:.2f}", summary.get("age", ""),
                summary.get("sex", ""), summary.get("education", ""),
                summary.get("state", ""), summary.get("district", ""),
                summary.get("sector", ""), summary.get("occupation", ""),
                summary.get("industry", ""), row["inputs_json"],
            ]
        )
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    server_version = "PLFSWageUI/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.path.startswith("/api/predict"):
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_text(
        self,
        body: str,
        status: int = HTTPStatus.OK,
        content_type: str = "text/plain; charset=utf-8",
        headers: dict[str, str] | None = None,
    ) -> None:
        raw = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(raw)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("Empty request body.")
        if length > 1_000_000:
            raise ValueError("Request body is too large.")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object.")
        return payload

    # ------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            return self.serve_static("index.html")
        if path.startswith("/static/"):
            return self.serve_static(path[len("/static/") :])

        if path == "/api/health":
            return self.send_json(
                {
                    "ok": True,
                    "model_version": MODEL.version,
                    "device": str(MODEL.device),
                    "weights_sha256": MODEL.weights_sha256,
                    "predictions_stored": count_predictions(),
                }
            )

        if path == "/api/schema":
            return self.send_json(SCHEMA)

        if path == "/api/predictions":
            query = parse_qs(parsed.query)
            try:
                limit = int(query.get("limit", ["20"])[0])
            except ValueError:
                limit = 20
            limit = max(1, min(limit, 100))
            return self.send_json({"items": list_predictions(limit)})

        if path == "/api/predictions.csv":
            body = predictions_csv()
            return self.send_text(
                body,
                content_type="text/csv; charset=utf-8",
                headers={
                    "Content-Disposition": 'attachment; filename="plfs_wage_predictions.csv"'
                },
            )

        return self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def serve_static(self, name: str) -> None:
        safe = (STATIC_DIR / name).resolve()
        try:
            safe.relative_to(STATIC_DIR.resolve())
        except ValueError:
            return self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        if not safe.is_file():
            return self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        content_type = mimetypes.guess_type(str(safe))[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript"}:
            content_type += "; charset=utf-8"
        body = safe.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------
    # POST
    # ------------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            return self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        try:
            body = self.read_json()
            inputs = body.get("inputs", body)
            if not isinstance(inputs, dict):
                raise ValueError("'inputs' must be an object.")
            result = MODEL.predict(inputs)
            prediction_id = insert_prediction(result["inputs_used"], result)
        except ValueError as exc:
            return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - defensive API boundary
            sys.stderr.write(f"Prediction error: {exc!r}\n")
            return self.send_json(
                {"error": "The prediction could not be computed. See the server log."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        return self.send_json(
            {
                "id": prediction_id,
                "created_at": utc_now(),
                "prediction": {
                    "q05": result["q05"],
                    "q50": result["q50"],
                    "q95": result["q95"],
                    "interval_width": result["interval_width"],
                    "currency": "INR",
                    "period": "month",
                    "price_base": "2025",
                },
                "warnings": result["warnings"],
                "inputs_used": result["inputs_used"],
                "model": result["model"],
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="PLFS wage model web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    init_db()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print("=" * 72)
    print("PLFS regular-wage model UI")
    print(f"Model:  {MODEL.version}")
    print(f"Weights: {MODEL.weights_sha256}")
    print(f"Serving: http://{args.host}:{args.port}")
    print(f"Database: {DB_PATH}")
    print("Press Ctrl+C to stop.")
    print("=" * 72)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
