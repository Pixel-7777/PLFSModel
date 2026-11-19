# Runtime data

`predictions.db` is created automatically when the server starts. It is not
included in the bundle so the project starts with an empty log.

SQLite table:

```sql
CREATE TABLE predictions (
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
);
```

`inputs_json` stores the model-ready input row after derived fields and
state/district construction. `summary_json` stores the short labels used by
the recent-predictions table.

To reset the log, stop the server and delete this file.
