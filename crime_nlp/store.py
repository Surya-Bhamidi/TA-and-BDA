"""Disk-backed case retrieval and explicit opt-in feedback; no full text in overview memory."""
import json
import sqlite3
import hashlib
from contextlib import closing
from datetime import datetime, timezone
import pandas as pd
from .config import PROCESSED

DB = PROCESSED / "cases.sqlite"


def build_store(frame):
    target = DB.with_suffix(".building.sqlite")
    rows = frame.copy()
    rows.insert(0, "row_number", range(len(rows)))
    rows["event_time"] = rows.event_time.astype(str)
    with closing(sqlite3.connect(target)) as connection:
        rows.to_sql("reports", connection, if_exists="replace", index=False, chunksize=2000)
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_row ON reports(row_number)")
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_report ON reports(report_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_filters ON reports(crime_type,district,reported_at)")
        connection.commit()
    target.replace(DB)


def fetch_reports(row_numbers):
    positions = [int(i) for i in row_numbers]
    if not positions:
        return pd.DataFrame()
    with closing(sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)) as connection:
        frame = pd.read_sql_query(f"SELECT * FROM reports WHERE row_number IN ({','.join('?' for _ in positions)})", connection, params=positions)
    return frame.set_index("row_number").reindex(positions)


def save_feedback(text, predicted, correction, comment=""):
    """Store a hash and correction only. Submitted narrative text is not persisted."""
    path = PROCESSED / "feedback.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS feedback (created_at TEXT, narrative_sha256 TEXT, predicted TEXT, corrected TEXT, comment TEXT)")
        connection.execute("INSERT INTO feedback VALUES (?,?,?,?,?)", (datetime.now(timezone.utc).isoformat(), hashlib.sha256(text.encode()).hexdigest(), predicted, correction, comment[:1000]))
        connection.commit()


def entity_matches(label="Any", value=""):
    clauses, params = [], []
    if label != "Any":
        clauses.append("json_extract(e.value,'$.label') = ?")
        params.append(label)
    if value.strip():
        clauses.append("instr(lower(json_extract(e.value,'$.text')),lower(?)) > 0")
        params.append(value.strip())
    if not clauses:
        return None
    sql = "SELECT row_number FROM reports r WHERE EXISTS (SELECT 1 FROM json_each(r.predicted_entities_json) e WHERE " + " AND ".join(clauses) + ")"
    with closing(sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)) as connection:
        return [row[0] for row in connection.execute(sql, params)]


def record_audit(event, report_id):
    """Opt-in local access events contain only case IDs and event names."""
    with closing(sqlite3.connect(PROCESSED / "audit.sqlite")) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS events (at TEXT, event TEXT, report_id TEXT)")
        connection.execute("INSERT INTO events VALUES (?,?,?)", (datetime.now(timezone.utc).isoformat(), event, report_id))
        connection.commit()
