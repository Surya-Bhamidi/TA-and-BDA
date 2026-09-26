"""Validate external JSONL into a separate staging area without changing the benchmark."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, CATEGORIES, ENTITY_LABELS, write_json


def validate_record(row):
    if not isinstance(row, dict):
        raise ValueError("Each line must contain a JSON object")
    for field in ("report_id", "narrative", "reported_at", "district"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    if len(row["narrative"]) > 30000 or "\x00" in row["narrative"]:
        raise ValueError("Narrative must contain at most 30,000 characters and no null bytes")
    datetime.fromisoformat(row["reported_at"].replace("Z", "+00:00"))
    if row.get("crime_type") is not None and row["crime_type"] not in CATEGORIES:
        raise ValueError("Unknown crime_type")
    entities = row.get("entities", [])
    if not isinstance(entities, list):
        raise ValueError("entities must be a list")
    previous_end = 0
    for entity in sorted(entities, key=lambda e: e.get("start", -1)):
        start, end = entity.get("start"), entity.get("end")
        if type(start) is not int or type(end) is not int or not (previous_end <= start < end <= len(row["narrative"])):
            raise ValueError("Entity spans must be valid non-overlapping character offsets")
        if entity.get("label") not in ENTITY_LABELS or row["narrative"][start:end] != entity.get("text"):
            raise ValueError("Entity label or span text is invalid")
        previous_end = end
    return {key: row[key] for key in ("report_id", "narrative", "reported_at", "district", "crime_type", "entities") if key in row}


def stage_jsonl(source, source_url, license_name, output_dir=None):
    if not source_url.strip() or not license_name.strip():
        raise ValueError("Record the source URL and permission/license before importing")
    source = Path(source)
    target = Path(output_dir) if output_dir else ROOT / "data" / "external" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    target.mkdir(parents=True, exist_ok=False)
    count, rejected, seen = 0, 0, set()
    with source.open(encoding="utf-8-sig") as input_file, (target / "validated.jsonl").open("w", encoding="utf-8") as accepted, (target / "rejections.jsonl").open("w", encoding="utf-8") as errors:
        for number, line in enumerate(input_file, 1):
            try:
                row = validate_record(json.loads(line))
                if row["report_id"] in seen:
                    raise ValueError("Duplicate report_id")
                seen.add(row["report_id"])
                accepted.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                errors.write(json.dumps({"line": number, "reason": str(error)}) + "\n")
                rejected += 1
    receipt = {"accepted": count, "rejected": rejected, "source_url": source_url, "license": license_name,
               "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "created_at": datetime.now(timezone.utc).isoformat(),
               "status": "staged only; not added to synthetic training or dashboard"}
    write_json(target / "provenance.json", receipt)
    return target, receipt
