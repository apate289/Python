"""
Daily ETL Pipeline — Banking Clients
Implements SCD Type 2 (Slowly Changing Dimension) versioning:
  - New records → insert with version=1, is_active=True
  - Changed records → archive old version (is_active=False, end_date=now),
                      insert new version (version+1, is_active=True)
  - Unchanged records → skip
  - Removed records → mark is_active=False

Run:  python etl/pipeline.py [--source data/clients.json] [--simulate-changes]
"""

import argparse
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError

# ── local imports ─────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from schemas.mongo_schema import setup_collections
from config.settings import MONGO_URI, DB_NAME

# ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("etl.pipeline")

# ─────────────────────────────────────────────────
CHANGE_FIELDS = [
    ("banking_profile.credit_score",   lambda r: max(300, min(850, r["banking_profile"]["credit_score"] + random.randint(-30, 30)))),
    ("banking_profile.kyc_verified",   lambda r: not r["banking_profile"]["kyc_verified"]),
    ("banking_profile.preferred_contact", lambda r: random.choice(["Email", "Phone", "Mail", "SMS"])),
    ("address.city",                   lambda r: "Updated City"),
    ("accounts.0.balance",             lambda r: round(r["accounts"][0]["balance"] * random.uniform(0.85, 1.15), 2)),
]


def _deep_set(d: dict, dotted_key: str, value: Any):
    """Set a value in a nested dict using dot-notation key."""
    keys = dotted_key.split(".")
    obj = d
    for k in keys[:-1]:
        try:
            k = int(k)
        except ValueError:
            pass
        obj = obj[k]
    try:
        last = int(keys[-1])
    except ValueError:
        last = keys[-1]
    obj[last] = value


def simulate_changes(clients: list[dict], change_ratio: float = 0.2) -> list[dict]:
    """Randomly mutate a fraction of records to simulate daily changes."""
    to_change = random.sample(clients, int(len(clients) * change_ratio))
    for rec in to_change:
        field, mutator = random.choice(CHANGE_FIELDS)
        try:
            _deep_set(rec, field, mutator(rec))
            log.debug("  Simulated change: %s → %s", rec["client_id"], field)
        except Exception:
            pass
    return clients


def load_source(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    log.info("Loaded %d records from %s", len(data), path)
    return data


def _fingerprint(record: dict) -> str:
    """Stable hash of mutable fields (excludes versioning metadata)."""
    import hashlib, json as _json
    exclude = {"is_active","version","start_date","end_date","etl_batch_id","created_at","updated_at","_id"}
    filtered = {k: v for k, v in record.items() if k not in exclude}
    return hashlib.sha256(_json.dumps(filtered, sort_keys=True, default=str).encode()).hexdigest()


class ETLPipeline:
    def __init__(self, db):
        self.col = db["banking_clients"]
        self.audit = db["etl_audit_log"]
        self.batch_id = str(uuid.uuid4())
        self.now = datetime.utcnow().isoformat()
        self.stats = {
            "records_processed": 0,
            "records_inserted": 0,
            "records_updated": 0,
            "records_archived": 0,
            "records_unchanged": 0,
            "errors": [],
        }

    # ── helpers ──────────────────────────────────
    def _get_active(self, client_id: str) -> dict | None:
        return self.col.find_one({"client_id": client_id, "is_active": True})

    def _archive(self, doc: dict):
        self.col.update_one(
            {"_id": doc["_id"]},
            {"$set": {"is_active": False, "end_date": self.now}},
        )
        self.stats["records_archived"] += 1

    def _insert_version(self, record: dict, version: int):
        record = dict(record)
        record.pop("_id", None)
        record.update({
            "is_active": True,
            "version": version,
            "start_date": self.now,
            "end_date": None,
            "etl_batch_id": self.batch_id,
            "updated_at": self.now,
            "created_at": record.get("created_at", self.now),
        })
        self.col.insert_one(record)

    # ── core ─────────────────────────────────────
    def process_record(self, incoming: dict):
        self.stats["records_processed"] += 1
        client_id = incoming["client_id"]

        existing = self._get_active(client_id)

        if existing is None:
            # Brand-new client
            self._insert_version(incoming, version=1)
            self.stats["records_inserted"] += 1
            log.debug("  INSERT  %s  (new client)", client_id)
            return

        old_fp = _fingerprint(existing)
        new_fp = _fingerprint(incoming)

        if old_fp == new_fp:
            self.stats["records_unchanged"] += 1
            log.debug("  SKIP    %s  (no changes)", client_id)
            return

        # Changed → SCD Type 2
        new_version = existing["version"] + 1
        self._archive(existing)
        self._insert_version(incoming, version=new_version)
        self.stats["records_updated"] += 1
        log.debug("  UPDATE  %s  v%d → v%d", client_id, existing["version"], new_version)

    def deactivate_removed(self, source_ids: set[str]):
        """Mark any active client not in today's source as inactive."""
        active_ids = {
            doc["client_id"]
            for doc in self.col.find({"is_active": True}, {"client_id": 1})
        }
        removed = active_ids - source_ids
        if removed:
            result = self.col.update_many(
                {"client_id": {"$in": list(removed)}, "is_active": True},
                {"$set": {"is_active": False, "end_date": self.now}},
            )
            self.stats["records_archived"] += result.modified_count
            log.info("Deactivated %d removed clients.", result.modified_count)

    def run(self, records: list[dict]):
        log.info("ETL batch %s started — %d records", self.batch_id, len(records))
        t0 = time.time()

        for rec in records:
            try:
                self.process_record(rec)
            except Exception as exc:
                msg = f"{rec.get('client_id','?')}: {exc}"
                self.stats["errors"].append(msg)
                log.error("  ERROR  %s", msg)

        source_ids = {r["client_id"] for r in records}
        self.deactivate_removed(source_ids)

        duration = round(time.time() - t0, 3)
        self._write_audit(duration)
        self._print_summary(duration)

    def _write_audit(self, duration: float):
        self.audit.insert_one({
            "batch_id":          self.batch_id,
            "run_date":          self.now,
            "status":            "failed" if self.stats["errors"] else "success",
            "records_processed": self.stats["records_processed"],
            "records_inserted":  self.stats["records_inserted"],
            "records_updated":   self.stats["records_updated"],
            "records_archived":  self.stats["records_archived"],
            "errors":            self.stats["errors"] or None,
            "duration_seconds":  duration,
        })

    def _print_summary(self, duration: float):
        s = self.stats
        log.info(
            "ETL complete in %.2fs | "
            "processed=%d  inserted=%d  updated=%d  archived=%d  unchanged=%d  errors=%d",
            duration,
            s["records_processed"], s["records_inserted"], s["records_updated"],
            s["records_archived"], s["records_unchanged"], len(s["errors"]),
        )


# ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Banking ETL Pipeline")
    parser.add_argument("--source", default="data/clients.json", help="Source JSON file")
    parser.add_argument("--simulate-changes", action="store_true",
                        help="Randomly mutate 20%% of records to simulate daily delta")
    args = parser.parse_args()

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    setup_collections(db)

    source_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.source)
    records = load_source(source_path)

    if args.simulate_changes:
        log.info("Simulating changes on 20%% of records …")
        records = simulate_changes(records, change_ratio=0.2)

    pipeline = ETLPipeline(db)
    pipeline.run(records)
    client.close()


if __name__ == "__main__":
    main()
