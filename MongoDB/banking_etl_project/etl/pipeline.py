"""
Daily ETL Pipeline — Banking Clients (v2 — Field-Level Hash Comparison)
=======================================================================
SCD Type 2 versioning with granular field-level SHA-256 fingerprinting:

  DOCUMENT hash  — did anything change at all?
  SECTION  hashes — which top-level section changed?
  FIELD    hashes — which exact leaf field changed?
  ACCOUNT  hashes — which account changed, and which fields inside it?

Every version stored in MongoDB carries a `_hashes` manifest.
Every update produces a `field_diff` document in the `etl_field_diffs`
collection recording exactly what changed, with old/new values and hashes.

Run:
  python etl/pipeline.py                       # initial load
  python etl/pipeline.py --simulate-changes    # simulate daily delta
  python etl/pipeline.py --report CLT00001     # print diff history for one client
"""

from __future__ import annotations

import argparse
import json, pathlib
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pymongo import MongoClient

from config.settings import MONGO_URI, DB_NAME
from schemas.mongo_schema import setup_collections
from etl.hash_engine import (
    build_hash_manifest,
    diff_manifests,
    explain_diff,
    FieldDiff,
)

# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("etl.pipeline")

# ─────────────────────────────────────────────
#  Simulated field mutations
# ─────────────────────────────────────────────
CHANGE_FIELDS = [
    ("banking_profile.credit_score",
     lambda r: max(300, min(850, r["banking_profile"]["credit_score"] + random.randint(-40, 40)))),
    ("banking_profile.kyc_verified",
     lambda r: not r["banking_profile"]["kyc_verified"]),
    ("banking_profile.preferred_contact",
     lambda r: random.choice(["Email", "Phone", "Mail", "SMS"])),
    ("banking_profile.relationship_manager",
     lambda r: random.choice(["Alice Chen", "Bob Kumar", "Diana Park", "Ethan Ross"])),
    ("address.city",
     lambda r: random.choice(["Chicago", "Austin", "Denver", "Phoenix", "Portland"])),
    ("address.zip_code",
     lambda r: str(random.randint(10000, 99999))),
    ("personal_information.email",
     lambda r: r["personal_information"]["email"].replace("@", f"{random.randint(1,9)}@")),
    ("personal_information.phone_primary",
     lambda r: f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}"),
    ("check_printing.check_style",
     lambda r: random.choice(["High Security", "Standard", "Business", "Laser"])),
    ("accounts.0.balance",
     lambda r: round(r["accounts"][0]["balance"] * random.uniform(0.80, 1.20), 2)),
    ("accounts.0.status",
     lambda r: random.choice(["Active", "Dormant", "Frozen"])),
    ("accounts.0.interest_rate",
     lambda r: round(random.uniform(0.5, 5.5), 2)),
]


def _deep_set(obj: Any, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
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


def simulate_changes(clients: list[dict], change_ratio: float = 0.25) -> list[dict]:
    n_change = max(1, int(len(clients) * change_ratio))
    to_change = random.sample(clients, n_change)
    for rec in to_change:
        n_mutations = random.randint(1, 3)
        mutations = random.sample(CHANGE_FIELDS, min(n_mutations, len(CHANGE_FIELDS)))
        for dotted, mutator in mutations:
            try:
                _deep_set(rec, dotted, mutator(rec))
            except Exception:
                pass
    return clients


def load_source(directory: str) -> list[dict]:
    
    path = pathlib.Path(directory)
    json_contents = []
    # Validation: Ensure the directory exists
    if not path.is_dir():
        logging.error(f"Directory not found: {directory}")
        return []
    
    # Iterate: Use glob for efficient file filtering
    #for root, dirs, files in os.walk('your_directory_path'):
    for file_path in path.glob("*.json"):
        try:
            # Safe Access: Use context managers and explicit encoding
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                json_contents.append(data)
                
        # Error Handling: Catch malformed JSON or OS issues
        except json.JSONDecodeError as e:
            logging.warning(f"Skipping malformed JSON file {file_path.name}: {e}")
        except IOError as e:
            logging.error(f"Could not read file {file_path.name}: {e}")

    logging.info(f"Successfully loaded {len(json_contents)} JSON files.")
    
    #with open(path) as f:
    #    data = json.load(f)
    #log.info("Loaded %d source records from %s", len(data), path)
    #return data
    return json_contents


# ─────────────────────────────────────────────
class ETLPipeline:
    def __init__(self, db):
        self.col   = db["banking_clients"]
        self.diffs = db["etl_field_diffs"]
        self.audit = db["etl_audit_log"]
        self.batch_id = str(uuid.uuid4())
        self.now      = datetime.utcnow().isoformat()
        self.stats = {
            "records_processed":  0,
            "records_inserted":   0,
            "records_updated":    0,
            "records_archived":   0,
            "records_unchanged":  0,
            "fields_changed":     0,
            "sections_changed":   0,
            "errors":             [],
        }

    def _get_active(self, client_id: str) -> dict | None:
        return self.col.find_one({"client_id": client_id, "is_active": True})

    def _archive(self, doc: dict) -> None:
        self.col.update_one(
            {"_id": doc["_id"]},
            {"$set": {"is_active": False, "end_date": self.now}},
        )
        self.stats["records_archived"] += 1

    def _insert_version(self, record: dict, version: int, hashes: dict) -> None:
        rec = dict(record)
        rec.pop("_id", None)
        rec.update({
            "is_active":    True,
            "version":      version,
            "start_date":   self.now,
            "end_date":     None,
            "etl_batch_id": self.batch_id,
            "updated_at":   self.now,
            "created_at":   rec.get("created_at", self.now),
            "_hashes":      hashes,
        })
        self.col.insert_one(rec)

    def _store_diff(self, diff: FieldDiff, old_hashes: dict, new_hashes: dict) -> None:
        def _fc(fc):
            return {
                "path":        fc.path,
                "level":       fc.level,
                "old_hash":    fc.old_hash,
                "new_hash":    fc.new_hash,
                "old_value":   str(fc.old_value) if fc.old_value is not None else None,
                "new_value":   str(fc.new_value) if fc.new_value is not None else None,
                "change_type": fc.change_type,
            }

        def _ac(ac):
            return {
                "account_number": ac["account_number"],
                "change_type":    ac["change_type"],
                "field_changes":  [_fc(fc) for fc in ac.get("field_changes", [])],
            }

        self.diffs.insert_one({
            "client_id":             diff.client_id,
            "batch_id":              self.batch_id,
            "run_date":              self.now,
            "old_version":           diff.old_version,
            "new_version":           diff.new_version,
            "document_changed":      diff.document_changed,
            "changed_sections":      diff.changed_sections,
            "field_changes":         [_fc(fc) for fc in diff.field_changes],
            "account_changes":       [_ac(ac) for ac in diff.account_changes],
            "total_field_changes":   len(diff.field_changes),
            "total_account_changes": len(diff.account_changes),
            "old_document_hash":     old_hashes.get("document"),
            "new_document_hash":     new_hashes.get("document"),
        })

    def process_record(self, incoming: dict) -> None:
        self.stats["records_processed"] += 1
        client_id  = incoming["client_id"]
        new_hashes = build_hash_manifest(incoming)
        existing   = self._get_active(client_id)

        if existing is None:
            self._insert_version(incoming, version=1, hashes=new_hashes)
            self.stats["records_inserted"] += 1
            log.debug("  INSERT  %s", client_id)
            return

        old_hashes = existing.get("_hashes") or build_hash_manifest(existing)

        if old_hashes.get("document") == new_hashes.get("document"):
            self.stats["records_unchanged"] += 1
            log.debug("  SKIP    %s  (no changes)", client_id)
            return

        diff        = diff_manifests(existing, incoming, old_hashes, new_hashes)
        new_version = existing["version"] + 1

        self._archive(existing)
        self._insert_version(incoming, version=new_version, hashes=new_hashes)
        self._store_diff(diff, old_hashes, new_hashes)

        self.stats["records_updated"]  += 1
        self.stats["fields_changed"]   += len(diff.field_changes)
        self.stats["sections_changed"] += len(diff.changed_sections)

        log.info(
            "  UPDATE  %-10s  v%d→v%d  sections=%s  fields=%s",
            client_id, existing["version"], new_version,
            diff.changed_sections,
            [fc.path for fc in diff.field_changes],
        )

    def deactivate_removed(self, source_ids: set[str]) -> None:
        active_ids = {
            d["client_id"]
            for d in self.col.find({"is_active": True}, {"client_id": 1})
        }
        removed = active_ids - source_ids
        if removed:
            res = self.col.update_many(
                {"client_id": {"$in": list(removed)}, "is_active": True},
                {"$set": {"is_active": False, "end_date": self.now}},
            )
            self.stats["records_archived"] += res.modified_count
            log.info("Deactivated %d removed clients.", res.modified_count)

    def run(self, records: list[dict]) -> None:
        log.info("ETL batch %s — %d records", self.batch_id, len(records))
        t0 = time.time()

        for rec in records:
            try:
                self.process_record(rec)
            except Exception as exc:
                msg = f"{rec.get('client_id','?')}: {exc}"
                self.stats["errors"].append(msg)
                log.error("  ERROR  %s", msg)

        self.deactivate_removed({r["client_id"] for r in records})
        duration = round(time.time() - t0, 3)
        self._write_audit(duration)
        self._print_summary(duration)

    def _write_audit(self, duration: float) -> None:
        self.audit.insert_one({
            "batch_id":               self.batch_id,
            "run_date":               self.now,
            "status":                 "failed" if self.stats["errors"] else "success",
            "records_processed":      self.stats["records_processed"],
            "records_inserted":       self.stats["records_inserted"],
            "records_updated":        self.stats["records_updated"],
            "records_archived":       self.stats["records_archived"],
            "total_fields_changed":   self.stats["fields_changed"],
            "total_sections_changed": self.stats["sections_changed"],
            "errors":                 self.stats["errors"] or None,
            "duration_seconds":       duration,
        })

    def _print_summary(self, duration: float) -> None:
        s = self.stats
        log.info(
            "ETL complete %.2fs | processed=%d inserted=%d updated=%d "
            "archived=%d unchanged=%d fields_changed=%d errors=%d",
            duration,
            s["records_processed"], s["records_inserted"], s["records_updated"],
            s["records_archived"],  s["records_unchanged"],
            s["fields_changed"],    len(s["errors"]),
        )


def print_diff_report(db, client_id: str) -> None:
    versions  = list(db["banking_clients"].find({"client_id": client_id}).sort("version", 1))
    diff_docs = list(db["etl_field_diffs"].find({"client_id": client_id}).sort("new_version", 1))

    if not versions:
        print(f"No records found for {client_id}")
        return

    print(f"\n{'='*70}")
    print(f"  DIFF REPORT  {client_id}  ({len(versions)} version(s))")
    print(f"{'='*70}")

    if not diff_docs:
        print("  No diffs recorded.")
        return

    for d in diff_docs:
        print(f"\n  v{d['old_version']} -> v{d['new_version']}  batch={d['batch_id'][:8]}  {d['run_date'][:19]}")
        print(f"  doc hash: {(d['old_document_hash'] or '')[:20]}... -> {(d['new_document_hash'] or '')[:20]}...")
        print(f"  changed sections: {d['changed_sections']}")
        for fc in d.get("field_changes", []):
            print(f"    FIELD {fc['path']}")
            print(f"          hash {(fc['old_hash'] or '')[:16]}... -> {(fc['new_hash'] or '')[:16]}...")
            print(f"          old: {fc['old_value']}  |  new: {fc['new_value']}")
        for ac in d.get("account_changes", []):
            print(f"    ACCOUNT [{ac['account_number']}] {ac['change_type'].upper()}")
            for afc in ac.get("field_changes", []):
                fn = afc['path'].split('.')[-1]
                print(f"      {fn}: {afc['old_value']} -> {afc['new_value']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Banking ETL Pipeline v2")
    #parser.add_argument("--source",           default="data/clients.json")
    parser.add_argument("--source",           default="data\\clients_data")
    parser.add_argument("--simulate-changes", action="store_true")
    parser.add_argument("--report",           metavar="CLIENT_ID")
    args = parser.parse_args()

    mongo = MongoClient(MONGO_URI)
    db    = mongo[DB_NAME]
    setup_collections(db)

    if args.report:
        print_diff_report(db, args.report.upper())
        mongo.close()
        return

    root    = os.path.dirname(os.path.dirname(__file__))
    #print("Loading source data...",root)
    records = load_source(os.path.join(root, args.source))
    #print(records)

    if args.simulate_changes:
        log.info("Simulating changes on ~25%% of records (1-3 fields each)...")
        records = simulate_changes(records, change_ratio=0.25)

    ETLPipeline(db).run(records)
    mongo.close()


if __name__ == "__main__":
    main()
