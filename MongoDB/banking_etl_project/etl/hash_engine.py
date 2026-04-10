"""
Field-Level Hash Engine
=======================
Computes SHA-256 fingerprints at four granularity levels:

  1. document_hash   — entire record (excluding metadata)
  2. section_hashes  — one hash per top-level section
                       (personal_information, address, banking_profile,
                        accounts, check_printing)
  3. field_hashes    — one hash per leaf scalar field (dot-notation keys)
  4. account_hashes  — one hash per account in the accounts[] array

All hashes are stored inside the MongoDB document under a `_hashes` sub-document
so every version carries its own fingerprint manifest.  On the next ETL run the
engine compares old vs new manifests to produce a precise, field-by-field diff.

Public API
----------
  build_hash_manifest(record)  → dict   (the _hashes block to embed)
  diff_manifests(old, new)     → FieldDiff
  explain_diff(diff)           → list[str]   (human-readable change lines)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
#  Constants — sections that get individual hashes
# ─────────────────────────────────────────────────────────────────────────────
SECTIONS = [
    "personal_information",
    "address",
    "banking_profile",
    "accounts",
    "check_printing",
]

# Fields whose hash we track individually (dot-notation)
TRACKED_FIELDS = [
    # personal_information
    "personal_information.first_name",
    "personal_information.last_name",
    "personal_information.date_of_birth",
    "personal_information.ssn_last4",
    "personal_information.email",
    "personal_information.phone_primary",
    "personal_information.phone_secondary",
    # address
    "address.street",
    "address.city",
    "address.state",
    "address.zip_code",
    "address.country",
    # banking_profile
    "banking_profile.customer_since",
    "banking_profile.branch_id",
    "banking_profile.relationship_manager",
    "banking_profile.credit_score",
    "banking_profile.kyc_verified",
    "banking_profile.pep_flag",
    "banking_profile.preferred_contact",
    # check_printing
    "check_printing.authorized",
    "check_printing.check_style",
    "check_printing.last_check_order_date",
    "check_printing.starting_check_number",
]

# Metadata keys excluded from ALL hashes
_METADATA_KEYS = frozenset({
    "is_active", "version", "start_date", "end_date",
    "etl_batch_id", "created_at", "updated_at", "_id", "_hashes",
})


# ─────────────────────────────────────────────────────────────────────────────
#  Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────
def _sha256(value: Any) -> str:
    """Deterministic SHA-256 of any JSON-serialisable value."""
    canonical = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _get_dotted(record: dict, dotted_key: str) -> Any:
    """Safely retrieve a value using dot-notation (returns None if missing)."""
    parts = dotted_key.split(".")
    obj = record
    for p in parts:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(p)
    return obj


def _strip_metadata(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in _METADATA_KEYS}


# ─────────────────────────────────────────────────────────────────────────────
#  Hash manifest builder
# ─────────────────────────────────────────────────────────────────────────────
def build_hash_manifest(record: dict) -> dict:
    """
    Return a `_hashes` dict to be stored inside the MongoDB document.

    Structure
    ---------
    {
        "document":  "<sha256 of whole record minus metadata>",
        "sections": {
            "personal_information": "<sha256>",
            "address":              "<sha256>",
            "banking_profile":      "<sha256>",
            "accounts":             "<sha256>",
            "check_printing":       "<sha256>",
        },
        "fields": {
            "personal_information.email":        "<sha256>",
            "banking_profile.credit_score":      "<sha256>",
            ...  (one entry per TRACKED_FIELDS)
        },
        "accounts": [
            {
                "account_number": "CHK1234567890",
                "hash":           "<sha256 of this account object>",
                "field_hashes": {
                    "balance":      "<sha256>",
                    "status":       "<sha256>",
                    "interest_rate":"<sha256>",
                    ...
                }
            },
            ...
        ]
    }
    """
    clean = _strip_metadata(record)

    # 1 ── document-level hash
    doc_hash = _sha256(clean)

    # 2 ── section hashes
    section_hashes: dict[str, str] = {}
    for sec in SECTIONS:
        val = clean.get(sec)
        section_hashes[sec] = _sha256(val) if val is not None else _sha256(None)

    # 3 ── field hashes (leaf scalars via dot-notation)
    field_hashes: dict[str, str] = {}
    for dotted in TRACKED_FIELDS:
        val = _get_dotted(clean, dotted)
        field_hashes[dotted] = _sha256(val)

    # 4 ── per-account hashes
    account_hashes: list[dict] = []
    for acct in clean.get("accounts", []):
        acct_field_hashes = {
            k: _sha256(acct.get(k))
            for k in [
                "account_number", "routing_number", "account_type",
                "holding_name", "balance", "currency", "status",
                "date_opened", "interest_rate", "overdraft_protection",
            ]
        }
        account_hashes.append({
            "account_number": acct.get("account_number", "UNKNOWN"),
            "hash":           _sha256(acct),
            "field_hashes":   acct_field_hashes,
        })

    return {
        "document": doc_hash,
        "sections": section_hashes,
        "fields":   field_hashes,
        "accounts": account_hashes,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Diff engine
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FieldChange:
    path:      str
    level:     str          # "document" | "section" | "field" | "account_field"
    old_hash:  str | None
    new_hash:  str | None
    old_value: Any = None
    new_value: Any = None
    change_type: str = "modified"   # "modified" | "added" | "removed"


@dataclass
class FieldDiff:
    client_id:        str
    old_version:      int
    new_version:      int
    document_changed: bool = False
    changed_sections: list[str]    = field(default_factory=list)
    field_changes:    list[FieldChange] = field(default_factory=list)
    account_changes:  list[dict]   = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        return len(self.field_changes) + len(self.account_changes)

    @property
    def is_unchanged(self) -> bool:
        return not self.document_changed


def diff_manifests(
    old_record:  dict,
    new_record:  dict,
    old_manifest: dict,
    new_manifest: dict,
) -> FieldDiff:
    """
    Compare two hash manifests and return a structured FieldDiff.
    Also attaches old/new values to each FieldChange for display.
    """
    cid         = new_record.get("client_id", "?")
    old_ver     = old_record.get("version", 0)
    new_ver     = new_record.get("version", old_ver + 1)
    diff        = FieldDiff(client_id=cid, old_version=old_ver, new_version=new_ver)

    # ── document level ────────────────────────────────────────────────────────
    diff.document_changed = old_manifest.get("document") != new_manifest.get("document")
    if not diff.document_changed:
        return diff   # nothing changed at all — fast path

    # ── section level ─────────────────────────────────────────────────────────
    old_sec = old_manifest.get("sections", {})
    new_sec = new_manifest.get("sections", {})
    for sec in SECTIONS:
        if old_sec.get(sec) != new_sec.get(sec):
            diff.changed_sections.append(sec)

    # ── field level ───────────────────────────────────────────────────────────
    old_fld = old_manifest.get("fields", {})
    new_fld = new_manifest.get("fields", {})
    for dotted in TRACKED_FIELDS:
        oh = old_fld.get(dotted)
        nh = new_fld.get(dotted)
        if oh != nh:
            diff.field_changes.append(FieldChange(
                path=dotted,
                level="field",
                old_hash=oh,
                new_hash=nh,
                old_value=_get_dotted(old_record, dotted),
                new_value=_get_dotted(new_record, dotted),
                change_type="modified",
            ))

    # ── account level ─────────────────────────────────────────────────────────
    old_accts = {a["account_number"]: a for a in old_manifest.get("accounts", [])}
    new_accts = {a["account_number"]: a for a in new_manifest.get("accounts", [])}

    old_acct_data = {a["account_number"]: a for a in old_record.get("accounts", [])}
    new_acct_data = {a["account_number"]: a for a in new_record.get("accounts", [])}

    all_acct_nums = set(old_accts) | set(new_accts)
    for acct_num in sorted(all_acct_nums):
        old_a = old_accts.get(acct_num)
        new_a = new_accts.get(acct_num)

        if old_a is None:
            diff.account_changes.append({
                "account_number": acct_num,
                "change_type":    "added",
                "field_changes":  [],
            })
            continue

        if new_a is None:
            diff.account_changes.append({
                "account_number": acct_num,
                "change_type":    "removed",
                "field_changes":  [],
            })
            continue

        if old_a["hash"] == new_a["hash"]:
            continue   # account unchanged

        # compare field-level hashes within the account
        acct_field_changes = []
        old_fh = old_a.get("field_hashes", {})
        new_fh = new_a.get("field_hashes", {})
        old_data = old_acct_data.get(acct_num, {})
        new_data = new_acct_data.get(acct_num, {})

        for fld_name in set(old_fh) | set(new_fh):
            oh = old_fh.get(fld_name)
            nh = new_fh.get(fld_name)
            if oh != nh:
                acct_field_changes.append(FieldChange(
                    path=f"accounts[{acct_num}].{fld_name}",
                    level="account_field",
                    old_hash=oh,
                    new_hash=nh,
                    old_value=old_data.get(fld_name),
                    new_value=new_data.get(fld_name),
                    change_type="modified",
                ))

        diff.account_changes.append({
            "account_number": acct_num,
            "change_type":    "modified",
            "field_changes":  acct_field_changes,
        })

    return diff


# ─────────────────────────────────────────────────────────────────────────────
#  Human-readable explanation
# ─────────────────────────────────────────────────────────────────────────────
def explain_diff(diff: FieldDiff) -> list[str]:
    """Return a list of plain-English change descriptions."""
    if diff.is_unchanged:
        return ["No changes detected."]

    lines = []
    lines.append(f"Client {diff.client_id}: v{diff.old_version} → v{diff.new_version}")
    lines.append(f"  Changed sections: {', '.join(diff.changed_sections) or 'none'}")

    for fc in diff.field_changes:
        lines.append(f"  FIELD  {fc.path}")
        lines.append(f"         old: {fc.old_value!r}")
        lines.append(f"         new: {fc.new_value!r}")
        lines.append(f"         hash {fc.old_hash[:12]}… → {fc.new_hash[:12]}…")

    for ac in diff.account_changes:
        ct = ac["change_type"].upper()
        lines.append(f"  ACCOUNT [{ac['account_number']}] {ct}")
        for fc in ac.get("field_changes", []):
            lines.append(f"    FIELD  {fc.path.split('.')[-1]}")
            lines.append(f"           old: {fc.old_value!r}")
            lines.append(f"           new: {fc.new_value!r}")

    return lines
