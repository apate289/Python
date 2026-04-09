"""
Data Access Layer — all MongoDB queries used by the Streamlit app.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime
from typing import Any
from pymongo import MongoClient
from config.settings import MONGO_URI, DB_NAME


# ── singleton connection ──────────────────────
_client: MongoClient | None = None

def get_db():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client[DB_NAME]


# ── helpers ───────────────────────────────────
def _clients(db): return db["banking_clients"]
def _audit(db):   return db["etl_audit_log"]


# ═════════════════════════════════════════════
#  KPI / dashboard
# ═════════════════════════════════════════════
def get_kpis(db) -> dict:
    col = _clients(db)
    pipeline = [
        {"$match": {"is_active": True}},
        {"$group": {
            "_id": None,
            "total_clients":  {"$sum": 1},
            "kyc_verified":   {"$sum": {"$cond": ["$banking_profile.kyc_verified", 1, 0]}},
            "pep_flagged":    {"$sum": {"$cond": ["$banking_profile.pep_flag", 1, 0]}},
            "avg_credit":     {"$avg": "$banking_profile.credit_score"},
            "total_accounts": {"$sum": {"$size": "$accounts"}},
        }},
        {"$project": {
            "_id": 0,
            "total_clients": 1, "kyc_verified": 1, "pep_flagged": 1,
            "avg_credit": {"$round": ["$avg_credit", 0]},
            "total_accounts": 1,
        }}
    ]
    result = list(col.aggregate(pipeline))
    if not result:
        return {"total_clients": 0, "kyc_verified": 0, "pep_flagged": 0,
                "avg_credit": 0, "total_accounts": 0}
    kpis = result[0]

    # total balance across all active accounts
    bal_pipe = [
        {"$match": {"is_active": True}},
        {"$unwind": "$accounts"},
        {"$match": {"accounts.status": "Active"}},
        {"$group": {"_id": None, "total_balance": {"$sum": "$accounts.balance"}}},
    ]
    bal = list(col.aggregate(bal_pipe))
    kpis["total_balance"] = round(bal[0]["total_balance"], 2) if bal else 0
    return kpis


def get_account_type_distribution(db) -> list[dict]:
    pipeline = [
        {"$match": {"is_active": True}},
        {"$unwind": "$accounts"},
        {"$group": {"_id": "$accounts.account_type", "count": {"$sum": 1},
                    "total_balance": {"$sum": "$accounts.balance"}}},
        {"$sort": {"total_balance": -1}},
    ]
    return list(_clients(db).aggregate(pipeline))


def get_credit_score_buckets(db) -> list[dict]:
    pipeline = [
        {"$match": {"is_active": True}},
        {"$bucket": {
            "groupBy": "$banking_profile.credit_score",
            "boundaries": [300, 580, 670, 740, 800, 851],
            "default": "Other",
            "output": {"count": {"$sum": 1}},
        }},
    ]
    labels = {300: "Poor (300-579)", 580: "Fair (580-669)",
              670: "Good (670-739)", 740: "Very Good (740-799)", 800: "Excellent (800-850)"}
    rows = list(_clients(db).aggregate(pipeline))
    for r in rows:
        r["label"] = labels.get(r["_id"], str(r["_id"]))
    return rows


def get_state_distribution(db) -> list[dict]:
    pipeline = [
        {"$match": {"is_active": True}},
        {"$group": {"_id": "$address.state", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    return list(_clients(db).aggregate(pipeline))


def get_branch_summary(db) -> list[dict]:
    pipeline = [
        {"$match": {"is_active": True}},
        {"$group": {
            "_id": "$banking_profile.branch_id",
            "client_count": {"$sum": 1},
            "avg_credit":   {"$avg": "$banking_profile.credit_score"},
        }},
        {"$sort": {"client_count": -1}},
        {"$limit": 15},
    ]
    return list(_clients(db).aggregate(pipeline))


def get_version_history_stats(db) -> list[dict]:
    """Count of records by version number (active + historical)."""
    pipeline = [
        {"$group": {"_id": "$version", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    return list(_clients(db).aggregate(pipeline))


# ═════════════════════════════════════════════
#  Filtered client listing
# ═════════════════════════════════════════════
def get_clients(
    db,
    is_active: bool = True,
    state: str | None = None,
    kyc: bool | None = None,
    pep: bool | None = None,
    min_credit: int = 300,
    max_credit: int = 850,
    account_type: str | None = None,
    account_status: str | None = None,
    search: str | None = None,
    branch_id: str | None = None,
    skip: int = 0,
    limit: int = 25,
) -> tuple[list[dict], int]:
    query: dict[str, Any] = {"is_active": is_active}

    if state:               query["address.state"]                    = state
    if kyc is not None:     query["banking_profile.kyc_verified"]     = kyc
    if pep is not None:     query["banking_profile.pep_flag"]         = pep
    if branch_id:           query["banking_profile.branch_id"]        = branch_id
    if account_type:        query["accounts.account_type"]            = account_type
    if account_status:      query["accounts.status"]                  = account_status

    query["banking_profile.credit_score"] = {"$gte": min_credit, "$lte": max_credit}

    if search:
        query["$or"] = [
            {"client_id": {"$regex": search, "$options": "i"}},
            {"personal_information.first_name": {"$regex": search, "$options": "i"}},
            {"personal_information.last_name":  {"$regex": search, "$options": "i"}},
            {"personal_information.email":      {"$regex": search, "$options": "i"}},
        ]

    col = _clients(db)
    total = col.count_documents(query)
    docs = list(col.find(query, {"_id": 0}).skip(skip).limit(limit))
    return docs, total


# ═════════════════════════════════════════════
#  Single client + version history
# ═════════════════════════════════════════════
def get_client_by_id(db, client_id: str) -> dict | None:
    return _clients(db).find_one({"client_id": client_id, "is_active": True}, {"_id": 0})


def get_client_version_history(db, client_id: str) -> list[dict]:
    return list(
        _clients(db)
        .find({"client_id": client_id}, {"_id": 0})
        .sort("version", -1)
    )


# ═════════════════════════════════════════════
#  ETL Audit log
# ═════════════════════════════════════════════
def get_audit_logs(db, limit: int = 20) -> list[dict]:
    return list(_audit(db).find({}, {"_id": 0}).sort("run_date", -1).limit(limit))


def get_filter_options(db) -> dict:
    col = _clients(db)
    states   = sorted(col.distinct("address.state"))
    branches = sorted(col.distinct("banking_profile.branch_id"))
    acct_types = [
        "Checking","Savings","Business Checking","Business Savings",
        "CD","Trust","Money Market","IRA"
    ]
    return {"states": states, "branches": branches, "account_types": acct_types}
