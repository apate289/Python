"""
MongoDB Schema Definitions using PyMongo
Includes JSON Schema validation, indexes, and versioning support
"""

# ─────────────────────────────────────────────
#  Collection: banking_clients
# ─────────────────────────────────────────────
BANKING_CLIENTS_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["client_id", "personal_information", "address", "banking_profile",
                     "accounts", "is_active", "version", "start_date"],
        "additionalProperties": True,
        "properties": {
            "client_id": {
                "bsonType": "string",
                "pattern": "^[a-zA-Z0-9]+(_[a-zA-Z0-9]+)*$",
                "description": "Unique client identifier — required, format CLTxxxxx"
            },
            "personal_information": {
                "bsonType": "object",
                "required": ["first_name", "last_name", "date_of_birth", "ssn_last4", "email"],
                "properties": {
                    "first_name":  {"bsonType": "string", "minLength": 1, "maxLength": 100},
                    "last_name":   {"bsonType": "string", "minLength": 1, "maxLength": 100},
                    "date_of_birth": {
                        "bsonType": "string",
                        "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
                    },
                    "ssn_last4": {
                        "bsonType": "string",
                        "pattern": "^[0-9]{4}$"
                    },
                    "email": {
                        "bsonType": "string",
                        "pattern": "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"
                    },
                    "phone_primary":   {"bsonType": ["string", "null"]},
                    "phone_secondary": {"bsonType": ["string", "null"]},
                }
            },
            "address": {
                "bsonType": "object",
                "required": ["street", "city", "state", "zip_code", "country"],
                "properties": {
                    "street":   {"bsonType": "string"},
                    "city":     {"bsonType": "string"},
                    "state":    {"bsonType": "string", "minLength": 2, "maxLength": 2},
                    "zip_code": {"bsonType": "string"},
                    "country":  {"bsonType": "string"},
                }
            },
            "banking_profile": {
                "bsonType": "object",
                "required": ["customer_since", "branch_id", "credit_score", "kyc_verified"],
                "properties": {
                    "customer_since":      {"bsonType": "string"},
                    "branch_id":           {"bsonType": "string"},
                    "relationship_manager":{"bsonType": ["string", "null"]},
                    "credit_score": {
                        "bsonType": "int",
                        "minimum": 300,
                        "maximum": 850,
                    },
                    "kyc_verified": {"bsonType": "bool"},
                    "pep_flag":     {"bsonType": "bool"},
                    "preferred_contact": {
                        "bsonType": "string",
                        "enum": ["Email", "Phone", "Mail", "SMS"]
                    },
                }
            },
            "accounts": {
                "bsonType": "array",
                "minItems": 1,
                "items": {
                    "bsonType": "object",
                    "required": ["account_number", "routing_number", "account_type",
                                 "balance", "currency", "status", "date_opened"],
                    "properties": {
                        "account_number": {"bsonType": "string"},
                        "routing_number": {"bsonType": "string", "minLength": 9, "maxLength": 9},
                        "account_type": {
                            "bsonType": "string",
                            "enum": ["Checking","Savings","Business Checking","Business Savings",
                                     "CD","Trust","Money Market","IRA"]
                        },
                        "holding_name": {"bsonType": ["string","null"]},
                        "balance":      {"bsonType": ["double","int"], "minimum": 0},
                        "currency":     {"bsonType": "string", "enum": ["USD","EUR","GBP","CAD"]},
                        "status": {
                            "bsonType": "string",
                            "enum": ["Active","Dormant","Closed","Frozen"]
                        },
                        "date_opened":         {"bsonType": "string"},
                        "interest_rate":       {"bsonType": ["double","int","null"]},
                        "overdraft_protection":{"bsonType": ["bool","null"]},
                    }
                }
            },
            "check_printing": {
                "bsonType": ["object","null"],
                "properties": {
                    "authorized":            {"bsonType": "bool"},
                    "check_style":           {"bsonType": "string"},
                    "last_check_order_date": {"bsonType": ["string","null"]},
                    "starting_check_number": {"bsonType": ["int","null"]},
                }
            },
            # ── Versioning fields ──────────────────
            "is_active":    {"bsonType": "bool"},
            "version":      {"bsonType": "int", "minimum": 1},
            "start_date":   {"bsonType": "string"},
            "end_date":     {"bsonType": ["string","null"]},
            "etl_batch_id": {"bsonType": ["string","null"]},
            "created_at":   {"bsonType": "string"},
            "updated_at":   {"bsonType": "string"},
        }
    }
}

# ─────────────────────────────────────────────
#  Collection: etl_audit_log
# ─────────────────────────────────────────────
ETL_AUDIT_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["batch_id", "run_date", "status", "records_processed"],
        "properties": {
            "batch_id":          {"bsonType": "string"},
            "run_date":          {"bsonType": "string"},
            "status":            {"bsonType": "string", "enum": ["started","success","failed","partial"]},
            "records_processed": {"bsonType": "int"},
            "records_inserted":  {"bsonType": ["int","null"]},
            "records_updated":   {"bsonType": ["int","null"]},
            "records_archived":  {"bsonType": ["int","null"]},
            "errors":            {"bsonType": ["array","null"]},
            "duration_seconds":  {"bsonType": ["double","null"]},
        }
    }
}


def create_indexes(db):
    """Create all necessary indexes on collections."""
    clients = db["banking_clients"]

    # Compound unique: client_id + version (SCD Type 2 support)
    clients.create_index(
        [("client_id", 1), ("version", -1)],
        unique=True,
        name="idx_client_version_unique"
    )
    # Fast lookup for active record per client
    clients.create_index(
        [("client_id", 1), ("is_active", 1)],
        name="idx_client_active"
    )
    # Date range queries for versioning
    clients.create_index(
        [("start_date", 1), ("end_date", 1)],
        name="idx_version_dates"
    )
    # Account-level lookups (multi-key index on embedded array)
    clients.create_index(
        [("accounts.account_number", 1)],
        name="idx_account_number"
    )
    clients.create_index(
        [("accounts.status", 1)],
        name="idx_account_status"
    )
    # Profile filters
    clients.create_index([("banking_profile.credit_score", 1)],   name="idx_credit_score")
    clients.create_index([("banking_profile.kyc_verified", 1)],   name="idx_kyc")
    clients.create_index([("banking_profile.pep_flag", 1)],       name="idx_pep")
    clients.create_index([("banking_profile.branch_id", 1)],      name="idx_branch")
    clients.create_index([("address.state", 1)],                  name="idx_state")
    clients.create_index([("etl_batch_id", 1)],                   name="idx_batch")

    # Audit log
    db["etl_audit_log"].create_index([("batch_id", 1)], unique=True, name="idx_audit_batch")
    db["etl_audit_log"].create_index([("run_date", -1)],            name="idx_audit_date")

    # Field diffs
    diffs = db["etl_field_diffs"]
    diffs.create_index([("client_id", 1), ("new_version", -1)],  name="idx_diff_client_ver")
    diffs.create_index([("batch_id",   1)],                       name="idx_diff_batch")
    diffs.create_index([("run_date",   -1)],                      name="idx_diff_date")
    diffs.create_index([("field_changes.path", 1)],               name="idx_diff_field_path")
    diffs.create_index([("changed_sections",   1)],               name="idx_diff_sections")

    print("✅ All indexes created.")


def setup_collections(db):
    """Create collections with schema validation (idempotent)."""
    existing = db.list_collection_names()

    for name, validator in [
        ("banking_clients", BANKING_CLIENTS_VALIDATOR),
        ("etl_audit_log",   ETL_AUDIT_VALIDATOR),
        ("etl_field_diffs", {}),   # schemaless — flexible diff payloads
    ]:
        if name not in existing:
            if validator:
                db.create_collection(name, validator=validator)
            else:
                db.create_collection(name)
            print(f"✅ Collection '{name}' created.")
        elif validator:
            db.command("collMod", name, validator=validator)
            print(f"♻️  Collection '{name}' validator updated.")
        else:
            print(f"✔  Collection '{name}' already exists.")

    create_indexes(db)
