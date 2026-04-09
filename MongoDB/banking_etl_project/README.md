# 🏦 Banking Client Intelligence — MongoDB + Python ETL + Streamlit

A **production-grade** data engineering project that demonstrates:

- **Complex nested JSON** modeling for banking clients
- **MongoDB** with JSON Schema validation, compound indexes, and PyMongo
- **SCD Type 2 (Slowly Changing Dimension)** versioning via `is_active`, `start_date`, `end_date`, `version`
- **Daily ETL pipeline** with insert / update (archive + re-insert) / deactivate logic
- **Streamlit dashboard** with real-time querying, filtering, and version history exploration

---

## 📁 Project Structure

```
banking_etl/
├── config/
│   └── settings.py           # Env-based configuration
├── data/
│   ├── generate_data.py      # Faker-based JSON generator (50 clients)
│   └── clients.json          # Generated source data
├── schemas/
│   └── mongo_schema.py       # MongoDB validators + index definitions
├── etl/
│   └── pipeline.py           # Daily ETL — SCD Type 2 versioning
├── app/
│   ├── queries.py            # Data access layer (all MongoDB queries)
│   └── streamlit_app.py      # Streamlit dashboard
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- MongoDB 6.0+ running locally (`mongod`) or Atlas connection string

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env — set MONGO_URI if not localhost
```

### 4. Generate dummy data
```bash
python data/generate_data.py
# → data/clients.json  (50 banking clients)
```

### 5. Run initial ETL load
```bash
python etl/pipeline.py
# Creates collections, applies validators, indexes, and loads all 50 records
```

### 6. Simulate daily changes and re-run ETL
```bash
python etl/pipeline.py --simulate-changes
# Randomly mutates ~20% of records
# Old versions archived (is_active=False), new versions inserted (version++)
```

### 7. Launch Streamlit dashboard
```bash
streamlit run app/streamlit_app.py
# → http://localhost:8501
```

---

## 🗄️ MongoDB Schema

### Collection: `banking_clients`

| Field | Type | Description |
|---|---|---|
| `client_id` | String | Unique identifier (`CLTxxxxx`) |
| `personal_information` | Object | Name, DOB, SSN last 4, email, phones |
| `address` | Object | Street, city, state, zip, country |
| `banking_profile` | Object | Credit score, KYC, PEP, branch, RM |
| `accounts` | Array | 1–5 accounts (type, balance, status, etc.) |
| `check_printing` | Object | Auth, style, last order |
| **`is_active`** | Boolean | `true` = current record |
| **`version`** | Int | Starts at 1, increments on each change |
| **`start_date`** | ISO String | When this version became active |
| **`end_date`** | ISO String / null | When this version was archived |
| `etl_batch_id` | UUID | Which ETL run created this version |
| `created_at` | ISO String | Original record creation |
| `updated_at` | ISO String | Last ETL write |

### Collection: `etl_audit_log`

Tracks every ETL run with counts of inserted / updated / archived records, errors, and duration.

---

## 🔄 ETL Versioning Logic (SCD Type 2)

```
Incoming record
     │
     ├── No existing active record?   → INSERT  version=1, is_active=True
     │
     ├── Existing active record found:
     │       │
     │       ├── Fingerprint matches?  → SKIP  (no change)
     │       │
     │       └── Fingerprint differs?  → ARCHIVE old (is_active=False, end_date=now)
     │                                   INSERT  new  (version+1, is_active=True)
     │
     └── Client in DB but NOT in source file?
                                       → DEACTIVATE (is_active=False, end_date=now)
```

---

## 📊 Streamlit Dashboard Pages

| Page | Description |
|---|---|
| **Dashboard** | KPI cards, account distribution, credit score buckets, state choropleth, branch performance, version distribution |
| **Client Explorer** | Paginated, filterable table of clients with drill-down to full record |
| **Version History** | Search any client ID to see all versions with diffs |
| **ETL Audit Log** | Table + stacked bar chart of every ETL run |

### Sidebar Filters
- Active / Historical / All Versions toggle
- Free-text search (ID, name, email)
- State, Branch, Account Type, Account Status
- KYC / PEP flags
- Credit score range slider
- Page size control

---

## 🔧 Key Technical Choices

- **PyMongo** (not Motor) for simplicity; swap to Motor + `asyncio` for high-throughput writes
- **SHA-256 fingerprinting** for change detection — avoids field-by-field comparison
- **`collMod`** for idempotent validator updates — safe to re-run `setup_collections()`
- **Multi-key index** on `accounts.account_number` for sub-document queries
- **Compound unique index** `(client_id, version)` enforces versioning integrity
- **`@st.cache_resource`** for the MongoDB connection — one connection pool per Streamlit session

---

## 📌 Example Queries

```python
# Get active record
db.banking_clients.find_one({"client_id": "CLT00001", "is_active": True})

# Get full version history
db.banking_clients.find({"client_id": "CLT00001"}).sort("version", -1)

# All PEP-flagged active clients in TX
db.banking_clients.find({
    "is_active": True,
    "banking_profile.pep_flag": True,
    "address.state": "TX"
})

# Total balance by account type (active clients only)
db.banking_clients.aggregate([
    {"$match": {"is_active": True}},
    {"$unwind": "$accounts"},
    {"$group": {"_id": "$accounts.account_type", "total": {"$sum": "$accounts.balance"}}}
])
```
