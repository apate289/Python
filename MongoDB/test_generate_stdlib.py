"""
Stdlib-only data generator (no Faker) — for environments without pip access.
The main generate_data.py uses Faker for richer names/addresses.
"""
import json, random, uuid
from datetime import date, datetime, timedelta

random.seed(42)

FIRST_NAMES = ["Michael","Sarah","James","Emily","Robert","Jessica","William","Ashley","David","Amanda",
               "Richard","Stephanie","Thomas","Nicole","Charles","Elizabeth","Christopher","Jennifer",
               "Daniel","Linda","Matthew","Barbara","Anthony","Susan","Mark","Patricia","Donald","Mary"]
LAST_NAMES  = ["Lewis","Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Wilson",
               "Martinez","Anderson","Taylor","Thomas","Moore","Jackson","Martin","Lee","Perez","White"]
STREETS     = ["Oak Ave","Main St","Lake View Dr","Elm Blvd","Park Rd","Maple Dr","Cedar Ln","Pine St","River Rd","Valley Way"]
CITIES      = ["Dallas","Austin","Houston","Denver","Atlanta","Chicago","Phoenix","Seattle","Boston","Miami"]
STATES      = ["TX","CA","NY","FL","IL","PA","OH","GA","NC","MI"]
COMPANIES   = ["Lewis & Associates LLC","Smith Holdings","Johnson Family Trust","Williams Enterprises","Brown Capital"]
BRANCHES    = [f"BR{str(i).zfill(3)}" for i in range(1, 40)]
ACCT_TYPES  = ["Checking","Savings","Business Checking","Business Savings","CD","Trust","Money Market","IRA"]
ACCT_PFXS   = {"Checking":"CHK","Savings":"SAV","Business Checking":"BCK","Business Savings":"BSA",
               "CD":"CDA","Trust":"TRU","Money Market":"MMA","IRA":"IRA"}
STATUSES    = ["Active","Active","Active","Active","Dormant","Closed","Frozen"]
MANAGERS    = ["Daniel Torres","Maria Rodriguez","James Park","Lisa Chen","Ahmed Hassan","Priya Patel"]

def rdate(s=1990, e=2023):
    d0 = date(s,1,1); d1 = date(e,12,31)
    return (d0 + timedelta(days=random.randint(0,(d1-d0).days))).isoformat()

def acct(t):
    px = ACCT_PFXS[t]
    return {
        "account_number": px+"".join(str(random.randint(0,9)) for _ in range(10)),
        "routing_number": "".join(str(random.randint(0,9)) for _ in range(9)),
        "account_type": t,
        "holding_name": random.choice(COMPANIES) if "Business" in t or t in ["Trust","IRA"] else f"{random.choice(LAST_NAMES)} Family",
        "balance": round(random.uniform(500,250000),2),
        "currency": random.choices(["USD","EUR","GBP","CAD"],[85,7,5,3])[0],
        "status": random.choice(STATUSES),
        "date_opened": rdate(1995,2022),
        "interest_rate": round(random.uniform(0.01,5.5),2),
        "overdraft_protection": random.choice([True,False]),
    }

def gen(i):
    fn = random.choice(FIRST_NAMES); ln = random.choice(LAST_NAMES)
    return {
        "client_id": f"CLT{str(i).zfill(5)}",
        "personal_information": {
            "first_name": fn, "last_name": ln,
            "date_of_birth": rdate(1940,2000),
            "ssn_last4": str(random.randint(1000,9999)),
            "email": f"{fn.lower()}.{ln.lower()}{random.randint(1,99)}@{random.choice(['gmail.com','outlook.com','yahoo.com'])}",
            "phone_primary": f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}",
            "phone_secondary": f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}" if random.random()>0.3 else None,
        },
        "address": {
            "street": f"{random.randint(100,9999)} {random.choice(STREETS)}",
            "city": random.choice(CITIES), "state": random.choice(STATES),
            "zip_code": str(random.randint(10000,99999)), "country": "USA",
        },
        "banking_profile": {
            "customer_since": rdate(2000,2023),
            "branch_id": random.choice(BRANCHES),
            "relationship_manager": random.choice(MANAGERS),
            "credit_score": random.randint(300,850),
            "kyc_verified": random.choices([True,False],[85,15])[0],
            "pep_flag": random.choices([True,False],[3,97])[0],
            "preferred_contact": random.choice(["Email","Phone","Mail","SMS"]),
        },
        "accounts": [acct(t) for t in random.sample(ACCT_TYPES, random.randint(1,4))],
        "check_printing": {
            "authorized": random.choices([True,False],[60,40])[0],
            "check_style": random.choice(["High Security","Standard","Business","Personal"]),
            "last_check_order_date": rdate(1990,2023),
            "starting_check_number": random.randint(1000,9999),
        },
        "is_active": True, "version": 1,
        "start_date": datetime.utcnow().isoformat(), "end_date": None,
        "etl_batch_id": str(uuid.uuid4()),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }

data = [gen(i+1) for i in range(50)]
with open("/home/claude/banking_etl/data/clients.json","w") as f:
    json.dump(data, f, indent=2)
print(f"Generated {len(data)} records")
print("Sample:", json.dumps(data[20], indent=2)[:600])
