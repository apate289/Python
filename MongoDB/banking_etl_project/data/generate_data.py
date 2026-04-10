"""
Generate production-grade dummy banking client JSON data
"""
import json
import random
import uuid
from datetime import datetime, timedelta, date
from faker import Faker

fake = Faker()
random.seed(42)

ACCOUNT_TYPES = ["Checking", "Savings", "Business Checking", "Business Savings", "CD", "Trust", "Money Market", "IRA"]
ACCOUNT_PREFIXES = {
    "Checking": "CHK", "Savings": "SAV", "Business Checking": "BCK",
    "Business Savings": "BSA", "CD": "CDA", "Trust": "TRU",
    "Money Market": "MMA", "IRA": "IRA"
}
STATUSES = ["Active", "Dormant", "Closed", "Frozen"]
CHECK_STYLES = ["High Security", "Standard", "Business", "Personal", "Laser"]
PREFERRED_CONTACTS = ["Email", "Phone", "Mail", "SMS"]
STATES = ["TX", "CA", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
CURRENCIES = ["USD", "EUR", "GBP", "CAD"]

def random_date(start_year=1990, end_year=2023):
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))

def random_routing():
    return "".join([str(random.randint(0, 9)) for _ in range(9)])

def random_account_number(prefix):
    return prefix + "".join([str(random.randint(0, 9)) for _ in range(10)])

def generate_accounts(n=None):
    n = n or random.randint(1, 5)
    used_types = random.sample(ACCOUNT_TYPES, min(n, len(ACCOUNT_TYPES)))
    accounts = []
    for atype in used_types:
        prefix = ACCOUNT_PREFIXES[atype]
        opened = random_date(1995, 2022)
        accounts.append({
            "account_number": random_account_number(prefix),
            "routing_number": random_routing(),
            "account_type": atype,
            "holding_name": fake.company() if "Business" in atype or atype in ["Trust", "IRA"] else f"{fake.last_name()} {random.choice(['Family Trust', 'LLC', 'Holdings', 'TTEE', ''])}".strip(),
            "balance": round(random.uniform(500, 250000), 2),
            "currency": random.choices(CURRENCIES, weights=[85, 7, 5, 3])[0],
            "status": random.choices(STATUSES, weights=[75, 10, 10, 5])[0],
            "date_opened": opened.isoformat(),
            "interest_rate": round(random.uniform(0.01, 5.5), 2),
            "overdraft_protection": random.choice([True, False]),
        })
    return accounts

def generate_client(client_index):
    first = fake.first_name()
    last = fake.last_name()
    dob = random_date(1940, 2000)
    customer_since = random_date(2000, 2023)
    return {
        "client_id": f"CLT{str(client_index).zfill(5)}",
        "personal_information": {
            "first_name": first,
            "last_name": last,
            "date_of_birth": dob.isoformat(),
            "ssn_last4": str(random.randint(1000, 9999)),
            "email": f"{first.lower()}.{last.lower()}{random.randint(1,99)}@{random.choice(['gmail.com','outlook.com','yahoo.com','icloud.com'])}",
            "phone_primary": fake.phone_number(),
            "phone_secondary": fake.phone_number() if random.random() > 0.3 else None,
        },
        "address": {
            "street": fake.street_address(),
            "city": fake.city(),
            "state": random.choice(STATES),
            "zip_code": fake.zipcode(),
            "country": "USA",
        },
        "banking_profile": {
            "customer_since": customer_since.isoformat(),
            "branch_id": f"BR{str(random.randint(1,99)).zfill(3)}",
            "relationship_manager": fake.name(),
            "credit_score": random.randint(300, 850),
            "kyc_verified": random.choices([True, False], weights=[85, 15])[0],
            "pep_flag": random.choices([True, False], weights=[3, 97])[0],
            "preferred_contact": random.choice(PREFERRED_CONTACTS),
        },
        "accounts": generate_accounts(),
        "check_printing": {
            "authorized": random.choices([True, False], weights=[60, 40])[0],
            "check_style": random.choice(CHECK_STYLES),
            "last_check_order_date": random_date(1990, 2023).isoformat(),
            "starting_check_number": random.randint(1000, 9999),
        },
        # Versioning fields
        "is_active": True,
        "version": 1,
        "start_date": datetime.utcnow().isoformat(),
        "end_date": None,
        "etl_batch_id": str(uuid.uuid4()),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }

def generate_dataset(n=50):
    clients = [generate_client(i + 1) for i in range(n)]
    return clients

if __name__ == "__main__":
    data = generate_dataset(50)
    with open("/home/claude/banking_etl/data/clients.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {len(data)} client records → data/clients.json")
