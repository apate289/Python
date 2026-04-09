#python -m pip install "pymongo[srv]==3.12"
import logging, os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import OperationFailure


# Ensure logs are in the same folder as the script
current_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(current_dir, 'app.log')

logging.basicConfig(filename=log_path, level=logging.INFO)
# Log some messages
#logging.info('This is an info message.')
#logging.error('This is an error message.')

uri = "mongodb+srv://ankit5907:Ankit%405907@cluster0.h0f1gaj.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
#client = MongoClient(
#    host="cluster0.h0f1gaj.mongodb.net",
#    username="ankit5907",
#    password="Ankit@5907"
#)
#client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    logging.info("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    logging.error(e)

#client.getMongo().getDBNames().indexOf("testDB")
dbnames = client.list_database_names()
logging.info("Databases: %s", dbnames)

if 'testDB' in dbnames:
    logging.info("------------Database exists------------")
    db = client['testDB']
else:
    logging.error("Database does not exist")
    #db = client['testDB']
    #logging.info("Database created but need collection to be created")

# Check if collection name is in the list of existing collections
#col_list = db.list_collection_names()

try:
    db.validate_collection("testCollection")
    logging.info("The collection exists.")
    collection = db['testCollection']
except OperationFailure:
    logging.error("The collection does not exist.")
    #collection = db['testCollection']
    #logging.info("Collection created")
"""
if 'testCollection' in col_list:
    logging.info("------------Collection exists------------")
else:    
    logging.error("Collection does not exist")
    collection = db['testCollection']
    logging.info("Collection created")
"""

test_document = {
  "client_id": "CLT00001",
  "personal_information": {
    "first_name": "Linda",
    "last_name": "Johnson",
    "date_of_birth": "1971-12-21",
    "ssn_last4": "9935",
    "email": "linda.johnson12@icloud.com",
    "phone_primary": "(425) 659-5557",
    "phone_secondary": "(206) 977-3615"
  },
  "address": {
    "street": "2386 Pine Road",
    "city": "Chicago",
    "state": "IL",
    "zip_code": "60601",
    "country": "USA"
  },
  "banking_profile": {
    "customer_since": "2018-12-16",
    "branch_id": "BR022",
    "relationship_manager": "Karen Martinez",
    "credit_score": 690,
    "kyc_verified": True,
    "pep_flag": False,
    "preferred_contact": "Phone"
  },
  "accounts": [
    {
      "account_number": "CHK1402418010",
      "routing_number": "041227216",
      "account_type": "Checking",
      "holding_name": "Linda Johnson",
      "balance": 126586.14,
      "currency": "USD",
      "status": "Inactive",
      "date_opened": "2008-12-01",
      "interest_rate": 0.0,
      "overdraft_protection": False
    }
  ],
  "check_printing": {
    "authorized": True,
    "check_style": "Personal",
    "last_check_order_date": "2004-02-28",
    "starting_check_number": 2557
  }
}

insert_test_doc = collection.insert_one(test_document)
logging.info("Inserted document with _id: %s", insert_test_doc.inserted_id)

client.close()
logging.info("Closed MongoDB connection.")