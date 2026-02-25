# File location: 5G_Emulator_API/core_network/db.py
# MongoDB Persistence Layer for 5G Core Network Functions
# Provides unified database access for all NFs with fallback to in-memory storage

import os
import logging
from typing import Dict, List, Optional, Any, TypeVar, Generic
from datetime import datetime, timezone
from pydantic import BaseModel
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB configuration from environment
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.environ.get("MONGODB_DATABASE", "open5g_core")
MONGODB_ENABLED = os.environ.get("MONGODB_ENABLED", "false").lower() == "true"

# Try to import motor (async MongoDB driver)
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False
    logger.warning("Motor/PyMongo not installed. Using in-memory storage only.")

T = TypeVar('T')


class InMemoryCollection(Generic[T]):
    """In-memory fallback collection that mimics MongoDB collection interface"""

    def __init__(self, name: str):
        self.name = name
        self._data: Dict[str, Dict[str, Any]] = {}

    async def find_one(self, filter_dict: Dict) -> Optional[Dict]:
        """Find a single document matching the filter"""
        for doc in self._data.values():
            if self._matches_filter(doc, filter_dict):
                return doc
        return None

    async def find(self, filter_dict: Dict = None) -> List[Dict]:
        """Find all documents matching the filter"""
        if filter_dict is None:
            return list(self._data.values())
        return [doc for doc in self._data.values()
                if self._matches_filter(doc, filter_dict)]

    async def insert_one(self, document: Dict) -> str:
        """Insert a single document"""
        doc_id = document.get("_id") or document.get("id") or str(len(self._data))
        document["_id"] = doc_id
        document["created_at"] = datetime.now(timezone.utc).isoformat()
        self._data[doc_id] = document
        return doc_id

    async def update_one(self, filter_dict: Dict, update: Dict, upsert: bool = False) -> bool:
        """Update a single document"""
        for doc_id, doc in self._data.items():
            if self._matches_filter(doc, filter_dict):
                if "$set" in update:
                    doc.update(update["$set"])
                else:
                    doc.update(update)
                doc["updated_at"] = datetime.now(timezone.utc).isoformat()
                return True

        if upsert:
            new_doc = filter_dict.copy()
            if "$set" in update:
                new_doc.update(update["$set"])
            else:
                new_doc.update(update)
            await self.insert_one(new_doc)
            return True
        return False

    async def delete_one(self, filter_dict: Dict) -> bool:
        """Delete a single document"""
        for doc_id, doc in list(self._data.items()):
            if self._matches_filter(doc, filter_dict):
                del self._data[doc_id]
                return True
        return False

    async def delete_many(self, filter_dict: Dict) -> int:
        """Delete all matching documents"""
        to_delete = []
        for doc_id, doc in self._data.items():
            if self._matches_filter(doc, filter_dict):
                to_delete.append(doc_id)
        for doc_id in to_delete:
            del self._data[doc_id]
        return len(to_delete)

    async def count_documents(self, filter_dict: Dict = None) -> int:
        """Count matching documents"""
        if filter_dict is None:
            return len(self._data)
        return len(await self.find(filter_dict))

    def _matches_filter(self, doc: Dict, filter_dict: Dict) -> bool:
        """Check if document matches filter"""
        for key, value in filter_dict.items():
            if key.startswith("$"):
                continue  # Skip operators for simple matching
            if "." in key:
                # Handle nested keys like "plmnId.mcc"
                parts = key.split(".")
                current = doc
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        return False
                if current != value:
                    return False
            elif key not in doc or doc[key] != value:
                return False
        return True


class Database:
    """Unified database interface supporting MongoDB with in-memory fallback"""

    _instance = None
    _client = None
    _db = None
    _collections: Dict[str, Any] = {}
    _is_connected = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def connect(self) -> bool:
        """Establish database connection"""
        if self._is_connected:
            return True

        if MONGODB_ENABLED and MOTOR_AVAILABLE:
            try:
                self._client = AsyncIOMotorClient(
                    MONGODB_URI,
                    serverSelectionTimeoutMS=5000
                )
                # Test connection
                await self._client.admin.command('ping')
                self._db = self._client[MONGODB_DATABASE]
                self._is_connected = True
                logger.info(f"Connected to MongoDB: {MONGODB_DATABASE}")
                return True
            except Exception as e:
                logger.warning(f"MongoDB connection failed: {e}. Using in-memory storage.")
                self._is_connected = False
                return False
        else:
            logger.info("Using in-memory storage (MongoDB disabled or unavailable)")
            self._is_connected = False
            return False

    def get_collection(self, name: str):
        """Get or create a collection"""
        if name not in self._collections:
            if self._is_connected and self._db is not None:
                self._collections[name] = self._db[name]
            else:
                self._collections[name] = InMemoryCollection(name)
        return self._collections[name]

    async def close(self):
        """Close database connection"""
        if self._client:
            self._client.close()
            self._is_connected = False
            logger.info("Database connection closed")


# Singleton instance
db = Database()


# Collection name constants for each NF
class Collections:
    # NRF
    NF_PROFILES = "nf_profiles"
    NF_SUBSCRIPTIONS = "nf_subscriptions"
    ACCESS_TOKENS = "access_tokens"

    # UDM/UDR
    SUBSCRIBERS = "subscribers"
    AUTHENTICATION_DATA = "authentication_data"
    SESSION_MANAGEMENT_DATA = "session_management_data"

    # AMF
    UE_CONTEXTS = "ue_contexts"
    REGISTRATIONS = "registrations"

    # SMF
    PDU_SESSIONS = "pdu_sessions"

    # PCF
    POLICY_ASSOCIATIONS = "policy_associations"
    SM_POLICY_DECISIONS = "sm_policy_decisions"
    AM_POLICY_DATA = "am_policy_data"
    PCC_RULES = "pcc_rules"
    QOS_DATA = "qos_data"

    # AUSF
    AUTHENTICATION_CONTEXTS = "authentication_contexts"

    # NSSF
    NSSAI_AVAILABILITY = "nssai_availability"
    AMF_AVAILABILITY = "amf_availability"
    NETWORK_SLICES = "network_slices"

    # BSF
    PCF_BINDINGS = "pcf_bindings"

    # CHF
    CHARGING_DATA = "charging_data"
    CDRS = "cdrs"
    RATING_GROUPS = "rating_groups"

    # SEPP
    SECURITY_POLICIES = "security_policies"
    ROAMING_PARTNERS = "roaming_partners"
    N32_CONNECTIONS = "n32_connections"

    # NEF
    AF_SUBSCRIPTIONS = "af_subscriptions"
    TRAFFIC_INFLUENCE = "traffic_influence"
    MONITORING_EVENTS = "monitoring_events"


# Helper functions for common operations
async def store_nf_profile(nf_profile: Dict) -> str:
    """Store NF profile in database"""
    collection = db.get_collection(Collections.NF_PROFILES)
    nf_id = nf_profile.get("nfInstanceId")
    await collection.update_one(
        {"nfInstanceId": nf_id},
        {"$set": nf_profile},
        upsert=True
    )
    return nf_id


async def get_nf_profile(nf_instance_id: str) -> Optional[Dict]:
    """Retrieve NF profile from database"""
    collection = db.get_collection(Collections.NF_PROFILES)
    return await collection.find_one({"nfInstanceId": nf_instance_id})


async def delete_nf_profile(nf_instance_id: str) -> bool:
    """Delete NF profile from database"""
    collection = db.get_collection(Collections.NF_PROFILES)
    return await collection.delete_one({"nfInstanceId": nf_instance_id})


async def find_nf_profiles(filter_dict: Dict = None) -> List[Dict]:
    """Find NF profiles matching filter"""
    collection = db.get_collection(Collections.NF_PROFILES)
    return await collection.find(filter_dict or {})


async def store_subscriber(supi: str, data: Dict) -> str:
    """Store subscriber data"""
    collection = db.get_collection(Collections.SUBSCRIBERS)
    data["supi"] = supi
    await collection.update_one(
        {"supi": supi},
        {"$set": data},
        upsert=True
    )
    return supi


async def get_subscriber(supi: str) -> Optional[Dict]:
    """Retrieve subscriber data"""
    collection = db.get_collection(Collections.SUBSCRIBERS)
    return await collection.find_one({"supi": supi})


async def store_pdu_session(session_id: str, data: Dict) -> str:
    """Store PDU session data"""
    collection = db.get_collection(Collections.PDU_SESSIONS)
    data["sessionId"] = session_id
    await collection.update_one(
        {"sessionId": session_id},
        {"$set": data},
        upsert=True
    )
    return session_id


async def get_pdu_session(session_id: str) -> Optional[Dict]:
    """Retrieve PDU session data"""
    collection = db.get_collection(Collections.PDU_SESSIONS)
    return await collection.find_one({"sessionId": session_id})


async def store_charging_record(record_id: str, data: Dict) -> str:
    """Store charging data record"""
    collection = db.get_collection(Collections.CDRS)
    data["recordId"] = record_id
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    await collection.insert_one(data)
    return record_id


# Initialize database on module load
async def init_database():
    """Initialize database connection"""
    await db.connect()
    logger.info("Database initialization complete")
