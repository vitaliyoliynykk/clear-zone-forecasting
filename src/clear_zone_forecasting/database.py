"""MongoDB connection helpers."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient:
    """Create and cache a MongoDB client from ``MONGODB_URI``."""
    # Accept the shorter name as well for compatibility with existing .env files.
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError(
            "MONGODB_URI is not set. Add it to the project .env file."
        )
    return MongoClient(uri, serverSelectionTimeoutMS=5000)


def get_database() -> Database:
    """Return the configured MongoDB database."""
    database_name = os.getenv("MONGODB_DATABASE")
    if not database_name:
        raise RuntimeError(
            "MONGODB_DATABASE is not set. Add it to the project .env file."
        )
    return get_mongo_client()[database_name]


def check_mongodb_connection() -> None:
    """Verify that the configured MongoDB server is reachable."""
    get_mongo_client().admin.command("ping")
