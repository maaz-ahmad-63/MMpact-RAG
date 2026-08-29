import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings

settings = get_settings()
print(f"MONGODB_URL loaded: {repr(settings.mongodb_url)}")
print(f"DATABASE: {repr(settings.mongodb_database)}")
print(f"COLLECTION: {repr(settings.mongodb_collection)}")
