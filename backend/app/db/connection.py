import sys
import logging
from pathlib import Path
from typing import Optional

# Add backend/ directory to sys.path so we can import app.config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from supabase import create_client, Client
from app.config import get_settings

logger = logging.getLogger(__name__)

_supabase_client: Optional[Client] = None

def get_supabase() -> Client:
    """
    Lazy initialization of Supabase client to avoid import-time network or auth errors.
    """
    global _supabase_client
    if _supabase_client is None:
        settings = get_settings()
        url = settings.supabase_url
        key = settings.supabase_secret_key
        if not url or not key:
            raise ValueError(
                "SUPABASE_URL or SUPABASE_SECRET_KEY is not configured in settings."
            )
        _supabase_client = create_client(url, key)
    return _supabase_client

# For backward compatibility if someone accesses module attribute directly
def __getattr__(name: str):
    if name == "supabase":
        return get_supabase()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

if __name__ == "__main__":
    print("Testing Supabase connection...")
    try:
        client = get_supabase()
        # 1. Insert a test row into 'claims'
        test_claim = {
            "input_type": "text",
            "raw_input": "Test connection raw input",
            "extracted_claim": "Test connection extracted claim"
        }
        print(f"Inserting row: {test_claim}")
        insert_response = client.table("claims").insert(test_claim).execute()
        
        if not insert_response.data:
            print("Failed to insert data: No data returned.")
            sys.exit(1)
            
        inserted_id = insert_response.data[0]["id"]
        print(f"Successfully inserted claim with ID: {inserted_id}")
        
        # 2. Query the inserted row
        print(f"Querying claim with ID: {inserted_id}")
        query_response = client.table("claims").select("*").eq("id", inserted_id).execute()
        
        if not query_response.data:
            print("Failed to query data: No data returned.")
            sys.exit(1)
            
        retrieved_claim = query_response.data[0]
        print(f"Successfully retrieved claim: {retrieved_claim}")
        
        # 3. Clean up the test row
        print(f"Cleaning up: deleting test claim with ID: {inserted_id}")
        delete_response = client.table("claims").delete().eq("id", inserted_id).execute()
        print("Cleanup successful.")
        print("Supabase connection and verification test PASSED!")
        
    except Exception as e:
        print(f"An error occurred during verification: {e}")
        sys.exit(1)
