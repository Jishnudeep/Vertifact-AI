import sys
from pathlib import Path

# Add backend/ directory to sys.path so we can import app.config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from supabase import create_client, Client
from app.config import SUPABASE_URL, SUPABASE_KEY

# Initialize the Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase() -> Client:
    return supabase

if __name__ == "__main__":
    print("Testing Supabase connection...")
    try:
        # 1. Insert a test row into 'claims'
        test_claim = {
            "input_type": "text",
            "raw_input": "Test connection raw input",
            "extracted_claim": "Test connection extracted claim"
        }
        print(f"Inserting row: {test_claim}")
        insert_response = supabase.table("claims").insert(test_claim).execute()
        
        if not insert_response.data:
            print("Failed to insert data: No data returned.")
            sys.exit(1)
            
        inserted_id = insert_response.data[0]["id"]
        print(f"Successfully inserted claim with ID: {inserted_id}")
        
        # 2. Query the inserted row
        print(f"Querying claim with ID: {inserted_id}")
        query_response = supabase.table("claims").select("*").eq("id", inserted_id).execute()
        
        if not query_response.data:
            print("Failed to query data: No data returned.")
            sys.exit(1)
            
        retrieved_claim = query_response.data[0]
        print(f"Successfully retrieved claim: {retrieved_claim}")
        
        # 3. Clean up the test row
        print(f"Cleaning up: deleting test claim with ID: {inserted_id}")
        delete_response = supabase.table("claims").delete().eq("id", inserted_id).execute()
        print("Cleanup successful.")
        print("Supabase connection and verification test PASSED!")
        
    except Exception as e:
        print(f"An error occurred during verification: {e}")
        sys.exit(1)
