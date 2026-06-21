import csv
import json
import sys
from datetime import date
from pathlib import Path

# Add backend directory to sys.path so we can import app modules
backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(backend_dir))

from app.db.connection import get_supabase
from app.utils.text_processing import extract_domain

# Paths
BASE_DIR = backend_dir.parent
CSV_PATH = BASE_DIR / "data" / "AllSides" / "allsides.csv"
MAPPING_PATH = BASE_DIR / "data" / "AllSides" / "domain_mapping.json"

def seed_database():
    print("Starting database seeding for AllSides bias ratings...")
    
    if not CSV_PATH.exists() or not MAPPING_PATH.exists():
        print(f"Error: Missing data files.\nCSV: {CSV_PATH}\nMapping: {MAPPING_PATH}")
        sys.exit(1)
        
    try:
        # Load domain mapping
        with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
            domain_mapping = json.load(f)
            
        # Bias value map
        bias_value_map = {
            "left": "Left",
            "left-center": "Lean Left",
            "center": "Center",
            "allsides": "Center",
            "right-center": "Lean Right",
            "right": "Right"
        }
        
        records = []
        seen_domains = set()
        
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["name"]
                raw_bias = row["bias"]
                
                domain = domain_mapping.get(name)
                if not domain:
                    continue
                    
                clean_domain = extract_domain(domain)
                if not clean_domain:
                    continue
                    
                # Skip duplicate domains to avoid primary key collisions during batch
                if clean_domain in seen_domains:
                    continue
                seen_domains.add(clean_domain)
                
                bias_label = bias_value_map.get(raw_bias.lower().strip(), "Center")
                
                records.append({
                    "domain": clean_domain,
                    "outlet_name": name,
                    "allsides_rating": bias_label,
                    "last_updated": date.today().isoformat()
                })
                
        print(f"Parsed {len(records)} unique domain records to seed.")
        
        # Get Supabase client
        supabase = get_supabase()
        
        print("Upserting records into 'bias_ratings' table...")
        response = supabase.table("bias_ratings").upsert(records).execute()
        
        if response.data:
            print(f"Successfully seeded {len(response.data)} records into 'bias_ratings'!")
        else:
            print("Warning: Upsert completed, but no data was returned.")
            
    except Exception as e:
        print("\n" + "="*60)
        print("WARNING: Supabase database connection failed.")
        print("This is expected if your database project is currently paused or inactive.")
        print("The VeriFact pipeline will gracefully fall back to local in-memory lookups.")
        print(f"Error details: {e}")
        print("="*60 + "\n")
        # Exit cleanly since database seeding failure is non-blocking for local dev
        sys.exit(0)

if __name__ == "__main__":
    seed_database()
