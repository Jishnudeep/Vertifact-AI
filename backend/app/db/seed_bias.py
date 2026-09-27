import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import List

# Add backend directory to sys.path so we can import app modules
backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(backend_dir))

from app.db.connection import get_supabase
from app.services.bias_lookup import UNRATED_BRIDGING_DOMAINS
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
            "right-center": "Lean Right",
            "right": "Right"
        }
        
        records = []
        seen_domains = set()
        
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["name"]
                raw_bias = row.get("bias", "")

                # AllSides is the only source of truth: skip rows whose bias value
                # is blank or outside the known set rather than defaulting to "Center".
                clean_raw_bias = raw_bias.lower().strip() if raw_bias else ""
                if clean_raw_bias not in bias_value_map:
                    continue

                domain = domain_mapping.get(name)
                if not domain:
                    continue
                    
                clean_domain = extract_domain(domain)
                if not clean_domain:
                    continue

                # AllSides bridging/depolarization orgs carry no left-right rating: never seed them
                if clean_domain in UNRATED_BRIDGING_DOMAINS:
                    continue


                # Skip duplicate domains to avoid primary key collisions during batch
                if clean_domain in seen_domains:
                    continue
                seen_domains.add(clean_domain)
                
                bias_label = bias_value_map[clean_raw_bias]
                
                records.append({
                    "domain": clean_domain,
                    "outlet_name": name,
                    "allsides_rating": bias_label,
                    "last_updated": date.today().isoformat()
                })
                
        print(f"Parsed {len(records)} unique domain records to seed.")
        
        if not records:
            print("Error: no valid AllSides records parsed; refusing to touch 'bias_ratings'.")
            sys.exit(1)

        # Get Supabase client
        supabase = get_supabase()

        # R7.D: upsert FIRST, prune AFTER. The previous order ran a full-table
        # delete().neq("domain", "") before the upsert, outside any transaction — a failed
        # upsert left bias_ratings empty and every domain silently became "Unrated" in prod.
        print("Upserting records into 'bias_ratings' table...")
        response = supabase.table("bias_ratings").upsert(records).execute()

        if response.data:
            print(f"Successfully seeded {len(response.data)} records into 'bias_ratings'!")
        else:
            print("Warning: Upsert completed, but no data was returned.")

        print("Pruning stale/unrated bridging records from 'bias_ratings' table...")
        try:
            supabase.table("bias_ratings").delete().in_("domain", sorted(UNRATED_BRIDGING_DOMAINS)).execute()

            # Remove rows that are not in this fresh seed (stale domains from earlier runs).
            # Read-then-delete in chunks rather than a single giant NOT IN filter.
            #
            # PostgREST caps an unbounded .select() at ~1000 rows, so the sweep MUST paginate:
            # without .range() the tail of the table is invisible and never pruned, and the
            # "did I read enough to prune safely?" floor below would compare against a
            # truncated view.
            page_size = 1000
            existing_domains: List[str] = []
            offset = 0
            while True:
                page = (
                    supabase.table("bias_ratings")
                    .select("domain")
                    .range(offset, offset + page_size - 1)
                    .execute()
                )
                rows = page.data or []
                existing_domains.extend(row["domain"] for row in rows if row.get("domain"))
                if len(rows) < page_size:
                    break
                offset += page_size

            # Safety floor: the rows we just upserted must all be visible in the sweep. If the
            # read came back materially short (paging failure, partial outage, replica lag),
            # every "missing" row would look stale and the prune would wipe live ratings.
            if len(existing_domains) < 0.8 * len(records):
                print(
                    f"Aborting prune: sweep saw only {len(existing_domains)} domains but "
                    f"{len(records)} were just seeded (< 80%). Refusing to delete on a "
                    f"partial read; fresh rows are already committed."
                )
            else:
                stale = [d for d in existing_domains if d not in seen_domains]
                for i in range(0, len(stale), 100):
                    supabase.table("bias_ratings").delete().in_("domain", stale[i:i + 100]).execute()
                if stale:
                    print(f"Pruned {len(stale)} stale domain records.")
        except Exception as purge_err:
            print(f"Notice during prune (fresh rows are already committed): {purge_err}")


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
