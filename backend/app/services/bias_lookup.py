import csv
import json
import logging
from pathlib import Path
from typing import Optional
from app.utils.text_processing import extract_domain
from app.db.connection import get_supabase

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CSV_PATH = BASE_DIR / "data" / "AllSides" / "allsides.csv"
MAPPING_PATH = BASE_DIR / "data" / "AllSides" / "domain_mapping.json"

# In-memory dictionary: domain -> bias_label
BIAS_RATINGS: dict[str, str] = {}
# In-memory dictionary: domain -> outlet_name
OUTLET_NAMES: dict[str, str] = {}

def load_bias_ratings():
    """Load AllSides ratings from Supabase database, falling back to local files if unavailable."""
    global BIAS_RATINGS, OUTLET_NAMES
    BIAS_RATINGS.clear()
    OUTLET_NAMES.clear()
    
    # 1. Try loading from Supabase database first
    try:
        supabase = get_supabase()
        response = supabase.table("bias_ratings").select("domain, outlet_name, allsides_rating").execute()
        if response.data:
            for row in response.data:
                BIAS_RATINGS[row["domain"]] = row["allsides_rating"]
                OUTLET_NAMES[row["domain"]] = row["outlet_name"]
            
            # Re-apply aliases for domain robustness even when loading from DB
            aliases = {
                "bbc.co.uk": "bbc.com",
                "theguardian.co.uk": "theguardian.com",
                "guardian.co.uk": "theguardian.com",
                "associated-press.com": "apnews.com"
            }
            for alias, target in aliases.items():
                if target in BIAS_RATINGS:
                    BIAS_RATINGS[alias] = BIAS_RATINGS[target]
                    OUTLET_NAMES[alias] = OUTLET_NAMES[target]
            
            logger.info(f"Successfully loaded {len(BIAS_RATINGS)} bias ratings from Supabase database.")
            return
    except Exception as e:
        logger.warning(f"Failed to load from Supabase database: {e}")
        logger.info("Falling back to local CSV and JSON mapping...")
        
    # 2. Local fallback if database query failed
    if not CSV_PATH.exists() or not MAPPING_PATH.exists():
        # Do not crash if files are missing, log a warning
        logger.warning(f"Bias rating files missing. CSV: {CSV_PATH}, Mapping: {MAPPING_PATH}")
        return
        
    try:
        # Load domain mapping JSON
        with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
            domain_mapping = json.load(f)
            
        # Map raw CSV bias values to the standard DB rating constraints
        bias_value_map = {
            "left": "Left",
            "left-center": "Lean Left",
            "center": "Center",
            "allsides": "Center",
            "right-center": "Lean Right",
            "right": "Right"
        }
        
        # Track raw outlet names to handle duplicate domains (e.g. News vs Opinion)
        bias_outlet_names = {}
        
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["name"]
                raw_bias = row["bias"]
                bias_label = bias_value_map.get(raw_bias.lower().strip(), "Center")
                
                # Get the domain for this outlet name from mapping
                domain = domain_mapping.get(name)
                if domain:
                    clean_domain = extract_domain(domain)
                    existing_name = bias_outlet_names.get(clean_domain)
                    
                    if existing_name:
                        # Prefer News over Opinion/Editorial/Blog/Column/Humor
                        existing_is_op = any(kw in existing_name.lower() for kw in ["opinion", "editorial", "blog", "column", "humor"])
                        new_is_op = any(kw in name.lower() for kw in ["opinion", "editorial", "blog", "column", "humor"])
                        
                        if existing_is_op and not new_is_op:
                            # Overwrite: existing was opinion, new is news
                            BIAS_RATINGS[clean_domain] = bias_label
                            bias_outlet_names[clean_domain] = name
                        elif not existing_is_op and new_is_op:
                            # Keep existing: existing is news, new is opinion
                            pass
                        else:
                            # Both are same category, keep the first one
                            pass
                    else:
                        BIAS_RATINGS[clean_domain] = bias_label
                        bias_outlet_names[clean_domain] = name
                        
        # Add common domain aliases
        aliases = {
            "bbc.co.uk": "bbc.com",
            "theguardian.co.uk": "theguardian.com",
            "guardian.co.uk": "theguardian.com",
            "associated-press.com": "apnews.com"
        }
        for alias, target in aliases.items():
            if target in BIAS_RATINGS:
                BIAS_RATINGS[alias] = BIAS_RATINGS[target]
                bias_outlet_names[alias] = bias_outlet_names[target]
                
        # Populate global OUTLET_NAMES
        for domain, name in bias_outlet_names.items():
            OUTLET_NAMES[domain] = name
                
        logger.info(f"Successfully loaded {len(BIAS_RATINGS)} bias ratings from local CSV fallback.")
    except Exception as local_err:
        logger.error(f"Error loading bias ratings from local files: {local_err}")

def get_bias_label(url_or_domain: str) -> str:
    """
    Get the bias rating label for a URL or domain.
    Returns 'Unrated' if the domain is not in the AllSides database.
    """
    if not url_or_domain:
        return "Unrated"
        
    # Lazy load if BIAS_RATINGS has not been populated
    if not BIAS_RATINGS:
        load_bias_ratings()
        
    domain = extract_domain(url_or_domain)
    return BIAS_RATINGS.get(domain, "Unrated")

def get_outlet_name(url_or_domain: str) -> Optional[str]:
    """
    Get the outlet name for a URL or domain.
    Returns None if the domain is not in the AllSides database.
    """
    if not url_or_domain:
        return None
        
    # Lazy load if OUTLET_NAMES has not been populated
    if not OUTLET_NAMES:
        load_bias_ratings()
        
    domain = extract_domain(url_or_domain)
    return OUTLET_NAMES.get(domain)

# Auto-load on import removed. Ratings are loaded at app startup or lazy-loaded on demand.
