import csv
import json
import os
import re
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CSV_PATH = BASE_DIR / "data" / "AllSides" / "allsides.csv"
OUTPUT_JSON_PATH = BASE_DIR / "data" / "AllSides" / "domain_mapping.json"

# Curated manual overrides for main news sources and local/specialized publications
MANUAL_MAPPING = {
    "ABC News (Online)": "abcnews.go.com",
    "ABC News": "abcnews.go.com",
    "Above The Law": "abovethelaw.com",
    "AJ+": "ajplus.net",
    "Al Jazeera": "aljazeera.com",
    "AlterNet": "alternet.org",
    "American Greatness": "amgreatness.com",
    "American Spectator": "spectator.org",
    "American Thinker": "americanthinker.com",
    "Associated Press": "apnews.com",
    "AP": "apnews.com",
    "Atlanta Black Star": "atlantablackstar.com",
    "Atlanta Journal-Constitution": "ajc.com",
    "Austin American-Statesman": "statesman.com",
    "Axios": "axios.com",
    "Babylon Bee (Humor)": "babylonbee.com",
    "BBC News": "bbc.com",
    "Bloomberg": "bloomberg.com",
    "Boston Globe": "bostonglobe.com",
    "Breitbart News": "breitbart.com",
    "Business Insider": "businessinsider.com",
    "Insider": "businessinsider.com",
    "Bustle": "bustle.com",
    "BuzzFeed News": "buzzfeednews.com",
    "C-SPAN": "c-span.org",
    "CalMatters": "calmatters.org",
    "CBN": "cbn.com",
    "CBS News (Online)": "cbsnews.com",
    "CBS News": "cbsnews.com",
    "Center for Public Integrity": "publicintegrity.org",
    "Chicago Sun-Times": "chicago.suntimes.com",
    "Chicago Tribune": "chicagotribune.com",
    "Christian Science Monitor": "csmonitor.com",
    "CNBC": "cnbc.com",
    "CNET": "cnet.com",
    "CNN (Online News)": "cnn.com",
    "CNN (Opinion)": "cnn.com",
    "CNN Business": "cnn.com",
    "CNSNews.com": "cnsnews.com",
    "Daily Beast": "thedailybeast.com",
    "Daily Caller": "dailycaller.com",
    "Daily Kos": "dailykos.com",
    "Daily Mail": "dailymail.co.uk",
    "Daily Signal": "dailysignal.com",
    "Daily Wire": "dailywire.com",
    "Democracy Now": "democracynow.org",
    "Deseret News": "deseret.com",
    "Drudge Report": "drudgereport.com",
    "Financial Times": "ft.com",
    "Fiscal Times": "thefiscaltimes.com",
    "FiveThirtyEight": "fivethirtyeight.com",
    "Forbes": "forbes.com",
    "Foreign Affairs": "foreignaffairs.com",
    "Foreign Policy": "foreignpolicy.com",
    "Fox Business": "foxbusiness.com",
    "Fox News (Online News)": "foxnews.com",
    "Fox News (Opinion)": "foxnews.com",
    "Fox News": "foxnews.com",
    "Gizmodo": "gizmodo.com",
    "Google News": "news.google.com",
    "Grist": "grist.org",
    "HotAir": "hotair.com",
    "HuffPost": "huffpost.com",
    "Independent Journal Review": "ijr.com",
    "InfoWars": "infowars.com",
    "Intellectual Conservative": "intellectualconservative.com",
    "Intercept": "theintercept.com",
    "The Intercept": "theintercept.com",
    "Jacobin": "jacobin.com",
    "Jezebel": "jezebel.com",
    "Los Angeles Times": "latimes.com",
    "MarketWatch": "marketwatch.com",
    "Mashable": "mashable.com",
    "Media Matters": "mediamatters.org",
    "MSNBC": "msnbc.com",
    "Mother Jones": "motherjones.com",
    "National Review": "nationalreview.com",
    "NBC News (Online)": "nbcnews.com",
    "New Republic": "newrepublic.com",
    "New York Daily News": "nydailynews.com",
    "New York Magazine": "nymag.com",
    "New York Post (News)": "nypost.com",
    "New York Post (Opinion)": "nypost.com",
    "New York Post": "nypost.com",
    "New York Times (News)": "nytimes.com",
    "New York Times (Opinion)": "nytimes.com",
    "New York Times": "nytimes.com",
    "Newsmax (News)": "newsmax.com",
    "Newsmax - Opinion": "newsmax.com",
    "Newsweek": "newsweek.com",
    "NPR (Online News)": "npr.org",
    "NPR (Opinion)": "npr.org",
    "NPR": "npr.org",
    "One America News Network (OAN)": "oann.com",
    "OAN": "oann.com",
    "Politico": "politico.com",
    "ProPublica": "propublica.org",
    "Quartz": "qz.com",
    "Reason": "reason.com",
    "RealClearPolitics": "realclearpolitics.com",
    "Reuters": "reuters.com",
    "Salon": "salon.com",
    "San Francisco Chronicle": "sfchronicle.com",
    "Scientific American": "scientificamerican.com",
    "Slate": "slate.com",
    "The American Conservative": "theamericanconservative.com",
    "The American Spectator": "spectator.org",
    "The Atlantic": "theatlantic.com",
    "The Daily Caller": "dailycaller.com",
    "The Daily Signal": "dailysignal.com",
    "The Daily Wire": "dailywire.com",
    "The Economist": "economist.com",
    "The Epoch Times": "theepochtimes.com",
    "The Federalist": "thefederalist.com",
    "The Guardian": "theguardian.com",
    "The Hill": "thehill.com",
    "The Huffington Post": "huffpost.com",
    "The Intercept": "theintercept.com",
    "The Nation": "thenation.com",
    "The New Yorker": "newyorker.com",
    "The Wall Street Journal- News": "wsj.com",
    "The Wall Street Journal- Opinion": "wsj.com",
    "The Wall Street Journal": "wsj.com",
    "Wall Street Journal": "wsj.com",
    "The Washington Post": "washingtonpost.com",
    "The Washington Times": "washingtontimes.com",
    "Time Magazine": "time.com",
    "Time": "time.com",
    "USA TODAY": "usatoday.com",
    "Vox": "vox.com",
    "Washington Examiner": "washingtonexaminer.com",
    "Washington Post": "washingtonpost.com",
    "Washington Times": "washingtontimes.com",
    "Yahoo News": "news.yahoo.com",
    "The Telegraph - UK": "telegraph.co.uk",
    "The Tennesseean": "tennessean.com",
    "U.S. News & World Report": "usnews.com",
    "The Sacramento Bee": "sacbee.com",
    "The Seattle Times": "seattletimes.com",
    "The Oregonian": "oregonlive.com",
    "The Philadelphia Inquirer": "inquirer.com",
    "VT Digger": "vtdigger.org",
    "WGBH": "wgbh.org",
    "WFAE": "wfae.org",
    "Washington Free Beacon": "freebeacon.com",
    "The Post Millennial": "thepostmillennial.com",
    "The Onion (Humor)": "theonion.com",
    "The Verge": "theverge.com",
    "The Root": "theroot.com",
    "ThinkProgress": "thinkprogress.org",
    "Truthout": "truthout.org",
    "Voice of America": "voanews.com",
    "Deutsche Welle": "dw.com",
    "Detroit Free Press": "freep.com",
    "Education Week": "edweek.org",
    "FAIR": "fair.org",
    "Falls Church News - Press": "fcnp.com",
    "Honolulu Civil Beat": "civilbeat.org",
    "Indiana Daily Student": "idsnews.com",
    "Investor's Business Daily": "investors.com",
    "JSTOR Daily": "daily.jstor.org",
    "Judicial Watch": "judicialwatch.org",
    "Just Security": "justsecurity.org",
    "KQED": "kqed.org",
    "Law & Crime": "lawandcrime.com",
    "Live Action News": "liveactionnews.org",
    "Los Angeles Daily News": "dailynews.com",
    "Louisville Courier-Journal": "courier-journal.com",
    "MIT News": "news.mit.edu",
    "New Hampshire Union Leader": "unionleader.com",
    "NewsBusters": "newsbusters.org",
    "Nieman Lab": "niemanlab.org",
    "Orange County Register": "ocregister.com",
    "Outkick the Coverage": "outkick.com",
    "Pacific Standard": "psmag.com",
    "PinkNews": "pinknews.co.uk",
    "Pittsburgh Post-Gazette": "post-gazette.com",
    "Portland Press Herald": "pressherald.com",
    "Poynter": "poynter.org",
    "Richmond Times Dispatch": "richmond.com",
    "San Antonio Express-News": "expressnews.com",
    "San Bernardino Sun": "sbsun.com",
    "San Gabriel Valley Tribune": "sgvtribune.com",
    "San Jose Mercury News": "mercurynews.com",
    "Smithsonian Magazine": "smithsonianmag.com",
    "South China Morning Post": "scmp.com",
    "StoryCorps": "storycorps.org",
    "Students For Life": "studentsforlife.org",
    "Tallahassee Democrat": "tallahassee.com",
    "Tampa Bay Times": "tampabay.com",
}

def clean_name_to_domain(name: str) -> str:
    # Check manual mapping first
    if name in MANUAL_MAPPING:
        return MANUAL_MAPPING[name]
    
    # Heuristic cleaning
    cleaned = name
    # Remove text in parentheses, e.g. " (Online)", " (Opinion)"
    cleaned = re.sub(r'\s*\([^)]*\)', '', cleaned)
    # Strip common editorial tags or prefixes
    cleaned = cleaned.replace(" - News", "").replace(" - Opinion/Editorial", "").replace(" Editorial", "")
    
    # Check if there is already a domain extension in the name (e.g. Bridgemi.com)
    m = re.search(r'([a-zA-Z0-9\-]+)\.(com|org|net|edu|gov|co\.uk)$', cleaned.lower())
    if m:
        return m.group(0)
    
    # Lowercase
    cleaned = cleaned.lower()
    # Strip spaces and punctuation
    cleaned = re.sub(r'[^a-z0-9]', '', cleaned)
    
    # Append domain suffix
    # Default is .com unless it looks like an org
    if any(org in name.lower() for org in ["association", "center", "project", "institute", "democracy", "public", "council"]):
        return f"{cleaned}.org"
    return f"{cleaned}.com"

def run_mapping():
    if not CSV_PATH.exists():
        print(f"Error: CSV file not found at {CSV_PATH}")
        return
        
    mapped = {}
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            domain = clean_name_to_domain(name)
            mapped[name] = domain
            
    # Create parent dirs if not exist
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(mapped, f, indent=2)
        
    print(f"Successfully mapped {len(mapped)} outlets.")
    print(f"Output saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    run_mapping()
