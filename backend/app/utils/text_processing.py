import re
from urllib.parse import urlparse

def extract_domain(url_or_domain: str) -> str:
    """
    Extracts and normalizes the primary domain name from a URL or raw domain string.
    
    Examples:
        'https://www.nytimes.com/path' -> 'nytimes.com'
        'opinion.nytimes.com' -> 'nytimes.com'
        'bbc.co.uk' -> 'bbc.co.uk'
        'news.bbc.co.uk' -> 'bbc.co.uk'
    """
    if not url_or_domain:
        return ""
        
    url_or_domain = url_or_domain.strip().lower()
    
    # Prepend scheme if not present to allow proper urlparse
    if not url_or_domain.startswith(("http://", "https://")):
        # If it doesn't look like a URL but has path/slashes, handle it
        if "/" in url_or_domain:
            url_or_domain = "https://" + url_or_domain
            netloc = urlparse(url_or_domain).netloc
        else:
            # It's a raw domain
            netloc = url_or_domain
    else:
        netloc = urlparse(url_or_domain).netloc
        
    if not netloc:
        # Fallback to parsing directly if urlparse fails or netloc is empty
        netloc = url_or_domain.split('/')[0]
        
    # Strip port if present
    if ":" in netloc:
        netloc = netloc.split(":")[0]
        
    # Strip leading www.
    if netloc.startswith("www."):
        netloc = netloc[4:]
        
    # List of common subdomains to strip for primary domain matching
    common_subdomains = ["opinion", "edition", "online", "m", "news", "blog", "us", "uk", "en", "amp", "www"]
    
    # We want to keep second-level domains like .co.uk, .com.au, .gov.uk
    parts = netloc.split(".")
    if len(parts) > 2:
        # e.g., bbc.co.uk (parts = ['bbc', 'co', 'uk'])
        # if the second to last part is 'co', 'com', 'org', 'net', 'gov', 'edu', 'ac'
        # and the last part is a 2-letter country code:
        if parts[-2] in ["co", "com", "org", "net", "gov", "edu", "ac"] and len(parts[-1]) == 2:
            primary_domain = ".".join(parts[-3:])
            # If there's an extra subdomain, e.g. ['news', 'bbc', 'co', 'uk']
            if len(parts) > 3 and parts[0] in common_subdomains:
                primary_domain = ".".join(parts[1:])
        else:
            # Standard TLD, e.g. nytimes.com. Keep the last 2 parts.
            primary_domain = ".".join(parts[-2:])
            # Strip common subdomain if present
            if parts[0] in common_subdomains:
                primary_domain = ".".join(parts[1:])
    else:
        primary_domain = netloc
        
    return primary_domain
