import pytest
from app.utils.text_processing import extract_domain
from app.services.bias_lookup import get_bias_label, load_bias_ratings

def test_extract_domain():
    # Test standard URLs
    assert extract_domain("https://www.nytimes.com/2026/06/20/world") == "nytimes.com"
    assert extract_domain("http://nytimes.com/story") == "nytimes.com"
    
    # Test subdomains
    assert extract_domain("https://opinion.nytimes.com/piece") == "nytimes.com"
    assert extract_domain("edition.cnn.com/index.html") == "cnn.com"
    assert extract_domain("news.yahoo.com") == "yahoo.com"
    
    # Test double extensions (country code SLDs)
    assert extract_domain("https://www.bbc.co.uk/news") == "bbc.co.uk"
    assert extract_domain("news.bbc.co.uk/story") == "bbc.co.uk"
    assert extract_domain("http://website.com.au") == "website.com.au"
    
    # Test raw domains and formatting
    assert extract_domain("  NYTimes.com  ") == "nytimes.com"
    assert extract_domain("https://www.google.com:8080/search?q=test") == "google.com"
    assert extract_domain("") == ""
    assert extract_domain(None) == ""

def test_get_bias_label():
    # Make sure database/in-memory ratings are loaded
    load_bias_ratings()
    
    # Test known outlets
    assert get_bias_label("nytimes.com") == "Lean Left"
    assert get_bias_label("https://www.nytimes.com/article") == "Lean Left"
    
    assert get_bias_label("foxnews.com") == "Right"
    assert get_bias_label("http://opinion.foxnews.com/piece") == "Right"
    
    assert get_bias_label("apnews.com") == "Center"
    
    assert get_bias_label("bbc.com") == "Center"
    assert get_bias_label("https://news.bbc.co.uk/world") == "Center"  # Resolves to bbc.co.uk which maps to bbc.com via override or bbc.co.uk
    
    # Test unknown domain -> 'Unrated'
    assert get_bias_label("unknown-domain-test-123.com") == "Unrated"
    assert get_bias_label("") == "Unrated"
    assert get_bias_label(None) == "Unrated"
