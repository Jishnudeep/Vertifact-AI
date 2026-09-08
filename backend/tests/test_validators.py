import pytest
from app.utils.validators import sanitize_text, validate_input_length, is_verifiable_claim, MAX_INPUT_LENGTH

def test_sanitize_text_strips_scripts_and_html():
    raw = "<script>alert('pwned');</script><b>NASA</b> announces <i>Mars discovery</i>"
    clean = sanitize_text(raw)
    assert "<script>" not in clean
    assert "alert" not in clean
    assert "<b>" not in clean
    assert clean == "NASA announces Mars discovery"

def test_sanitize_text_strips_tracking_params():
    url_input = "https://example.com/news/article?utm_source=twitter&utm_medium=social&fbclid=12345&real_id=999"
    clean = sanitize_text(url_input)
    assert "utm_source" not in clean
    assert "fbclid" not in clean
    assert "real_id=999" in clean

def test_validate_input_length():
    # Empty
    valid, err = validate_input_length("")
    assert not valid
    assert "empty" in err

    valid, err = validate_input_length("   ")
    assert not valid

    # Valid
    valid, err = validate_input_length("A valid factual claim.")
    assert valid
    assert err is None

    # Exceeds max length
    long_text = "a" * (MAX_INPUT_LENGTH + 1)
    valid, err = validate_input_length(long_text)
    assert not valid
    assert "exceeds maximum length" in err

def test_is_verifiable_claim_rejects_opinions():
    opinions = [
        "I think chocolate is the best flavor ever.",
        "In my opinion, this policy is terrible.",
        "Vanilla ice cream is the greatest dessert in history.",
        "That movie was completely overrated."
    ]
    for op in opinions:
        verifiable, err = is_verifiable_claim(op)
        assert not verifiable
        assert "opinion" in err

def test_is_verifiable_claim_rejects_questions():
    questions = [
        "Why do people believe the Earth is flat?",
        "How many planets are in the Solar System?",
        "Could you tell me what time it is?",
        "Is it true that sharks never sleep?"
    ]
    for q in questions:
        verifiable, err = is_verifiable_claim(q)
        assert not verifiable
        assert "question" in err

def test_is_verifiable_claim_accepts_factual_assertions():
    factual_claims = [
        "The Earth revolves around the Sun once every 365 days.",
        "NASA landed astronauts on the Moon in 1969.",
        "Drinking bleach cures viral infections.",
        "The United States Federal Reserve raised interest rates in 2023."
    ]
    for fc in factual_claims:
        verifiable, err = is_verifiable_claim(fc)
        assert verifiable
        assert err is None
