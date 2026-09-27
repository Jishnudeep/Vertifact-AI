import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "verifact-ai"

def test_verify_endpoint_success():
    mock_orchestrator_result = {
        "id": "result-12345",
        "claim": "The Earth orbits the Sun",
        "trust_score": {
            "c_total": 0.95,
            "s_fact": 1.0,
            "b_ling": 0.1,
            "v_consensus": 0.8,
            "weights": {"w1": 0.5, "w2": 0.2, "w3": 0.3}
        },
        "confidence": {
            "level": "high",
            "warnings": []
        },
        "linguistic_profile": {
            "clickbait_syntax": 0.0,
            "emotional_framing": 0.0,
            "sensationalism": 0.0,
            "informational_density": 0.9
        },
        "sources": [
            {
                "url": "https://snopes.com/earth-sun",
                "title": "Earth orbit claim",
                "publisher": "Snopes",
                "rating": "True",
                "bias_label": "Center",
                "type": "fact_check"
            }
        ],
        "processing_time_ms": 1500,
        "cached": False
    }

    with patch("app.api.routes.verify_claim", return_value=mock_orchestrator_result):
        # Test input field (primary)
        response = client.post("/api/verify", json={"input": "The Earth orbits the Sun"})
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "result-12345"
        assert data["trust_score"]["c_total"] == 0.95
        assert data["confidence"]["level"] == "high"
        assert len(data["sources"]) == 1

        # Test text field (alias for backward compatibility)
        response_text_alias = client.post("/api/verify", json={"text": "The Earth orbits the Sun"})
        assert response_text_alias.status_code == 200

def test_verify_endpoint_rejects_empty_input():
    response = client.post("/api/verify", json={"input": ""})
    # TDD §3.1 & B7: empty input is handled by validate_input_length and returns 400 Invalid Input
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "Invalid Input"

def test_verify_endpoint_rejects_whitespace_input():
    response = client.post("/api/verify", json={"input": "    "})
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "Invalid Input"

def test_verify_endpoint_rejects_opinion():
    response = client.post("/api/verify", json={"input": "I think summer is the greatest season in history."})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Non-Verifiable Claim"
    assert "opinion" in data["detail"].lower()

def test_verify_endpoint_rejects_question():
    response = client.post("/api/verify", json={"input": "Why do stars shine at night?"})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Non-Verifiable Claim"
    assert "question" in data["detail"].lower()

def test_verify_endpoint_rejects_payload_exceeding_max_length():
    huge_input = "a" * 10001
    response = client.post("/api/verify", json={"input": huge_input})
    assert response.status_code == 422

def test_get_result_found():
    valid_uuid = "a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d"
    mock_row = {
        "id": valid_uuid,
        "claim_id": "b2c3d4e5-f6a7-4b8c-9d0e-1f2a3b4c5d6e",
        "c_total": 0.88,
        "s_fact": 0.9,
        "b_ling": 0.1,
        "v_consensus": 0.8,
        "weights": {"w1": 0.5, "w2": 0.2, "w3": 0.3},
        "confidence_level": "high",
        "warnings": [],
        "source_citations": [],
        "linguistic_profile": None,
        "processing_time_ms": 1200,
        "claims": {"extracted_claim": "Historical verified claim"}
    }
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[mock_row])

    with patch("app.api.routes.get_supabase", return_value=mock_supabase):
        response = client.get(f"/api/results/{valid_uuid}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == valid_uuid
        assert data["claim"] == "Historical verified claim"
        assert data["trust_score"]["c_total"] == 0.88

def test_get_result_not_found():
    valid_uuid = "00000000-0000-0000-0000-000000000000"
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[])

    with patch("app.api.routes.get_supabase", return_value=mock_supabase):
        response = client.get(f"/api/results/{valid_uuid}")
        assert response.status_code == 404

def test_get_result_invalid_uuid_returns_404():
    """Verify non-blocking finding: Malformed UUID returns 404 instead of letting DB raise 500."""
    response = client.get("/api/results/non-existent-or-invalid-uuid")
    assert response.status_code == 404
    data = response.json()
    assert "Invalid result ID format" in data["detail"]


def test_get_result_legacy_row_is_labelled():
    """Rows saved before stance-aware scoring have no evidence_status; the API must say so."""
    row = {
        "id": "11111111-1111-1111-1111-111111111111", "c_total": 0.7, "s_fact": 0.5, "b_ling": 0.2,
        "v_consensus": 0.6, "confidence_level": "medium", "warnings": [],
        "claims": {"extracted_claim": "Old claim"}, "evidence_status": None,
    }
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[row])
    with patch("app.api.routes.get_supabase", return_value=mock_supabase):
        response = client.get("/api/results/11111111-1111-1111-1111-111111111111")
    assert response.status_code == 200
    body = response.json()
    assert body["confidence"]["evidence_status"] == "unverified"
    assert body["trust_score"]["v_refute"] == 0.0
    assert any("Legacy result" in w for w in body["confidence"]["warnings"])

