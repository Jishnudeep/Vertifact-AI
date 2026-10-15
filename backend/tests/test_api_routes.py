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


def test_schema_accepts_null_suggestions():
    from app.api.schemas import VerifyResponse, TrustScoreModel, ConfidenceModel, LinguisticProfileModel
    base_data = {
        "claim": "Claim text",
        "trust_score": TrustScoreModel(
            c_total=0.8, s_fact=0.8, b_ling=0.2, v_consensus=0.8, v_refute=0.0,
            weights={"w1": 0.5, "w2": 0.2, "w3": 0.3}
        ),
        "confidence": ConfidenceModel(level="high", warnings=[]),
        "linguistic_profile": LinguisticProfileModel(
            clickbait_syntax=0.0, emotional_framing=0.0, sensationalism=0.0, informational_density=0.8
        ),
        "sources": [],
        "processing_time_ms": 100,
        "cached": False,
    }
    resp_none = VerifyResponse(**base_data, search_suggestions_html=None)
    assert resp_none.search_suggestions_html is None

    resp_html = VerifyResponse(**base_data, search_suggestions_html="<div>Suggestions</div>")
    assert resp_html.search_suggestions_html == "<div>Suggestions</div>"


def test_get_result_not_marked_cached():
    valid_uuid = "22222222-2222-2222-2222-222222222222"
    mock_row = {
        "id": valid_uuid,
        "claim_id": valid_uuid,
        "c_total": 0.88,
        "s_fact": 0.9,
        "b_ling": 0.1,
        "v_consensus": 0.8,
        "weights": {"w1": 0.5, "w2": 0.2, "w3": 0.3},
        "confidence_level": "high",
        "evidence_status": "fact_checked",
        "warnings": [],
        "source_citations": [],
        "linguistic_profile": None,
        "processing_time_ms": 1200,
        "created_at": "2026-10-15T12:00:00Z",
        "claims": {"extracted_claim": "Historical claim"}
    }
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[mock_row])

    with patch("app.api.routes.get_supabase", return_value=mock_supabase):
        response = client.get(f"/api/results/{valid_uuid}")
        assert response.status_code == 200
        data = response.json()
        assert data["cached"] is False
        assert data["created_at"] == "2026-10-15T12:00:00Z"


def test_get_result_has_agents_and_suggestions():
    valid_uuid = "33333333-3333-3333-3333-333333333333"
    mock_row = {
        "id": valid_uuid,
        "c_total": 0.85,
        "s_fact": 0.9,
        "b_ling": 0.1,
        "v_consensus": 0.8,
        "weights": {"w1": 0.5, "w2": 0.2, "w3": 0.3},
        "confidence_level": "high",
        "evidence_status": "fact_checked",
        "warnings": [],
        "source_citations": [],
        "search_suggestions_html": "<div>Suggestions</div>",
        "agent_outputs": {
            "factcheck": {"degraded": False, "duration_ms": 150},
            "crossref": {"degraded": True, "duration_ms": 300, "error_details": {"code": "TIMEOUT"}},
            "linguistic": {"degraded": False, "duration_ms": 50},
            "scoring": {"raw_c_total": 0.92, "cap_rule": None},
        },
        "created_at": "2026-10-15T12:00:00Z",
        "claims": {"extracted_claim": "Agent test claim"}
    }
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[mock_row])

    with patch("app.api.routes.get_supabase", return_value=mock_supabase):
        response = client.get(f"/api/results/{valid_uuid}")
        assert response.status_code == 200
        data = response.json()
        assert data["search_suggestions_html"] == "<div>Suggestions</div>"
        assert len(data["agents"]) == 3
        fc_agent = next(a for a in data["agents"] if a["name"] == "factcheck")
        assert fc_agent["status"] == "ok"
        assert fc_agent["duration_ms"] == 150
        cr_agent = next(a for a in data["agents"] if a["name"] == "crossref")
        assert cr_agent["status"] == "degraded"
        assert cr_agent["error_code"] == "TIMEOUT"
        assert cr_agent["duration_ms"] == 300
        assert data["trust_score"]["raw_c_total"] == 0.92
        assert data["trust_score"]["cap_rule"] is None


def test_stored_legacy_row_formats():
    valid_uuid = "44444444-4444-4444-4444-444444444444"
    legacy_row = {
        "id": valid_uuid,
        "c_total": 0.70,
        "s_fact": 0.5,
        "b_ling": 0.2,
        "v_consensus": 0.6,
        "weights": {"w1": 0.5, "w2": 0.2, "w3": 0.3},
        "confidence_level": "medium",
        "evidence_status": None,
        "agent_outputs": None,
        "warnings": [],
        "source_citations": [],
        "created_at": "2026-09-01T12:00:00Z",
        "claims": {"extracted_claim": "Legacy claim"}
    }
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[legacy_row])

    with patch("app.api.routes.get_supabase", return_value=mock_supabase):
        response = client.get(f"/api/results/{valid_uuid}")
        assert response.status_code == 200
        data = response.json()
        assert data["agents"] == []
        assert data["trust_score"]["cap_rule"] is None
        assert data["confidence"]["evidence_status"] == "unverified"
        assert any("Legacy result" in w for w in data["confidence"]["warnings"])


def test_rate_limit_sets_retry_after():
    from app.api.routes import limiter
    limiter.reset()
    mock_result = {
        "id": "result-1", "claim": "claim",
        "trust_score": {"c_total": 0.8, "s_fact": 1.0, "b_ling": 0.1, "v_consensus": 0.8, "weights": {"w1": 0.5, "w2": 0.2, "w3": 0.3}},
        "confidence": {"level": "high", "warnings": []},
        "linguistic_profile": {"clickbait_syntax": 0.0, "emotional_framing": 0.0, "sensationalism": 0.0, "informational_density": 0.9},
        "sources": [], "processing_time_ms": 100, "cached": False, "agents": []
    }
    with patch("app.api.routes.verify_claim", return_value=mock_result):
        for _ in range(10):
            res = client.post("/api/verify", json={"input": "test claim"})
            assert res.status_code == 200
        res_limit = client.post("/api/verify", json={"input": "test claim"})
        assert res_limit.status_code == 429
        assert res_limit.headers.get("Retry-After") == "60"
    limiter.reset()


def test_recent_route_not_shadowed():
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    with patch("app.api.routes.get_supabase", return_value=mock_supabase):
        response = client.get("/api/results/recent")
        assert response.status_code == 200
        assert response.json() == []


def test_recent_only_featured_newest_first():
    mock_supabase = MagicMock()
    mock_table = mock_supabase.table.return_value
    mock_select = mock_table.select.return_value
    mock_eq = mock_select.eq.return_value
    mock_order = mock_eq.order.return_value
    mock_order.limit.return_value.execute.return_value = MagicMock(data=[])

    with patch("app.api.routes.get_supabase", return_value=mock_supabase):
        response = client.get("/api/results/recent")
        assert response.status_code == 200
        mock_select.eq.assert_called_once_with("featured", True)
        mock_eq.order.assert_called_once_with("created_at", desc=True)


def test_recent_limit_clamped():
    mock_supabase = MagicMock()
    mock_table = mock_supabase.table.return_value
    mock_select = mock_table.select.return_value
    mock_eq = mock_select.eq.return_value
    mock_order = mock_eq.order.return_value
    mock_order.limit.return_value.execute.return_value = MagicMock(data=[])

    with patch("app.api.routes.get_supabase", return_value=mock_supabase):
        # limit=500 clamped to 20
        response = client.get("/api/results/recent?limit=500")
        assert response.status_code == 200
        mock_order.limit.assert_called_with(20)

        # limit=0 clamped to 1
        response = client.get("/api/results/recent?limit=0")
        assert response.status_code == 200
        mock_order.limit.assert_called_with(1)


def test_recent_field_allow_list():
    sample_row = {
        "id": "11111111-1111-1111-1111-111111111111",
        "c_total": 0.85,
        "evidence_status": "fact_checked",
        "confidence_level": "high",
        "created_at": "2026-10-15T12:00:00Z",
        "claims": {"extracted_claim": "Moon landing was real"},
    }
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[sample_row])

    with patch("app.api.routes.get_supabase", return_value=mock_supabase):
        response = client.get("/api/results/recent")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        item = data[0]
        assert set(item.keys()) == {"id", "claim", "c_total", "evidence_status", "confidence_level", "created_at"}
        assert item["id"] == "11111111-1111-1111-1111-111111111111"
        assert item["claim"] == "Moon landing was real"
        assert item["c_total"] == 0.85
        assert item["evidence_status"] == "fact_checked"
        assert item["confidence_level"] == "high"
        assert item["created_at"] == "2026-10-15T12:00:00Z"


def test_recent_empty_list():
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])

    with patch("app.api.routes.get_supabase", return_value=mock_supabase):
        response = client.get("/api/results/recent")
        assert response.status_code == 200
        assert response.json() == []




