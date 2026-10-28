import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.agents.ocr_agent import OCRError, OCRResult
from app.agents.scraper_agent import ScrapeResult
from app.api.routes import limiter
from app.main import app

client = TestClient(app)

PIPELINE_RESULT = {
    "id": "r1",
    "claim": "NASA confirms water ice on the Moon",
    "trust_score": {"c_total": 0.6, "s_fact": 0.5, "b_ling": 0.2, "v_consensus": 0.5,
                    "weights": {"w1": 0.5, "w2": 0.2, "w3": 0.3}},
    "confidence": {"level": "medium", "warnings": []},
    "linguistic_profile": {"clickbait_syntax": 0.0, "emotional_framing": 0.0,
                           "sensationalism": 0.0, "informational_density": 0.5},
    "sources": [],
    "processing_time_ms": 10,
    "cached": False,
}


@pytest.fixture(autouse=True)
def _reset_limiter():
    limiter.reset()
    yield
    limiter.reset()


def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(buf, "PNG")
    return buf.getvalue()


def test_url_input_verifies_the_scraped_headline():
    scraped = ScrapeResult("NASA confirms water ice on the Moon", "body", "https://reuters.com/a", False)
    with patch("app.api.routes.scrape_url", AsyncMock(return_value=scraped)) as scrape, \
         patch("app.api.routes.verify_claim", AsyncMock(return_value=PIPELINE_RESULT)) as verify:
        res = client.post("/api/verify", json={
            "input": "https://reuters.com/a?utm_source=x&id=7", "input_type": "url"})
    assert res.status_code == 200
    scrape.assert_awaited_once_with("https://reuters.com/a?id=7")  # tracking parameter stripped
    verify.assert_awaited_once_with("NASA confirms water ice on the Moon", "url")
    body = res.json()
    assert body["source_url"] == "https://reuters.com/a" and body["scrape_degraded"] is False


def test_url_scrape_failure_is_400_and_never_scores():
    scraped = ScrapeResult("", "", "https://x.example/a", True, "Could not read that page.")
    with patch("app.api.routes.scrape_url", AsyncMock(return_value=scraped)), \
         patch("app.api.routes.verify_claim", AsyncMock()) as verify:
        res = client.post("/api/verify", json={"input": "https://x.example/a", "input_type": "url"})
    assert res.status_code == 400
    assert res.json()["code"] == "invalid_input" and res.json()["detail"] == "Could not read that page."
    verify.assert_not_awaited()


def test_text_input_does_not_scrape():
    with patch("app.api.routes.scrape_url", AsyncMock()) as scrape, \
         patch("app.api.routes.verify_claim", AsyncMock(return_value=PIPELINE_RESULT)):
        res = client.post("/api/verify", json={"input": "The Earth orbits the Sun"})
    assert res.status_code == 200 and res.json()["source_url"] is None
    scrape.assert_not_awaited()


def test_image_endpoint_ocrs_then_verifies():
    ocr = OCRResult("NASA confirms water ice on the Moon", 0.92)
    with patch("app.api.routes.extract_text", AsyncMock(return_value=ocr)) as extract, \
         patch("app.api.routes.verify_claim", AsyncMock(return_value=PIPELINE_RESULT)) as verify:
        res = client.post("/api/verify/image", files={"image": ("s.png", png_bytes(), "image/png")})
    assert res.status_code == 200
    extract.assert_awaited_once()
    verify.assert_awaited_once_with("NASA confirms water ice on the Moon", "image")
    body = res.json()
    assert body["ocr_extracted_text"] == PIPELINE_RESULT["claim"]
    assert body["ocr_confidence"] == 0.92


def test_image_wrong_content_type_rejected_without_ocr():
    with patch("app.api.routes.extract_text", AsyncMock()) as extract:
        res = client.post("/api/verify/image", files={"image": ("a.gif", b"GIF89a", "image/gif")})
    assert res.status_code == 400 and res.json()["code"] == "invalid_input"
    extract.assert_not_awaited()


def test_image_ocr_failure_is_400():
    with patch("app.api.routes.extract_text", AsyncMock(side_effect=OCRError("Could not extract text."))), \
         patch("app.api.routes.verify_claim", AsyncMock()) as verify:
        res = client.post("/api/verify/image", files={"image": ("s.png", png_bytes(), "image/png")})
    assert res.status_code == 400 and res.json()["detail"] == "Could not extract text."
    verify.assert_not_awaited()


def test_oversized_upload_refused_before_ocr():
    big = b"PNG" + b"0" * (6 * 1024 * 1024)
    with patch("app.api.routes.extract_text", AsyncMock()) as extract:
        res = client.post("/api/verify/image", files={"image": ("s.png", big, "image/png")})
    assert res.status_code == 400 and "5 MB" in res.json()["detail"]
    extract.assert_not_awaited()


def test_chunked_oversized_upload_aborted_at_cap():
    """No Content-Length: the stream itself is cut off at the cap."""
    def chunks():
        for _ in range(10):
            yield b"0" * (1024 * 1024)

    with patch("app.api.routes.extract_text", AsyncMock()) as extract:
        res = client.post("/api/verify/image", content=chunks(),
                          headers={"content-type": "multipart/form-data; boundary=x"})
    assert res.status_code == 400 and "5 MB" in res.json()["detail"]
    extract.assert_not_awaited()


def test_large_upload_is_not_spooled_to_disk():
    """A 2 MB part would exceed Starlette's 1 MB default spool and hit a temp file."""
    body = b"PNG" + b"0" * (2 * 1024 * 1024)
    seen = {}

    async def spy(data):
        seen["n"] = len(data)
        raise OCRError("stop")

    with patch("app.api.routes.extract_text", spy),          patch("tempfile.SpooledTemporaryFile.rollover", side_effect=AssertionError("spooled to disk")):
        res = client.post("/api/verify/image", files={"image": ("s.png", body, "image/png")})
    assert res.status_code == 400 and seen["n"] == len(body)


def test_non_multipart_and_missing_field_rejected():
    assert client.post("/api/verify/image", json={"a": 1}).status_code == 400
    res = client.post("/api/verify/image", files={"other": ("s.png", png_bytes(), "image/png")})
    assert res.status_code == 400


def test_image_input_type_on_text_route_rejected():
    with patch("app.api.routes.verify_claim", AsyncMock()) as verify:
        res = client.post("/api/verify", json={"input": "some claim text here", "input_type": "image"})
    assert res.status_code == 400
    verify.assert_not_awaited()
