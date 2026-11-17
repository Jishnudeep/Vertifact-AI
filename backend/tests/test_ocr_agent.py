import asyncio
import io
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from app.agents import ocr_agent
from app.agents.ocr_agent import OCRError, extract_text


@pytest.fixture(autouse=True)
def fresh_queue(monkeypatch):
    """A semaphore binds to the first event loop that contends for it; give each test its own."""
    monkeypatch.setattr(ocr_agent, "_QUEUE", asyncio.Semaphore(1))


def png(size=(200, 100), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, fmt)
    return buf.getvalue()


def word(value: str, conf: float):
    return SimpleNamespace(value=value, confidence=conf, geometry=((0.0, 0.0), (1.0, 1.0)))


def fake_predictor(lines, seen=None):
    """docTR-shaped result: one page, one block, one line per entry of `lines`."""
    def predict(pages):
        if seen is not None:
            seen.extend(pages)
        doc_lines = [SimpleNamespace(words=[word(v, c) for v, c in ln]) for ln in lines]
        return SimpleNamespace(pages=[SimpleNamespace(blocks=[SimpleNamespace(lines=doc_lines)])])
    return predict


@pytest.fixture
def engine(monkeypatch):
    def install(lines, seen=None):
        monkeypatch.setattr(ocr_agent, "get_predictor", lambda: fake_predictor(lines, seen))
    return install


@pytest.mark.anyio
async def test_extracts_text_in_reading_order_with_weighted_confidence(engine):
    engine([[("NASA", 1.0), ("confirms", 0.5)], [("water", 1.0), ("on", 1.0), ("Mars", 1.0)]])
    res = await extract_text(png())
    assert res.extracted_text == "NASA confirms water on Mars"
    assert len(res.bounding_boxes) == 5
    # 4 + 8*0.5 + 5 + 2 + 4 over 23 characters
    assert res.confidence == pytest.approx((4 + 4 + 5 + 2 + 4) / 23)
    assert res.processing_time_ms >= 0


@pytest.mark.anyio
async def test_blank_words_are_dropped(engine):
    engine([[("  ", 0.9), ("Breaking news today", 0.8)]])
    assert (await extract_text(png())).extracted_text == "Breaking news today"


@pytest.mark.anyio
async def test_too_little_text_is_rejected(engine):
    engine([[("Hi", 0.9)]])
    with pytest.raises(OCRError, match="Could not extract text"):
        await extract_text(png())


@pytest.mark.anyio
@pytest.mark.parametrize("payload,message", [
    pytest.param(b"", "empty", id="empty"),
    pytest.param(b"not an image at all", "not a readable", id="garbage"),
    pytest.param(png(fmt="GIF"), "Only PNG and JPEG", id="gif"),
    pytest.param(b"x" * (5 * 1024 * 1024 + 1), "larger than 5 MB", id="over-5mb"),
])
async def test_invalid_images_rejected_before_inference(monkeypatch, payload, message):
    monkeypatch.setattr(ocr_agent, "get_predictor", lambda: pytest.fail("engine must not run"))
    with pytest.raises(OCRError, match=message):
        await extract_text(payload)


@pytest.mark.anyio
async def test_oversized_dimensions_rejected(monkeypatch):
    monkeypatch.setattr(ocr_agent, "MAX_IMAGE_PIXELS", 1000)
    monkeypatch.setattr(ocr_agent, "get_predictor", lambda: pytest.fail("engine must not run"))
    with pytest.raises(OCRError, match="dimensions"):
        await extract_text(png((100, 100)))


@pytest.mark.anyio
async def test_large_image_resized_to_max_edge(engine):
    seen = []
    engine([[("A reasonably long claim", 0.9)]], seen)
    await extract_text(png((4096, 1024), fmt="JPEG"))
    height, width, _ = seen[0].shape
    assert (width, height) == (2048, 512)


@pytest.mark.anyio
async def test_engine_failure_is_masked(monkeypatch):
    def boom():
        raise RuntimeError("secret internal path")
    monkeypatch.setattr(ocr_agent, "get_predictor", boom)
    with pytest.raises(OCRError) as info:
        await extract_text(png())
    assert "secret" not in str(info.value)


@pytest.mark.anyio
async def test_timeout(monkeypatch):
    monkeypatch.setattr(ocr_agent.settings, "ocr_timeout_s", 0.05)
    monkeypatch.setattr(ocr_agent, "get_predictor", lambda: (lambda pages: time.sleep(0.3)))
    with pytest.raises(OCRError, match="too long"):
        await extract_text(png())
    # F4: the thread is still running, so the slot stays taken until it finishes, then frees itself.
    assert ocr_agent._QUEUE.locked()
    for _ in range(100):
        if not ocr_agent._QUEUE.locked():
            break
        await asyncio.sleep(0.02)
    assert not ocr_agent._QUEUE.locked()


@pytest.mark.anyio
async def test_a_caller_after_a_timeout_waits_for_the_slot_without_a_thread(monkeypatch):
    """F4: the next upload queues on the semaphore until the timed-out inference ends."""
    monkeypatch.setattr(ocr_agent.settings, "ocr_timeout_s", 0.05)
    monkeypatch.setattr(ocr_agent, "MAX_QUEUE_WAIT_S", 0.05)
    monkeypatch.setattr(ocr_agent, "get_predictor", lambda: (lambda pages: time.sleep(0.3)))
    with pytest.raises(OCRError, match="too long"):
        await extract_text(png())
    with pytest.raises(OCRError, match="Too many"):
        await extract_text(png())
    for _ in range(100):
        if not ocr_agent._QUEUE.locked():
            break
        await asyncio.sleep(0.02)
    assert not ocr_agent._QUEUE.locked()


@pytest.mark.anyio
async def test_cancelled_request_keeps_the_slot_until_inference_ends(monkeypatch):
    """F4: a client disconnect mid-inference must not free the slot while the thread still runs."""
    monkeypatch.setattr(ocr_agent, "get_predictor", lambda: (lambda pages: time.sleep(0.3)))
    task = asyncio.ensure_future(extract_text(png()))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ocr_agent._QUEUE.locked()
    for _ in range(100):
        if not ocr_agent._QUEUE.locked():
            break
        await asyncio.sleep(0.02)
    assert not ocr_agent._QUEUE.locked()


@pytest.mark.anyio
async def test_transparent_png_flattened_onto_white(engine):
    seen = []
    engine([[("A reasonably long claim", 0.9)]], seen)
    buf = io.BytesIO()
    Image.new("RGBA", (50, 50), (0, 0, 0, 0)).save(buf, "PNG")
    await extract_text(buf.getvalue())
    assert seen[0].min() == 255  # fully transparent -> white, not black


@pytest.mark.anyio
async def test_real_decompression_bomb_header_rejected(monkeypatch):
    monkeypatch.setattr(ocr_agent, "get_predictor", lambda: pytest.fail("engine must not run"))
    buf = io.BytesIO()
    Image.new("1", (30000, 30000)).save(buf, "PNG")  # ~1 GP declared, tiny on the wire
    with pytest.raises(OCRError, match="dimensions"):
        await extract_text(buf.getvalue())


@pytest.mark.anyio
async def test_queue_wait_is_bounded(monkeypatch):
    monkeypatch.setattr(ocr_agent, "MAX_QUEUE_WAIT_S", 0.05)
    await ocr_agent._QUEUE.acquire()
    try:
        with pytest.raises(OCRError, match="Too many"):
            await extract_text(png())
    finally:
        ocr_agent._QUEUE.release()
