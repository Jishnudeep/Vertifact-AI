"""
OCR agent (Build Order 2.1, TDD §4.5): screenshot bytes -> claim text.

Engine: docTR (Apache-2.0, PyTorch, CPU) with its detection -> recognition -> reading-order
pipeline (`db_mobilenet_v3_large` + `crnn_mobilenet_v3_small`, the small CPU-friendly pair).
The predictor is built lazily on first use and cached for the life of the process (it is loaded
at startup when `settings.ocr_preload` is on). Inference runs in a worker thread, one image at a
time (`_INFER_LOCK`), so a burst of uploads cannot multiply model memory.

The image is validated and decoded entirely in memory (PNG/JPEG only, pixel count capped before
decode, longest edge resized to MAX_EDGE_PX); image bytes are never written to disk or stored.
Every failure is an `OCRError` carrying a fixed, user-safe message (never raw exception text);
the route maps it to a 400.
"""
import asyncio
import io
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import settings

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_EDGE_PX = 2048
# Decompression-bomb guard: a small PNG can declare billions of pixels. Checked from the header,
# before any pixel data is decoded.
MAX_IMAGE_PIXELS = 40_000_000
MIN_TEXT_CHARS = 10
# Phone-camera JPEGs are often identified as MPO; the first frame is the photo.
ALLOWED_FORMATS = frozenset({"PNG", "JPEG", "MPO"})

_INFER_LOCK = threading.Lock()
_LOAD_LOCK = threading.Lock()
_predictor: Any = None
# Queue callers here (no thread held) so waiting uploads cannot occupy the shared default
# executor; at most one inference thread exists at a time.
_QUEUE = asyncio.Semaphore(1)
MAX_QUEUE_WAIT_S = 30.0


class OCRError(Exception):
    """The image cannot be turned into claim text. `str(exc)` is safe to show the user."""


@dataclass(frozen=True)
class OCRResult:
    extracted_text: str
    confidence: float  # 0.0 to 1.0, character-weighted mean word confidence
    bounding_boxes: List[dict] = field(default_factory=list)  # debugging only; never in API output
    processing_time_ms: int = 0


def get_predictor() -> Any:
    """Build the docTR predictor once (double-checked lock: two cold requests must not both
    build it). Heavy imports stay inside so unit tests and cold imports skip them."""
    global _predictor
    if _predictor is None:
        with _LOAD_LOCK:
            if _predictor is None:
                from doctr.models import ocr_predictor

                logger.info("Loading OCR models...")
                _predictor = ocr_predictor(
                    det_arch="db_mobilenet_v3_large", reco_arch="crnn_mobilenet_v3_small", pretrained=True,
                )
    return _predictor


def _decode(image_bytes: bytes) -> Image.Image:
    """Validate and decode PNG/JPEG bytes into an RGB image no larger than MAX_EDGE_PX."""
    if not image_bytes:
        raise OCRError("The uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise OCRError("The image is larger than 5 MB.")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.format not in ALLOWED_FORMATS:
            raise OCRError("Only PNG and JPEG images are supported.")
        width, height = img.size
        if width * height > MAX_IMAGE_PIXELS:
            raise OCRError("The image dimensions are too large.")
        img.load()
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "P"):
            # Flatten transparency onto white: convert("RGB") alone turns it black.
            rgba = img.convert("RGBA")
            flat = Image.new("RGB", rgba.size, "white")
            flat.paste(rgba, mask=rgba.getchannel("A"))
            img = flat
        else:
            img = img.convert("RGB")
    except OCRError:
        raise
    except Image.DecompressionBombError as exc:
        raise OCRError("The image dimensions are too large.") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise OCRError("The file is not a readable PNG or JPEG image.") from exc
    longest = max(img.size)
    if longest > MAX_EDGE_PX:
        scale = MAX_EDGE_PX / longest
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    return img


def _recognize(img: Image.Image) -> OCRResult:
    """Run the OCR pipeline (blocking) and flatten the result in reading order."""
    with _INFER_LOCK:
        document = get_predictor()([np.asarray(img)])
    boxes: List[dict] = []
    lines: List[str] = []
    weighted = 0.0
    chars = 0
    for page in document.pages:
        for block in page.blocks:
            for line in block.lines:
                words = [w for w in line.words if w.value.strip()]
                if not words:
                    continue
                lines.append(" ".join(w.value.strip() for w in words))
                for w in words:
                    weighted += float(w.confidence) * len(w.value)
                    chars += len(w.value)
                    boxes.append({"text": w.value, "confidence": float(w.confidence), "box": w.geometry})
    text = " ".join(lines)
    return OCRResult(extracted_text=text, confidence=(weighted / chars) if chars else 0.0, bounding_boxes=boxes)


async def extract_text(image_bytes: bytes) -> OCRResult:
    """
    Screenshot bytes -> OCRResult. Raises OCRError (user-safe message) for an invalid image, a
    timeout, an engine failure, or fewer than MIN_TEXT_CHARS characters of text.
    """
    start = time.perf_counter()
    img = await asyncio.to_thread(_decode, image_bytes)
    try:
        await asyncio.wait_for(_QUEUE.acquire(), timeout=MAX_QUEUE_WAIT_S)
    except (TimeoutError, asyncio.TimeoutError) as exc:
        raise OCRError("Too many screenshots are being read right now. Try again shortly.") from exc
    try:
        # Load outside the timeout: the first call may download and build the models (see
        # settings.ocr_preload); the timeout budgets inference only.
        await asyncio.to_thread(get_predictor)
        result = await asyncio.wait_for(asyncio.to_thread(_recognize, img), timeout=settings.ocr_timeout_s)
    except (TimeoutError, asyncio.TimeoutError) as exc:
        raise OCRError("Reading the image took too long. Try typing the claim instead.") from exc
    except Exception as exc:  # noqa: BLE001 — never leak engine internals to the client
        logger.warning("OCR engine failed (%s)", type(exc).__name__)
        raise OCRError("Could not extract text. Try typing the claim instead.") from exc
    finally:
        _QUEUE.release()
    if len(result.extracted_text.strip()) < MIN_TEXT_CHARS:
        raise OCRError("Could not extract text. Try typing the claim instead.")
    return OCRResult(
        extracted_text=result.extracted_text,
        confidence=result.confidence,
        bounding_boxes=result.bounding_boxes,
        processing_time_ms=int((time.perf_counter() - start) * 1000),
    )
