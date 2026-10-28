"""
URL scraper (Build Order 2.2, TDD §4.6). Turns a user-supplied article URL into the claim to verify.

The claim is the article headline (`og:title`, else `<title>`, else the first `<h1>`) with the
site-name suffix removed. The article body (first 1,500 words) is returned for the Linguistic
agent. On a 401/402/403/451 wall, a JavaScript-only page, or a page with too little text, the
result is `degraded` and the body falls back to `og:description` (PRD §8). The fetch itself
(SSRF validation per hop, IP pinning, 3 redirects, 2 MB, `text/html` only) lives in
`app.services.page_fetch.fetch_html`. `scrape_url` never raises (cancellation still propagates).
"""
import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup

from app.config import settings
from app.services.page_fetch import BLOCKED_STATUSES, MIN_TEXT_CHARS, fetch_html
from app.utils.url_safety import UnsafeURLError

logger = logging.getLogger(__name__)

MAX_BODY_WORDS = 1500
MAX_HEADLINE_CHARS = 300
_DROP_TAGS = ["script", "style", "nav", "footer", "aside", "noscript", "template"]
_WS = re.compile(r"\s+")
_SITE_SUFFIX = re.compile(r"\s+(?:\||-|–|—|::|•)\s+([^|\-–—:•]+)$")
MAX_SITE_NAME_WORDS = 4
MIN_HEADLINE_WORDS = 3


@dataclass(frozen=True)
class ScrapeResult:
    headline: str
    body: str
    final_url: str
    degraded: bool
    error: Optional[str] = None


def _collapse(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _meta(soup: BeautifulSoup, **attrs: str) -> str:
    tag = soup.find("meta", attrs=attrs)
    content = tag.get("content") if tag else None
    return _collapse(content) if isinstance(content, str) else ""


def strip_site_suffix(title: str) -> str:
    """'Mars water found | Reuters' -> 'Mars water found'; leaves a title alone if stripping
    would leave fewer than MIN_HEADLINE_WORDS words or the tail is too long to be a site name."""
    match = _SITE_SUFFIX.search(title)
    if not match:
        return title
    head = title[: match.start()].strip()
    tail_words = len(match.group(1).split())
    if tail_words > MAX_SITE_NAME_WORDS or len(head.split()) < MIN_HEADLINE_WORDS:
        return title
    return head


def parse_article(html: bytes, encoding: Optional[str], *, og_only: bool = False) -> tuple[str, str, str]:
    """
    (headline, body, description): body is the first MAX_BODY_WORDS words of paragraph text.
    With `og_only` (a blocked page) only `og:title` counts as a headline: the <title> of a
    bot-wall or login page ("Just a moment...") is not the article's.
    """
    soup = BeautifulSoup(html, "html.parser", from_encoding=encoding)
    headline = _meta(soup, property="og:title")
    if not headline and not og_only and soup.title:
        headline = _collapse(soup.title.get_text())
    if not headline and not og_only:
        h1 = soup.find("h1")
        headline = _collapse(h1.get_text(" ")) if h1 else ""
    description = _meta(soup, property="og:description") or _meta(soup, name="description")

    for tag in soup(_DROP_TAGS):
        tag.decompose()
    root = soup.find("article") or soup
    # html.parser nests unclosed <p>; join only outermost ones so text is not repeated.
    paragraphs = [p for p in root.find_all("p") if p.find_parent("p") is None]
    words = _collapse(" ".join(p.get_text(" ") for p in paragraphs)).split()
    body = " ".join(words[:MAX_BODY_WORDS])
    return strip_site_suffix(headline)[:MAX_HEADLINE_CHARS], body, description


def _without_userinfo(url: str) -> str:
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    return urlunsplit(parts._replace(netloc=parts.netloc.rsplit("@", 1)[1]))


def _failure(url: str, reason: str) -> ScrapeResult:
    return ScrapeResult(headline="", body="", final_url=url, degraded=True, error=reason)


async def scrape_url(
    url: str,
    *,
    transport: Optional[httpx.AsyncBaseTransport] = None,
    resolver: Optional[Callable] = None,
) -> ScrapeResult:
    """
    Fetch `url` and extract (headline, body). `error` is a fixed, user-safe message (never raw
    exception text) when no headline could be obtained; the caller maps it to a 400.
    """
    try:
        page = await fetch_html(
            url, timeout_s=settings.scrape_timeout_s, accept_blocked=True,
            transport=transport, resolver=resolver,
        )
    except UnsafeURLError:
        return _failure(url, "That URL cannot be fetched (it must be a public http(s) address).")
    except (TimeoutError, httpx.TimeoutException):
        return _failure(url, "The page took too long to load.")
    except Exception as exc:  # noqa: BLE001 — contract: never raises
        logger.info("scrape failed (%s)", type(exc).__name__)
        return _failure(url, "Could not read that page.")

    blocked = page.status in BLOCKED_STATUSES
    try:
        headline, body, description = await asyncio.to_thread(
            lambda: parse_article(page.body, page.encoding, og_only=blocked))
    except Exception as exc:  # noqa: BLE001 — contract: never raises
        logger.info("scrape parse failed (%s)", type(exc).__name__)
        return _failure(page.final_url, "Could not read that page.")
    final_url = _without_userinfo(page.final_url)
    if not headline:
        message = ("That page blocked access and has no headline we can use."
                   if blocked else "Could not find a headline on that page.")
        return _failure(final_url, message)

    degraded = blocked or len(body) < MIN_TEXT_CHARS
    if degraded:
        body = description
    return ScrapeResult(headline=headline, body=body, final_url=final_url, degraded=degraded)
