"""
SSRF-guarded fetcher for pages cited by grounded search (Plan 6 Task 2, TDD §11).

Every hop is validated with `resolve_public_url` and the connection is pinned to one of
the validated IPs (never re-resolved: DNS rebinding). Redirects are followed by hand, at
most MAX_REDIRECTS. Only `text/html` bodies up to MAX_BODY_BYTES are read, and the whole
fetch is bounded by `settings.page_fetch_timeout_s`. Any failure returns None.
"""
import asyncio
import logging
import re
import socket
from dataclasses import dataclass
from typing import Callable, FrozenSet, List, Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup

from app.config import settings
from app.utils.text_processing import extract_domain
from app.utils.url_safety import UnsafeURLError, resolve_public_url

logger = logging.getLogger(__name__)

MAX_REDIRECTS = 3
MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_TEXT_CHARS = 4000
MAX_TITLE_CHARS = 300
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_DROP_TAGS = ["script", "style", "nav", "footer"]
_WS = re.compile(r"\s+")
_HEADERS = {
    "User-Agent": "VeriFactAI/1.0 (+cited-source verification)",
    "Accept": "text/html,application/xhtml+xml;q=0.9",
}


@dataclass(frozen=True)
class PageText:
    final_url: str
    domain: str   # extract_domain(final_url)
    title: str    # <title> or og:title, stripped
    text: str     # visible paragraph text, whitespace-collapsed, first 4000 chars


class _FetchRejected(Exception):
    """Internal: the page must not be used. Never escapes fetch_page_text."""


async def fetch_page_text(
    url: str,
    expected_domain: Optional[str] = None,
    *,
    transport: Optional[httpx.AsyncBaseTransport] = None,
    resolver: Optional[Callable] = None,
) -> Optional[PageText]:
    """
    Fetch `url` and return its title and paragraph text, or None on any failure.

    If `expected_domain` is given, the final URL's host must equal it or be a subdomain of
    it, so a redirecting citation cannot be attributed to an outlet it isn't from.
    Never raises (cancellation still propagates).
    """
    timeout_s = settings.page_fetch_timeout_s
    try:
        return await asyncio.wait_for(
            _fetch(url, expected_domain, transport, resolver or socket.getaddrinfo, timeout_s),
            timeout=timeout_s,
        )
    except (TimeoutError, httpx.TimeoutException):
        logger.info("page fetch timed out after %.1fs: %s", timeout_s, _loggable(url))
    except (UnsafeURLError, _FetchRejected) as exc:
        logger.info("page fetch rejected (%s): %s", exc, _loggable(url))
    except Exception as exc:  # noqa: BLE001 — contract: never raises
        logger.warning("page fetch failed (%s): %s", type(exc).__name__, _loggable(url))
    return None


async def _fetch(
    url: str,
    expected_domain: Optional[str],
    transport: Optional[httpx.AsyncBaseTransport],
    resolver: Callable,
    timeout_s: float,
) -> PageText:
    # trust_env=False: an env proxy would bypass the IP pinning below.
    # No keep-alive: connections are pooled by IP, so a later hop to another hostname on the
    # same IP would reuse a TLS session verified for the previous hostname.
    async with httpx.AsyncClient(
        transport=transport, follow_redirects=False, timeout=timeout_s, trust_env=False,
        limits=httpx.Limits(max_keepalive_connections=0),
    ) as client:
        current = url
        redirects = 0
        while True:
            _, ips = await asyncio.to_thread(resolve_public_url, current, resolver)
            response = await _send_pinned(client, current, ips)
            try:
                if response.status_code in REDIRECT_STATUSES:
                    location = response.headers.get("location")
                    if not location:
                        raise _FetchRejected("redirect without Location")
                    redirects += 1
                    if redirects > MAX_REDIRECTS:
                        raise _FetchRejected("too many redirects")
                    current = urlunsplit(urlsplit(urljoin(current, location))._replace(fragment=""))
                    continue
                if not 200 <= response.status_code < 300:
                    raise _FetchRejected(f"status {response.status_code}")
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if content_type != "text/html":
                    raise _FetchRejected(f"content-type '{content_type}'")
                _check_domain(current, expected_domain)
                body = await _read_capped(response)
                encoding = response.charset_encoding
            finally:
                await response.aclose()
            break

    title, text = await asyncio.to_thread(_extract, body, encoding)
    return PageText(final_url=current, domain=extract_domain(current), title=title, text=text)


async def _send_pinned(client: httpx.AsyncClient, url: str, ips: FrozenSet[str]) -> httpx.Response:
    """Send GET `url` to a validated IP, keeping the hostname for Host and TLS SNI/verification."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    host_header = host if parts.port is None else f"{host}:{parts.port}"
    last_error: Optional[Exception] = None
    for ip in _ordered(ips):
        netloc = f"[{ip}]" if ":" in ip else ip
        if parts.port is not None:
            netloc = f"{netloc}:{parts.port}"
        pinned = urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))
        request = client.build_request(
            "GET", pinned, headers={**_HEADERS, "Host": host_header}, extensions={"sni_hostname": host},
        )
        try:
            return await client.send(request, stream=True)
        except httpx.ConnectError as exc:  # try the next validated address
            last_error = exc
    raise last_error or _FetchRejected("no address to connect to")


def _ordered(ips: FrozenSet[str]) -> List[str]:
    """IPv4 first, then IPv6; sorted for determinism."""
    return sorted(ips, key=lambda ip: (":" in ip, ip))


def _check_domain(url: str, expected_domain: Optional[str]) -> None:
    if expected_domain is None:
        return
    expected = expected_domain.strip().lower().rstrip(".").removeprefix("www.")
    host = (urlsplit(url).hostname or "").rstrip(".")
    if not expected or not (host == expected or host.endswith("." + expected)):
        raise _FetchRejected(f"final host '{host}' is not under '{expected}'")


async def _read_capped(response: httpx.Response) -> bytes:
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise _FetchRejected("declared body over cap")
    chunks: List[bytes] = []
    total = 0
    # aiter_bytes yields decoded bytes, so the cap also bounds compressed bodies.
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            raise _FetchRejected("body over cap")
        chunks.append(chunk)
    return b"".join(chunks)


def _collapse(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _extract(body: bytes, encoding: Optional[str]) -> tuple[str, str]:
    soup = BeautifulSoup(body, "html.parser", from_encoding=encoding)
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    title = _collapse(soup.title.get_text()) if soup.title else ""
    if not title:
        og = soup.find("meta", attrs={"property": "og:title"})
        content = og.get("content") if og else None
        title = _collapse(content) if isinstance(content, str) else ""
    text = _collapse(" ".join(p.get_text(" ") for p in soup.find_all("p")))
    return title[:MAX_TITLE_CHARS], text[:MAX_TEXT_CHARS]


def _loggable(url: str) -> str:
    """Scheme + host + path only: query strings can carry tokens."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))[:200]
