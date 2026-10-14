"""
SSRF-guarded fetcher for pages cited by grounded search (Plan 6 Task 2, TDD §11).

Every hop is validated with `resolve_public_url` and the connection is pinned to one of
the validated IPs (never re-resolved: DNS rebinding). Redirects are followed by hand, at
most MAX_REDIRECTS. Only `text/html` bodies up to MAX_BODY_BYTES (wire and decoded) are
read; the only accepted content-encoding is gzip, decoded here with a bounded inflater so a
compression bomb cannot block the event loop. The whole fetch is bounded by
`settings.page_fetch_timeout_s`. Any failure, and any page with too little body text to be
evidence, returns None.
"""
import asyncio
import ipaddress
import logging
import re
import socket
import zlib
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
MIN_TEXT_CHARS = 200
MAX_TITLE_CHARS = 300
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_DROP_TAGS = ["script", "style", "nav", "footer", "noscript", "template"]
_WS = re.compile(r"\s+")
# Accept-Encoding is pinned to gzip: httpx would otherwise advertise br/zstd and decode them
# without an output bound.
_HEADERS = {
    "User-Agent": "VeriFactAI/1.0 (+cited-source verification)",
    "Accept": "text/html",
    "Accept-Encoding": "gzip",
}
_GZIP_WBITS = 31  # zlib: gzip header + trailer


@dataclass(frozen=True)
class PageText:
    final_url: str
    domain: str   # extract_domain(final host, lowercased, trailing dot stripped)
    title: str    # <title> or og:title, stripped
    text: str     # visible paragraph text, whitespace-collapsed, first 4000 chars


class _FetchRejected(Exception):
    """Internal: the page must not be used. Never escapes fetch_page_text."""


class _Progress:
    """The hop currently being fetched, so failures log the URL that actually failed."""

    def __init__(self, url: str) -> None:
        self.url = url


async def fetch_page_text(
    url: str,
    expected_domain: Optional[str] = None,
    *,
    transport: Optional[httpx.AsyncBaseTransport] = None,
    resolver: Optional[Callable] = None,
) -> Optional[PageText]:
    """
    Fetch `url` and return its title and paragraph text, or None on any failure.

    If `expected_domain` is given, the final hop must be https and its host must equal it
    or be a subdomain of it (same registrable domain), so a redirecting citation cannot be
    attributed to an outlet it isn't from. Never raises (cancellation still propagates).
    """
    timeout_s = settings.page_fetch_timeout_s
    progress = _Progress(url)
    try:
        return await asyncio.wait_for(
            _fetch(progress, expected_domain, transport, resolver or socket.getaddrinfo, timeout_s),
            timeout=timeout_s,
        )
    except (TimeoutError, httpx.TimeoutException):
        logger.info("page fetch timed out after %.1fs at %s", timeout_s, _loggable(progress.url))
    except (UnsafeURLError, _FetchRejected) as exc:
        logger.info("page fetch rejected (%s) at %s", exc, _loggable(progress.url))
    except Exception as exc:  # noqa: BLE001 — contract: never raises
        logger.warning("page fetch failed (%s) at %s", type(exc).__name__, _loggable(progress.url))
    return None


async def _fetch(
    progress: _Progress,
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
        redirects = 0
        while True:
            current = progress.url
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
                    progress.url = urlunsplit(urlsplit(urljoin(current, location))._replace(fragment=""))
                    continue
                if not 200 <= response.status_code < 300:
                    raise _FetchRejected(f"status {response.status_code}")
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if content_type != "text/html":
                    raise _FetchRejected(f"content-type '{content_type}'")
                final_host = _check_domain(current, expected_domain)
                body = await _read_capped(response)
                encoding = response.charset_encoding
            finally:
                await response.aclose()
            break

    title, text = await asyncio.to_thread(_extract, body, encoding)
    if len(text) < MIN_TEXT_CHARS:
        raise _FetchRejected(f"only {len(text)} chars of body text")
    return PageText(final_url=current, domain=_domain_of(final_host), title=title, text=text[:MAX_TEXT_CHARS])


def _domain_of(host: str) -> str:
    """extract_domain for hostnames; an IP-literal host is its own domain (extract_domain
    would cut an IPv6 address at the first ':')."""
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        return extract_domain(host)


async def _send_pinned(client: httpx.AsyncClient, url: str, ips: FrozenSet[str]) -> httpx.Response:
    """Send GET `url` to a validated IP, keeping the hostname for Host and TLS SNI/verification."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    host_header = f"[{host}]" if ":" in host else host
    if parts.port is not None:
        host_header = f"{host_header}:{parts.port}"
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


def _normalize_host(host: str) -> str:
    return host.strip().lower().rstrip(".")


def _check_domain(url: str, expected_domain: Optional[str]) -> str:
    """Return the normalized final host; reject it if it is not attributable to expected_domain."""
    parts = urlsplit(url)
    host = _normalize_host(parts.hostname or "")
    if expected_domain is None:
        return host
    if parts.scheme.lower() != "https":
        raise _FetchRejected("attributed page must be served over https")
    expected = _normalize_host(expected_domain).removeprefix("www.")
    if "." not in expected:
        raise _FetchRejected(f"expected domain '{expected}' is not a registrable domain")
    under_expected = host == expected or host.endswith("." + expected)
    if not under_expected or extract_domain(host) != extract_domain(expected):
        raise _FetchRejected(f"final host '{host}' is not under '{expected}'")
    return host


def _content_coding(response: httpx.Response) -> str:
    """'identity' or 'gzip'; anything else (deflate, br, zstd, stacked codings) is rejected."""
    codings = [
        c.strip().lower()
        for value in response.headers.get_list("content-encoding")
        for c in value.split(",")
        if c.strip()
    ]
    if not codings or codings == ["identity"]:
        return "identity"
    if codings in (["gzip"], ["x-gzip"]):
        return "gzip"
    raise _FetchRejected(f"content-encoding {codings}")


async def _read_capped(response: httpx.Response) -> bytes:
    """
    Read the raw body, decoding gzip ourselves. Wire bytes and decoded bytes are both capped
    at MAX_BODY_BYTES, and no single inflate call may produce more than one byte past the cap.
    """
    coding = _content_coding(response)
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise _FetchRejected("declared body over cap")
    inflater = zlib.decompressobj(wbits=_GZIP_WBITS) if coding == "gzip" else None
    chunks: List[bytes] = []
    wire = 0
    decoded = 0
    async for raw in response.aiter_raw():
        wire += len(raw)
        if wire > MAX_BODY_BYTES:
            raise _FetchRejected("body over cap")
        if inflater is None:
            decoded += len(raw)
            chunks.append(raw)
            continue
        pending = raw
        while pending:
            try:
                out = inflater.decompress(pending, MAX_BODY_BYTES - decoded + 1)
            except zlib.error as exc:
                raise _FetchRejected("corrupt gzip body") from exc
            decoded += len(out)
            if decoded > MAX_BODY_BYTES:
                raise _FetchRejected("decoded body over cap")
            chunks.append(out)
            # unconsumed_tail is non-empty only when the call stopped at max_length, and that
            # output already tripped the cap above; the drain stays under the same bound.
            pending = inflater.unconsumed_tail
    if inflater is not None and not inflater.eof:
        raise _FetchRejected("truncated gzip body")
    return b"".join(chunks)


def _collapse(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _extract(body: bytes, encoding: Optional[str]) -> tuple[str, str]:
    """Title and whitespace-collapsed outermost-<p> text (untruncated; caller trims)."""
    soup = BeautifulSoup(body, "html.parser", from_encoding=encoding)
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    title = _collapse(soup.title.get_text()) if soup.title else ""
    if not title:
        og = soup.find("meta", attrs={"property": "og:title"})
        content = og.get("content") if og else None
        title = _collapse(content) if isinstance(content, str) else ""
    # html.parser nests unclosed <p>; join only outermost ones so text is not repeated.
    paragraphs = [p for p in soup.find_all("p") if p.find_parent("p") is None]
    text = _collapse(" ".join(p.get_text(" ") for p in paragraphs))
    return title[:MAX_TITLE_CHARS], text


def _loggable(url: str) -> str:
    """Scheme + hostname[:port] + path: never userinfo or the query string."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        port = parts.port
    except ValueError:
        return "<unparseable url>"
    netloc = host if port is None else f"{host}:{port}"
    return f"{parts.scheme}://{netloc}{parts.path}"[:200]
