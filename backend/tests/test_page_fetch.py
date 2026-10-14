import asyncio
import gzip
import socket
import zlib
from typing import Callable, Dict, List, Union

import httpx
import httpx._decoders
import pytest

from app.config import settings
from app.services import page_fetch
from app.services.page_fetch import (
    MAX_BODY_BYTES,
    MIN_TEXT_CHARS,
    PageText,
    fetch_page_text,
)

PUBLIC_IPS: Dict[str, Union[str, List[str]]] = {
    "reuters.com": "93.184.216.34",
    "www.reuters.com": "93.184.216.35",
    "vertexaisearch.cloud.google.com": "142.250.80.1",
    "hop.example": "93.184.216.40",
}

LONG = ("Scientists say the deposits are stable over geological time. " * 5).strip()
ARTICLE = (
    "<html><head><title>  Fact check:   moon ice  </title>"
    "<style>p{color:red}</style><script>var p = '<p>nope</p>';</script></head>"
    "<body><nav><p>Home | World</p></nav>"
    f"<p>NASA  confirmed\n water ice</p><p>on the Moon.</p><p>{LONG}</p>"
    "<footer><p>Copyright</p></footer></body></html>"
)
ARTICLE_TEXT = "NASA confirmed water ice on the Moon. " + LONG
GOOGLE_REDIRECT = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc"


def make_resolver(mapping: Dict[str, Union[str, List[str]]]) -> Callable:
    def _resolve(host, port, proto=0, **kwargs):
        if host not in mapping:
            raise socket.gaierror("unknown host")
        ips = mapping[host]
        ips = [ips] if isinstance(ips, str) else ips
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, proto, "", (ip, port))
                for ip in ips]
    return _resolve


def html_response(body: str = ARTICLE, **kwargs) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=body, **kwargs)


def redirect(location: str) -> httpx.Response:
    return httpx.Response(302, headers={"location": location})


class WireStream(httpx.AsyncByteStream):
    """Unread network-like stream: httpx pre-reads Response(content=bytes), which a real
    socket never does, and the fetcher reads with aiter_raw()."""

    def __init__(self, data: bytes, chunk: int = 64 * 1024):
        self._data = data
        self._chunk = chunk

    async def __aiter__(self):
        for i in range(0, len(self._data), self._chunk):
            yield self._data[i:i + self._chunk]


class Recorder:
    """MockTransport handler routing on the Host header (requests go to the pinned IP)."""

    def __init__(self, routes: Dict[str, httpx.Response]):
        self.routes = routes
        self.requests: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = request.headers["host"] + request.url.path
        return live(self.routes.get(key, httpx.Response(404)))


def live(response: httpx.Response) -> httpx.Response:
    """A fresh unread copy of a pre-read Response, so a route can be served more than once."""
    try:
        body = response.content
    except httpx.ResponseNotRead:  # already a live stream: single use
        return response
    return httpx.Response(response.status_code, headers=response.headers, stream=WireStream(body))


def encoded_response(coding: str, wire_body: bytes) -> httpx.Response:
    """Content-encoded response whose wire bytes are left undecoded (content= would decode them)."""
    headers = {"content-type": "text/html", "content-encoding": coding}
    return httpx.Response(200, headers=headers, stream=WireStream(wire_body))


async def fetch(url: str, expected: Union[str, None] = None, *, rec, mapping=None) -> Union[PageText, None]:
    return await fetch_page_text(
        url, expected, transport=httpx.MockTransport(rec), resolver=make_resolver(mapping or PUBLIC_IPS),
    )


@pytest.mark.anyio
async def test_extracts_title_and_text():
    rec = Recorder({"reuters.com/article": html_response()})
    page = await fetch("https://reuters.com/article", "reuters.com", rec=rec)
    assert isinstance(page, PageText)
    assert page.final_url == "https://reuters.com/article"
    assert page.domain == "reuters.com"
    assert page.title == "Fact check: moon ice"
    assert page.text == ARTICLE_TEXT
    # Pinned: connect to the validated IP, keep the hostname in Host and SNI.
    req = rec.requests[0]
    assert req.url.host == "93.184.216.34"
    assert req.headers["host"] == "reuters.com"
    assert req.extensions["sni_hostname"] == "reuters.com"
    assert req.headers["accept"] == "text/html"
    assert req.headers["accept-encoding"] == "gzip"

    # og:title is the fallback when there is no <title>.
    rec.routes["reuters.com/og"] = html_response(
        f'<html><head><meta property="og:title" content=" OG headline "></head><body><p>{LONG}</p></body></html>'
    )
    page = await fetch("https://reuters.com/og", rec=rec)
    assert page is not None and page.title == "OG headline"


@pytest.mark.anyio
async def test_page_without_body_text_returns_none():
    shell = (
        "<html><head><title>News App</title><script src='/app.js'></script></head>"
        "<body><div id='root'></div>"
        # Long enough to pass MIN_TEXT_CHARS on its own, so None proves noscript is dropped.
        "<noscript><p>" + "Enable JavaScript to read this article. " * 8 + "</p></noscript>"
        "<template><p>" + "templated " * 50 + "</p></template></body></html>"
    )
    rec = Recorder({"reuters.com/app": html_response(shell)})
    assert await fetch("https://reuters.com/app", "reuters.com", rec=rec) is None
    # Just under the minimum is still not evidence.
    rec.routes["reuters.com/short"] = html_response(f"<p>{'x' * (MIN_TEXT_CHARS - 1)}</p>")
    assert await fetch("https://reuters.com/short", rec=rec) is None


@pytest.mark.anyio
async def test_unclosed_paragraphs_are_not_duplicated():
    body = "<html><body><p>" + "alpha " * 40 + "<p>" + "beta " * 40 + "<p>" + "gamma " * 40 + "</body></html>"
    rec = Recorder({"reuters.com/p": html_response(body)})
    page = await fetch("https://reuters.com/p", rec=rec)
    assert page is not None
    assert page.text == " ".join(["alpha"] * 40 + ["beta"] * 40 + ["gamma"] * 40)


@pytest.mark.anyio
async def test_private_ip_is_refused():
    rec = Recorder({"intranet.example/": html_response()})
    page = await fetch("https://intranet.example/", rec=rec, mapping={"intranet.example": "10.0.0.5"})
    assert page is None
    assert rec.requests == []


@pytest.mark.anyio
async def test_redirect_to_private_host_is_refused():
    rec = Recorder({
        "reuters.com/a": redirect("http://metadata.internal/latest"),
        "metadata.internal/latest": html_response(),
    })
    mapping = {**PUBLIC_IPS, "metadata.internal": "169.254.169.254"}
    assert await fetch("https://reuters.com/a", rec=rec, mapping=mapping) is None
    assert [r.headers["host"] for r in rec.requests] == ["reuters.com"]


@pytest.mark.anyio
@pytest.mark.parametrize("location", [
    "file:///etc/passwd",
    "data:text/html,x",
    "https://reuters.com@evil.example/",
])
async def test_redirect_to_unsafe_location_returns_none(location):
    rec = Recorder({"reuters.com/a": redirect(location), "evil.example/": html_response()})
    mapping = {**PUBLIC_IPS, "evil.example": "93.184.216.99"}
    assert await fetch("https://reuters.com/a", rec=rec, mapping=mapping) is None
    assert len(rec.requests) == 1


@pytest.mark.anyio
async def test_more_than_three_redirects_returns_none():
    rec = Recorder({
        "reuters.com/0": redirect("/1"),       # relative Location resolves against the current URL
        "reuters.com/1": redirect("https://hop.example/2"),
        "hop.example/2": redirect("https://reuters.com/3"),
        "reuters.com/3": redirect("https://reuters.com/4"),
        "reuters.com/4": html_response(),
    })
    # Three redirects (from /1) are allowed...
    page = await fetch("https://reuters.com/1", rec=rec)
    assert page is not None and page.final_url == "https://reuters.com/4"
    # ...a fourth is not.
    assert await fetch("https://reuters.com/0", rec=rec) is None


@pytest.mark.anyio
async def test_non_html_returns_none():
    rec = Recorder({
        "reuters.com/data.json": httpx.Response(200, headers={"content-type": "application/json"}, text="{}"),
        "reuters.com/missing": httpx.Response(404, headers={"content-type": "text/html"}, text=ARTICLE),
    })
    for path in ("/data.json", "/missing"):
        assert await fetch(f"https://reuters.com{path}", rec=rec) is None


@pytest.mark.anyio
async def test_oversize_body_returns_none():
    big = b"<p>" + b"a" * (MAX_BODY_BYTES + 1 - 3)
    assert len(big) == MAX_BODY_BYTES + 1
    chunk = b"a" * (64 * 1024)
    chunks_sent: List[int] = []

    async def endless():
        while True:
            chunks_sent.append(1)
            yield chunk

    rec = Recorder({
        # Declared length over the cap.
        "reuters.com/declared": httpx.Response(200, headers={"content-type": "text/html"}, content=big),
    })
    assert await fetch("https://reuters.com/declared", rec=rec) is None

    # Chunked, no Content-Length: the cap is enforced while streaming, reading stops past it.
    rec.routes["reuters.com/chunked"] = httpx.Response(200, headers={"content-type": "text/html"}, content=endless())
    assert await fetch("https://reuters.com/chunked", rec=rec) is None
    assert len(chunks_sent) == MAX_BODY_BYTES // len(chunk) + 1


@pytest.mark.anyio
@pytest.mark.parametrize("wire_body", [
    gzip.compress(ARTICLE.encode()) + gzip.compress(b"<p>second member</p>"),
    gzip.compress(ARTICLE.encode()) + b"trailing garbage",
], ids=["two_members", "trailing_garbage"])
async def test_multi_member_gzip_returns_none(wire_body):
    rec = Recorder({"reuters.com/gz": encoded_response("gzip", wire_body)})
    assert await fetch("https://reuters.com/gz", "reuters.com", rec=rec) is None


@pytest.mark.anyio
async def test_gzip_page_is_decoded():
    rec = Recorder({"reuters.com/gz": encoded_response("gzip", gzip.compress(ARTICLE.encode()))})
    page = await fetch("https://reuters.com/gz", "reuters.com", rec=rec)
    assert page is not None and page.text == ARTICLE_TEXT


class _SpyInflater:
    def __init__(self, inner, outputs: List[int]):
        self._inner = inner
        self._outputs = outputs

    def decompress(self, data: bytes, max_length: int = 0) -> bytes:
        out = self._inner.decompress(data, max_length)
        self._outputs.append(len(out))
        return out

    @property
    def unconsumed_tail(self) -> bytes:
        return self._inner.unconsumed_tail

    @property
    def eof(self) -> bool:
        return self._inner.eof


@pytest.mark.anyio
@pytest.mark.parametrize("coding", ["gzip", "zstd"])
async def test_compressed_bomb_returns_none(coding, monkeypatch):
    plain = b"\0" * (32 * MAX_BODY_BYTES)  # 64 MiB of zeros, a few tens of KB on the wire
    if coding == "gzip":
        bomb = gzip.compress(plain, compresslevel=9)
    else:
        zstandard = pytest.importorskip("zstandard")
        bomb = zstandard.ZstdCompressor().compress(plain)
    assert len(bomb) < MAX_BODY_BYTES

    # Our inflater: record every output size.
    outputs: List[int] = []

    class SpyZlib:
        error = zlib.error

        @staticmethod
        def decompressobj(**kwargs):
            return _SpyInflater(zlib.decompressobj(**kwargs), outputs)

    monkeypatch.setattr(page_fetch, "zlib", SpyZlib)
    # httpx's own (unbounded) decoders must never run.
    httpx_decodes: List[str] = []
    for name in ("GZipDecoder", "ZStandardDecoder", "DeflateDecoder"):
        cls = getattr(httpx._decoders, name, None)
        if cls is not None:
            orig = cls.decode
            monkeypatch.setattr(cls, "decode", lambda self, data, _o=orig, _n=name: (httpx_decodes.append(_n), _o(self, data))[1])

    rec = Recorder({"reuters.com/bomb": encoded_response(coding, bomb)})
    assert await fetch("https://reuters.com/bomb", rec=rec) is None
    assert httpx_decodes == []
    assert sum(outputs) <= MAX_BODY_BYTES + 1
    assert all(n <= MAX_BODY_BYTES + 1 for n in outputs)
    if coding == "gzip":
        assert outputs, "the gzip path must go through the bounded inflater"
    else:
        assert outputs == [], "zstd is refused before any decoding"


@pytest.mark.anyio
async def test_final_domain_must_match_expected():
    rec = Recorder({
        "vertexaisearch.cloud.google.com/grounding-api-redirect/abc": redirect("https://evil.example/fake-reuters"),
        "evil.example/fake-reuters": html_response(),
    })
    mapping = {**PUBLIC_IPS, "evil.example": "93.184.216.99"}
    assert await fetch(GOOGLE_REDIRECT, "reuters.com", rec=rec, mapping=mapping) is None
    # Control: the same chain is fetchable, so the rejection above is the domain check.
    page = await fetch(GOOGLE_REDIRECT, rec=rec, mapping=mapping)
    assert page is not None and page.domain == "evil.example"

    # A look-alike suffix is not a subdomain either.
    rec.routes["vertexaisearch.cloud.google.com/grounding-api-redirect/abc"] = redirect("https://notreuters.com/x")
    rec.routes["notreuters.com/x"] = html_response()
    mapping["notreuters.com"] = "93.184.216.98"
    assert await fetch(GOOGLE_REDIRECT, "reuters.com", rec=rec, mapping=mapping) is None

    # A single-label expected domain would match every site under that TLD: refused.
    rec.routes["evil.com/x"] = html_response()
    mapping["evil.com"] = "93.184.216.97"
    assert await fetch("https://evil.com/x", "com", rec=rec, mapping=mapping) is None
    assert await fetch("https://evil.com/x", ".com.", rec=rec, mapping=mapping) is None

    # A public suffix passes the host-suffix test but not the registrable-domain equality.
    rec.routes["www.bbc.co.uk/news"] = html_response()
    mapping["www.bbc.co.uk"] = "93.184.216.94"
    assert await fetch("https://www.bbc.co.uk/news", "co.uk", rec=rec, mapping=mapping) is None
    page = await fetch("https://www.bbc.co.uk/news", "bbc.co.uk", rec=rec, mapping=mapping)
    assert page is not None and page.domain == "bbc.co.uk"

    # Case and trailing dots are normalized on both sides.
    rec.routes["www.snopes.com./fact-check/x"] = html_response()
    rec.routes["snopes.com/y"] = html_response()
    mapping["www.snopes.com."] = "93.184.216.96"
    mapping["snopes.com"] = "93.184.216.95"
    page = await fetch("https://www.snopes.com./fact-check/x", "snopes.com", rec=rec, mapping=mapping)
    assert page is not None and page.domain == "snopes.com"
    page = await fetch("https://snopes.com/y", "www.snopes.com.", rec=rec, mapping=mapping)
    assert page is not None and page.domain == "snopes.com"
    page = await fetch(GOOGLE_REDIRECT.replace("abc", "up"), "REUTERS.COM", rec=Recorder({
        "vertexaisearch.cloud.google.com/grounding-api-redirect/up": redirect("https://WWW.Reuters.com/world/x"),
        "www.reuters.com/world/x": html_response(),
    }))
    assert page is not None and page.domain == "reuters.com"


@pytest.mark.anyio
async def test_subdomain_matches_expected():
    rec = Recorder({
        "vertexaisearch.cloud.google.com/grounding-api-redirect/abc": redirect("https://www.reuters.com/world/x"),
        "www.reuters.com/world/x": html_response(),
    })
    page = await fetch(GOOGLE_REDIRECT, "reuters.com", rec=rec)
    assert page is not None
    assert page.final_url == "https://www.reuters.com/world/x"
    assert page.domain == "reuters.com"


@pytest.mark.anyio
async def test_http_final_hop_rejected_when_domain_expected():
    rec = Recorder({
        "reuters.com/plain": html_response(),
        "reuters.com/up": redirect("http://reuters.com/plain"),
    })
    assert await fetch("http://reuters.com/plain", "reuters.com", rec=rec) is None
    assert await fetch("https://reuters.com/up", "reuters.com", rec=rec) is None
    # Without attribution, plain http is still fetchable.
    page = await fetch("http://reuters.com/plain", rec=rec)
    assert page is not None and page.final_url == "http://reuters.com/plain"


@pytest.mark.anyio
async def test_falls_back_to_next_validated_ip_on_connect_error():
    tried: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        tried.append(request.url.host)
        if request.url.host == "93.184.216.34":
            raise httpx.ConnectError("refused", request=request)
        return live(html_response())

    page = await fetch_page_text(
        "https://reuters.com/a", "reuters.com", transport=httpx.MockTransport(handler),
        resolver=make_resolver({"reuters.com": ["93.184.216.36", "93.184.216.34"]}),
    )
    assert page is not None
    assert tried == ["93.184.216.34", "93.184.216.36"]


@pytest.mark.anyio
async def test_ipv6_pinning_uses_brackets():
    rec = Recorder({"reuters.com/a": html_response(), "[2606:4700::1111]/x": html_response()})
    page = await fetch("https://reuters.com/a", "reuters.com", rec=rec, mapping={"reuters.com": "2606:2800:220:1::1"})
    assert page is not None
    req = rec.requests[-1]
    assert req.url.host == "2606:2800:220:1::1"
    assert "://[2606:2800:220:1::1]/a" in str(req.url)
    assert req.headers["host"] == "reuters.com"

    # An IPv6-literal target keeps the brackets in the Host header.
    page = await fetch("https://[2606:4700::1111]/x", rec=rec, mapping={"2606:4700::1111": "2606:4700::1111"})
    assert page is not None
    assert rec.requests[-1].headers["host"] == "[2606:4700::1111]"
    assert page.domain == "2606:4700::1111"


@pytest.mark.anyio
async def test_timeout_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "page_fetch_timeout_s", 0.05)

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1.0)
        return html_response()

    page = await fetch_page_text(
        "https://reuters.com/slow", transport=httpx.MockTransport(slow), resolver=make_resolver(PUBLIC_IPS),
    )
    assert page is None


@pytest.mark.anyio
async def test_cancellation_propagates(monkeypatch):
    monkeypatch.setattr(settings, "page_fetch_timeout_s", 10.0)
    started = asyncio.Event()

    async def hang(request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.sleep(30)
        return html_response()

    task = asyncio.create_task(fetch_page_text(
        "https://reuters.com/hang", transport=httpx.MockTransport(hang), resolver=make_resolver(PUBLIC_IPS),
    ))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
