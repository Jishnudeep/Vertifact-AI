import asyncio
import socket
from typing import Callable, Dict, List

import httpx
import pytest

from app.config import settings
from app.services.page_fetch import MAX_BODY_BYTES, PageText, fetch_page_text

PUBLIC_IPS: Dict[str, str] = {
    "reuters.com": "93.184.216.34",
    "www.reuters.com": "93.184.216.35",
    "vertexaisearch.cloud.google.com": "142.250.80.1",
    "hop.example": "93.184.216.40",
}

ARTICLE = (
    "<html><head><title>  Fact check:   moon ice  </title>"
    "<style>p{color:red}</style><script>var p = '<p>nope</p>';</script></head>"
    "<body><nav><p>Home | World</p></nav>"
    "<p>NASA  confirmed\n water ice</p><p>on the Moon.</p>"
    "<footer><p>Copyright</p></footer></body></html>"
)


def make_resolver(mapping: Dict[str, str]) -> Callable:
    def _resolve(host, port, proto=0, **kwargs):
        if host not in mapping:
            raise socket.gaierror("unknown host")
        ip = mapping[host]
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, proto, "", (ip, port))]
    return _resolve


def html_response(body: str = ARTICLE, **kwargs) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=body, **kwargs)


def redirect(location: str) -> httpx.Response:
    return httpx.Response(302, headers={"location": location})


class Recorder:
    """MockTransport handler routing on the Host header (requests go to the pinned IP)."""

    def __init__(self, routes: Dict[str, httpx.Response]):
        self.routes = routes
        self.requests: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = request.headers["host"] + request.url.path
        return self.routes.get(key, httpx.Response(404))


@pytest.mark.anyio
async def test_extracts_title_and_text():
    rec = Recorder({"reuters.com/article": html_response()})
    page = await fetch_page_text(
        "https://reuters.com/article", "reuters.com",
        transport=httpx.MockTransport(rec), resolver=make_resolver(PUBLIC_IPS),
    )
    assert isinstance(page, PageText)
    assert page.final_url == "https://reuters.com/article"
    assert page.domain == "reuters.com"
    assert page.title == "Fact check: moon ice"
    assert page.text == "NASA confirmed water ice on the Moon."
    # Pinned: connect to the validated IP, keep the hostname in Host and SNI.
    req = rec.requests[0]
    assert req.url.host == "93.184.216.34"
    assert req.headers["host"] == "reuters.com"
    assert req.extensions["sni_hostname"] == "reuters.com"

    # og:title is the fallback when there is no <title>.
    rec.routes["reuters.com/og"] = html_response(
        '<html><head><meta property="og:title" content=" OG headline "></head><body><p>x</p></body></html>'
    )
    page = await fetch_page_text(
        "https://reuters.com/og", transport=httpx.MockTransport(rec), resolver=make_resolver(PUBLIC_IPS),
    )
    assert page is not None and page.title == "OG headline"


@pytest.mark.anyio
async def test_private_ip_is_refused():
    rec = Recorder({"intranet.example/": html_response()})
    page = await fetch_page_text(
        "https://intranet.example/", transport=httpx.MockTransport(rec),
        resolver=make_resolver({"intranet.example": "10.0.0.5"}),
    )
    assert page is None
    assert rec.requests == []


@pytest.mark.anyio
async def test_redirect_to_private_host_is_refused():
    rec = Recorder({
        "reuters.com/a": redirect("http://metadata.internal/latest"),
        "metadata.internal/latest": html_response(),
    })
    mapping = {**PUBLIC_IPS, "metadata.internal": "169.254.169.254"}
    page = await fetch_page_text(
        "https://reuters.com/a", transport=httpx.MockTransport(rec), resolver=make_resolver(mapping),
    )
    assert page is None
    assert [r.headers["host"] for r in rec.requests] == ["reuters.com"]


@pytest.mark.anyio
async def test_more_than_three_redirects_returns_none():
    routes = {
        "reuters.com/0": redirect("/1"),       # relative Location resolves against the current URL
        "reuters.com/1": redirect("https://hop.example/2"),
        "hop.example/2": redirect("https://reuters.com/3"),
        "reuters.com/3": redirect("https://reuters.com/4"),
        "reuters.com/4": html_response(),
    }
    transport = httpx.MockTransport(Recorder(routes))
    resolver = make_resolver(PUBLIC_IPS)

    # Three redirects (from /1) are allowed...
    page = await fetch_page_text("https://reuters.com/1", transport=transport, resolver=resolver)
    assert page is not None and page.final_url == "https://reuters.com/4"
    # ...a fourth is not.
    assert await fetch_page_text("https://reuters.com/0", transport=transport, resolver=resolver) is None


@pytest.mark.anyio
async def test_non_html_returns_none():
    rec = Recorder({
        "reuters.com/data.json": httpx.Response(200, headers={"content-type": "application/json"}, text="{}"),
        "reuters.com/missing": httpx.Response(404, headers={"content-type": "text/html"}, text=ARTICLE),
    })
    for path in ("/data.json", "/missing"):
        page = await fetch_page_text(
            f"https://reuters.com{path}", transport=httpx.MockTransport(rec), resolver=make_resolver(PUBLIC_IPS),
        )
        assert page is None


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
    resolver = make_resolver(PUBLIC_IPS)
    assert await fetch_page_text(
        "https://reuters.com/declared", transport=httpx.MockTransport(rec), resolver=resolver,
    ) is None

    # Chunked, no Content-Length: the cap is enforced while streaming, reading stops past it.
    rec.routes["reuters.com/chunked"] = httpx.Response(200, headers={"content-type": "text/html"}, content=endless())
    assert await fetch_page_text(
        "https://reuters.com/chunked", transport=httpx.MockTransport(rec), resolver=resolver,
    ) is None
    assert len(chunks_sent) == MAX_BODY_BYTES // len(chunk) + 1


@pytest.mark.anyio
async def test_final_domain_must_match_expected():
    google_redirect = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc"
    rec = Recorder({
        "vertexaisearch.cloud.google.com/grounding-api-redirect/abc": redirect("https://evil.example/fake-reuters"),
        "evil.example/fake-reuters": html_response(),
    })
    mapping = {**PUBLIC_IPS, "evil.example": "93.184.216.99"}
    page = await fetch_page_text(
        google_redirect, "reuters.com", transport=httpx.MockTransport(rec), resolver=make_resolver(mapping),
    )
    assert page is None
    # Control: the same chain is fetchable, so the rejection above is the domain check.
    page = await fetch_page_text(google_redirect, transport=httpx.MockTransport(rec), resolver=make_resolver(mapping))
    assert page is not None and page.domain == "evil.example"
    # A look-alike suffix is not a subdomain either.
    rec.routes["vertexaisearch.cloud.google.com/grounding-api-redirect/abc"] = redirect("https://notreuters.com/x")
    rec.routes["notreuters.com/x"] = html_response()
    mapping["notreuters.com"] = "93.184.216.98"
    page = await fetch_page_text(
        google_redirect, "reuters.com", transport=httpx.MockTransport(rec), resolver=make_resolver(mapping),
    )
    assert page is None


@pytest.mark.anyio
async def test_subdomain_matches_expected():
    rec = Recorder({
        "vertexaisearch.cloud.google.com/grounding-api-redirect/abc": redirect("https://www.reuters.com/world/x"),
        "www.reuters.com/world/x": html_response(),
    })
    page = await fetch_page_text(
        "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc", "reuters.com",
        transport=httpx.MockTransport(rec), resolver=make_resolver(PUBLIC_IPS),
    )
    assert page is not None
    assert page.final_url == "https://www.reuters.com/world/x"
    assert page.domain == "reuters.com"


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
