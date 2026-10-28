import httpx
import pytest

from app.agents.scraper_agent import ScrapeResult, scrape_url, strip_site_suffix
from tests.test_page_fetch import PUBLIC_IPS, Recorder, html_response, make_resolver, redirect

LONG = ("Scientists say the deposits are stable over geological time. " * 5).strip()
PRIVATE_IPS = {**PUBLIC_IPS, "internal.example": "10.0.0.5"}


def page(head: str = "", body: str = "") -> str:
    return f"<html><head>{head}</head><body>{body}</body></html>"


async def scrape(url: str, rec: Recorder, mapping=None) -> ScrapeResult:
    return await scrape_url(url, transport=httpx.MockTransport(rec), resolver=make_resolver(mapping or PUBLIC_IPS))


@pytest.mark.parametrize("title,expected", [
    ("Mars water found by NASA | Reuters", "Mars water found by NASA"),
    ("Mars water found by NASA - The New York Times", "Mars water found by NASA"),
    ("Mars water found by NASA — BBC News", "Mars water found by NASA"),
    ("Mars water found", "Mars water found"),
    ("Mars - Reuters", "Mars - Reuters"),  # stripping would leave a 1-word headline
    ("Water - found on Mars by the rover, scientists say today", "Water - found on Mars by the rover, scientists say today"),
    ("Mars water found by NASA", "Mars water found by NASA"),
])
def test_strip_site_suffix(title, expected):
    assert strip_site_suffix(title) == expected


@pytest.mark.anyio
async def test_extracts_headline_and_article_body():
    html = page(
        '<meta property="og:title" content="NASA confirms ice on Moon | Reuters">'
        "<title>ignored</title>",
        f"<p>Header blurb</p><article><p>{LONG}</p><p>More.</p></article><aside><p>Ad</p></aside>",
    )
    res = await scrape("https://reuters.com/a", Recorder({"reuters.com/a": html_response(html)}))
    assert res.error is None and not res.degraded
    assert res.headline == "NASA confirms ice on Moon"
    assert res.body == f"{LONG} More."
    assert res.final_url == "https://reuters.com/a"


@pytest.mark.anyio
async def test_headline_falls_back_to_title_then_h1():
    rec = Recorder({
        "reuters.com/t": html_response(page("<title>Title headline here today</title>", f"<p>{LONG}</p>")),
        "reuters.com/h": html_response(page("", f"<h1> H1  headline </h1><p>{LONG}</p>")),
    })
    assert (await scrape("https://reuters.com/t", rec)).headline == "Title headline here today"
    assert (await scrape("https://reuters.com/h", rec)).headline == "H1 headline"


@pytest.mark.anyio
async def test_body_capped_at_1500_words():
    words = " ".join(f"w{i}" for i in range(3000))
    rec = Recorder({"reuters.com/a": html_response(page("<title>A long article about words</title>", f"<p>{words}</p>"))})
    res = await scrape("https://reuters.com/a", rec)
    assert len(res.body.split()) == 1500


@pytest.mark.anyio
async def test_thin_page_degrades_to_og_description():
    html = page(
        '<meta property="og:title" content="App shell headline">'
        '<meta property="og:description" content="Summary of the story.">',
        "<div id=root></div>",
    )
    res = await scrape("https://reuters.com/a", Recorder({"reuters.com/a": html_response(html)}))
    assert res.degraded and res.error is None
    assert (res.headline, res.body) == ("App shell headline", "Summary of the story.")


@pytest.mark.anyio
async def test_403_with_html_uses_meta_fallback():
    html = page('<meta property="og:title" content="Walled story headline"><meta name="description" content="Teaser.">')
    resp = httpx.Response(403, headers={"content-type": "text/html"}, text=html)
    res = await scrape("https://reuters.com/a", Recorder({"reuters.com/a": resp}))
    assert res.degraded and res.error is None
    assert (res.headline, res.body) == ("Walled story headline", "Teaser.")


@pytest.mark.anyio
async def test_403_bot_wall_title_is_not_a_headline():
    resp = httpx.Response(403, headers={"content-type": "text/html"},
                          text=page("<title>Just a moment...</title>", "<h1>Checking your browser</h1>"))
    res = await scrape("https://reuters.com/a", Recorder({"reuters.com/a": resp}))
    assert res.headline == "" and res.error == "That page blocked access and has no headline we can use."


@pytest.mark.anyio
async def test_403_non_html_rejected():
    resp = httpx.Response(403, headers={"content-type": "application/json"}, text="{}")
    res = await scrape("https://reuters.com/a", Recorder({"reuters.com/a": resp}))
    assert res.error == "Could not read that page."


@pytest.mark.anyio
async def test_redirect_chain_ending_on_403_with_og_title():
    html = page('<meta property="og:title" content="Walled story headline">')
    rec = Recorder({
        "reuters.com/a": redirect("https://www.reuters.com/b"),
        "www.reuters.com/b": httpx.Response(403, headers={"content-type": "text/html"}, text=html),
    })
    res = await scrape("https://reuters.com/a", rec)
    assert res.headline == "Walled story headline" and res.degraded
    assert res.final_url == "https://www.reuters.com/b"


@pytest.mark.anyio
async def test_parser_failure_never_raises(monkeypatch):
    from app.agents import scraper_agent
    monkeypatch.setattr(scraper_agent, "parse_article", lambda *a, **k: 1 / 0)
    rec = Recorder({"reuters.com/a": html_response(page("<title>Fine headline for this</title>", f"<p>{LONG}</p>"))})
    assert (await scrape("https://reuters.com/a", rec)).error == "Could not read that page."


@pytest.mark.anyio
async def test_other_error_statuses_fail():
    rec = Recorder({"reuters.com/a": httpx.Response(500, headers={"content-type": "text/html"}, text="x")})
    res = await scrape("https://reuters.com/a", rec)
    assert res.error and res.degraded and res.headline == ""


@pytest.mark.anyio
async def test_no_headline_is_an_error():
    rec = Recorder({"reuters.com/a": html_response(page("", f"<p>{LONG}</p>"))})
    res = await scrape("https://reuters.com/a", rec)
    assert res.error == "Could not find a headline on that page."


@pytest.mark.anyio
async def test_private_host_never_requested():
    rec = Recorder({})
    res = await scrape("https://internal.example/a", rec, mapping=PRIVATE_IPS)
    assert res.error and "public" in res.error
    assert rec.requests == []


@pytest.mark.anyio
async def test_redirect_to_private_host_is_blocked():
    rec = Recorder({"reuters.com/a": redirect("https://internal.example/secret")})
    res = await scrape("https://reuters.com/a", rec, mapping=PRIVATE_IPS)
    assert res.error
    assert len(rec.requests) == 1


@pytest.mark.anyio
async def test_non_html_rejected_and_error_text_is_fixed():
    rec = Recorder({"reuters.com/a": httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF")})
    res = await scrape("https://reuters.com/a", rec)
    assert res.error == "Could not read that page."
