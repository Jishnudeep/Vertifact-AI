import pytest
from starlette.requests import Request

from app.api.client_ip import client_ip
from app.config import settings


def _req(xff=None, peer="10.0.0.1"):
    headers = [(b"x-forwarded-for", xff.encode())] if xff else []
    return Request({"type": "http", "headers": headers, "client": (peer, 1234)})


def test_no_trusted_proxy_ignores_forwarded_header(monkeypatch):
    monkeypatch.setattr(settings, "trusted_proxy_hops", 0)
    assert client_ip(_req("1.1.1.1")) == "10.0.0.1"


def test_one_hop_uses_rightmost_entry(monkeypatch):
    monkeypatch.setattr(settings, "trusted_proxy_hops", 1)
    assert client_ip(_req("203.0.113.9")) == "203.0.113.9"


def test_spoofed_left_entries_are_ignored(monkeypatch):
    monkeypatch.setattr(settings, "trusted_proxy_hops", 1)
    # client sent "X-Forwarded-For: 6.6.6.6"; the proxy appended the real peer
    assert client_ip(_req("6.6.6.6, 203.0.113.9")) == "203.0.113.9"


def test_short_header_falls_back_to_peer(monkeypatch):
    monkeypatch.setattr(settings, "trusted_proxy_hops", 2)
    assert client_ip(_req("203.0.113.9")) == "10.0.0.1"
