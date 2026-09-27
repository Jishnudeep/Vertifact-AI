import socket

import pytest

from app.utils.url_safety import UnsafeURLError, validate_public_url


def resolver_for(*ips):
    def _resolve(host, port, proto=0, **kwargs):
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, proto, "", (ip, port))
                for ip in ips]
    return _resolve


def test_public_https_url_is_allowed():
    url = "https://example.com/article"
    assert validate_public_url(url, resolver=resolver_for("93.184.216.34")) == url


@pytest.mark.parametrize("ip", [
    "127.0.0.1", "10.0.0.5", "192.168.1.10", "172.16.0.1", "169.254.169.254",
    "::1", "fe80::1", "fc00::1", "::ffff:127.0.0.1", "0.0.0.0",
])
def test_non_public_addresses_are_rejected(ip):
    with pytest.raises(UnsafeURLError):
        validate_public_url("https://evil.test/", resolver=resolver_for(ip))


def test_any_private_address_among_many_rejects():
    with pytest.raises(UnsafeURLError):
        validate_public_url("https://mixed.test/", resolver=resolver_for("93.184.216.34", "10.0.0.1"))


@pytest.mark.parametrize("url", [
    "ftp://example.com/x", "file:///etc/passwd", "javascript:alert(1)",
    "http://user:pass@example.com/", "http://example.com:8080/", "https:///nohost",
    "http://example.com:99999/", "https://example.com/" + "a" * 2100,
])
def test_bad_shapes_are_rejected(url):
    with pytest.raises(UnsafeURLError):
        validate_public_url(url, resolver=resolver_for("93.184.216.34"))


def test_unresolvable_host_is_rejected():
    def boom(*a, **k):
        raise socket.gaierror("nope")
    with pytest.raises(UnsafeURLError):
        validate_public_url("https://no-such-host.test/", resolver=boom)
