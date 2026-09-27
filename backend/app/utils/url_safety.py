import ipaddress
import socket
from typing import Callable
from urllib.parse import urlsplit

MAX_URL_LENGTH = 2048
ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {None, 80, 443}


class UnsafeURLError(ValueError):
    """The URL must not be fetched server-side (SSRF guard, H3)."""


def _is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip.split("%", 1)[0])  # drop IPv6 zone id
    if addr.version == 6 and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr.is_global


def validate_public_url(url: str, resolver: Callable = socket.getaddrinfo) -> str:
    """
    Allow only http(s) URLs on default ports, without credentials, whose host resolves
    exclusively to public addresses. Callers that fetch must re-validate every redirect hop
    and connect to the validated address (DNS-rebinding defence). See TDD §11.
    """
    if len(url) > MAX_URL_LENGTH:
        raise UnsafeURLError("URL too long")
    parts = urlsplit(url)
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeURLError(f"scheme '{parts.scheme}' not allowed")
    if parts.username or parts.password:
        raise UnsafeURLError("credentials in URL not allowed")
    try:
        port = parts.port
    except ValueError as exc:
        raise UnsafeURLError("invalid port") from exc
    if port not in ALLOWED_PORTS:
        raise UnsafeURLError(f"port {port} not allowed")
    host = parts.hostname
    if not host:
        raise UnsafeURLError("missing host")
    try:
        infos = resolver(host, port or (443 if parts.scheme.lower() == "https" else 80), proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError) as exc:
        raise UnsafeURLError("host does not resolve") from exc
    ips = {info[4][0] for info in infos}
    if not ips or not all(_is_public(ip) for ip in ips):
        raise UnsafeURLError("host resolves to a non-public address")
    return url
