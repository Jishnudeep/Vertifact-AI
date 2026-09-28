import ipaddress
import socket
from typing import Callable, FrozenSet, Tuple
from urllib.parse import urlsplit

MAX_URL_LENGTH = 2048
ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {None, 80, 443}
# RFC 6052 well-known NAT64 prefix: the last 32 bits are an IPv4 address.
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")
# IPv6 blocks that embed or tunnel IPv4, or are deprecated, rejected outright. Listed
# explicitly because `is_global` for some of them changed across 3.12 patch releases.
_BLOCKED_V6 = tuple(ipaddress.ip_network(n) for n in (
    "::/96",              # IPv4-compatible (deprecated, RFC 4291), incl. ::1 and ::
    "::ffff:0:0:0/96",    # SIIT IPv4-translated (RFC 2765)
    "64:ff9b:1::/48",     # local-use NAT64 (RFC 8215)
    "2001::/32",          # Teredo
    "2002::/16",          # 6to4
    "fec0::/10",          # site-local (deprecated)
))


class UnsafeURLError(ValueError):
    """The URL must not be fetched server-side (SSRF guard, H3)."""


def _is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip.split("%", 1)[0])  # drop IPv6 zone id
    if addr.version == 6:
        if any(addr in net for net in _BLOCKED_V6):
            return False
        if addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
        elif addr in _NAT64_PREFIX:
            addr = ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)
    # is_global alone treats multicast (224.0.0.0/4, ff00::/8) as public.
    return addr.is_global and not addr.is_multicast


def resolve_public_url(url: str, resolver: Callable = socket.getaddrinfo) -> Tuple[str, FrozenSet[str]]:
    """
    Allow only http(s) URLs on default ports, without credentials, whose host resolves
    exclusively to public addresses. Returns the URL and the validated IPs.

    Callers that fetch must connect to one of the returned IPs (never re-resolve the host:
    DNS rebinding) and re-validate every redirect hop. getaddrinfo blocks, so async callers
    must run this via asyncio.to_thread. See TDD §11.
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
    ips = frozenset(info[4][0] for info in infos)
    if not ips or not all(_is_public(ip) for ip in ips):
        raise UnsafeURLError("host resolves to a non-public address")
    return url, ips


def validate_public_url(url: str, resolver: Callable = socket.getaddrinfo) -> str:
    """resolve_public_url without the addresses. Prefer resolve_public_url when fetching."""
    return resolve_public_url(url, resolver=resolver)[0]
