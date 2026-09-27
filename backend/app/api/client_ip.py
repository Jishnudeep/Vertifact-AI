from starlette.requests import Request
from slowapi.util import get_remote_address

from app.config import settings


def client_ip(request: Request) -> str:
    """
    Rate-limit key (H3). Behind N trusted proxies the real client is the N-th entry from the
    RIGHT of X-Forwarded-For. Entries further left are client-supplied and spoofable.
    With no trusted proxy, use the socket peer.
    """
    hops = settings.trusted_proxy_hops
    if hops > 0:
        parts = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(parts) >= hops:
            return parts[-hops]
    return get_remote_address(request)
