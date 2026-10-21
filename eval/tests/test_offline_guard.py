"""The offline eval suite must not be able to reach the network (Task C, N13)."""
import socket

import pytest

from eval.conftest import NetworkBlocked


def test_a_non_loopback_connection_is_blocked_and_recorded(network_guard):
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("203.0.113.7", 443), timeout=1)
    with pytest.raises(NetworkBlocked):
        socket.getaddrinfo("example.supabase.co", 443)
    assert network_guard.attempts == ["203.0.113.7", "example.supabase.co"]
    network_guard.attempts.clear()      # provoked on purpose; do not fail the teardown check


def test_loopback_is_allowed(network_guard):
    network_guard.check("127.0.0.1")
    network_guard.check("localhost")
    assert network_guard.attempts == []


def test_tests_never_see_a_real_supabase_project():
    import os
    assert os.environ["SUPABASE_URL"] == "https://example.supabase.co"
