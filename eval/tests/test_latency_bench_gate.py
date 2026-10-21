"""latency_bench spends Gemini quota (about 8 requests per claim): it must never run without --live."""
import pytest

from eval import latency_bench


def test_latency_bench_refuses_without_live_flag():
    with pytest.raises(SystemExit) as exc:
        latency_bench._parse_args([])
    assert exc.value.code != 0


def test_latency_bench_accepts_live_flag():
    args = latency_bench._parse_args(["--live", "--n", "4"])
    assert args.live is True and args.n == 4


def test_latency_bench_prints_estimated_requests(capsys):
    latency_bench.print_estimate(4)
    out = capsys.readouterr().out
    assert "32" in out and "requests" in out
