from pathlib import Path
import pytest


def test_live_api_tests_are_skipped_by_default(pytester):
    conftest_content = (Path(__file__).parents[1] / "conftest.py").read_text(encoding="utf-8")
    pytester.makeconftest(conftest_content)
    pytester.makepyfile("""
        import pytest

        @pytest.mark.live_api
        def test_calls_api():
            raise AssertionError("must not run by default")
    """)
    result = pytester.runpytest()
    result.assert_outcomes(skipped=1)


def test_live_api_tests_run_with_flag(pytester):
    conftest_content = (Path(__file__).parents[1] / "conftest.py").read_text(encoding="utf-8")
    pytester.makeconftest(conftest_content)
    pytester.makepyfile("""
        import pytest

        @pytest.mark.live_api
        def test_calls_api():
            assert True
    """)
    result = pytester.runpytest("--run-live-api")
    result.assert_outcomes(passed=1)


def test_live_api_tests_run_with_env_var(pytester, monkeypatch):
    conftest_content = (Path(__file__).parents[1] / "conftest.py").read_text(encoding="utf-8")
    pytester.makeconftest(conftest_content)
    pytester.makepyfile("""
        import pytest

        @pytest.mark.live_api
        def test_calls_api():
            assert True
    """)
    monkeypatch.setenv("VERIFACT_RUN_LIVE_API", "1")
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
