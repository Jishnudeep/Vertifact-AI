import json
from pathlib import Path
from typing import Any, get_origin, get_args
import inspect
import pytest
from pydantic import BaseModel

from app.api.schemas import (
    VerifyResponse,
    TrustScoreModel,
    ConfidenceModel,
    LinguisticProfileModel,
    SourceCitationModel,
    AgentStatusModel,
    RecentResultModel,
    ErrorResponse,
)
from app.scoring.engine import ScoringEngine

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "src" / "fixtures"

VERIFY_FIXTURE_NAMES = [
    "fact_checked.json",
    "corroborated.json",
    "disputed_capped.json",
    "unverified.json",
    "partial.json",
    "cached.json",
    "legacy.json",
    "long_claim.json",
]


def assert_no_extra_keys(data: Any, model_cls: type[BaseModel]) -> None:
    if not isinstance(data, dict):
        return
    allowed_keys = set(model_cls.model_fields.keys())
    extra_keys = set(data.keys()) - allowed_keys
    assert not extra_keys, f"Extra keys found in {model_cls.__name__}: {extra_keys}"

    for field_name, field_info in model_cls.model_fields.items():
        if field_name not in data or data[field_name] is None:
            continue
        val = data[field_name]
        annotation = field_info.annotation

        target_models = []
        origin = get_origin(annotation)
        args = get_args(annotation)
        if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
            target_models.append((annotation, False))
        elif origin is list:
            for arg in args:
                if inspect.isclass(arg) and issubclass(arg, BaseModel):
                    target_models.append((arg, True))
        elif origin is not None:
            # Handle Union / Optional
            for arg in args:
                if inspect.isclass(arg) and issubclass(arg, BaseModel):
                    target_models.append((arg, False))

        for target_model, is_list in target_models:
            if is_list and isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        assert_no_extra_keys(item, target_model)
            elif not is_list and isinstance(val, dict):
                assert_no_extra_keys(val, target_model)


@pytest.mark.parametrize("filename", VERIFY_FIXTURE_NAMES)
def test_verify_fixture_schema_and_no_extra_keys(filename: str):
    path = FIXTURES_DIR / filename
    assert path.exists(), f"Fixture {filename} does not exist"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Schema validation
    validated = VerifyResponse.model_validate(data)
    assert validated is not None

    # 2. No extra keys recursively
    assert_no_extra_keys(data, VerifyResponse)


@pytest.mark.parametrize("filename", VERIFY_FIXTURE_NAMES)
def test_verify_fixture_producible_state(filename: str):
    path = FIXTURES_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ts = data.get("trust_score", {})
    conf = data.get("confidence", {})

    s_fact = ts.get("s_fact", 0.5)
    b_ling = ts.get("b_ling", 0.5)
    v_consensus = ts.get("v_consensus", 0.0)
    v_refute = ts.get("v_refute", 0.0)

    agents = data.get("agents", [])
    if agents:
        agents_succeeded = sum(1 for a in agents if a.get("status") == "ok")
        total_agents = len(agents)
    else:
        agents_succeeded = 3
        total_agents = 3

    verdict_found = conf.get("verdict_found", False)
    # fact_check_found is true if verdict_found is true, or if any source is fact_check, or if evidence_status is fact_checked
    fact_check_found = verdict_found or (conf.get("evidence_status") == "fact_checked") or any(
        s.get("type") == "fact_check" for s in data.get("sources", [])
    )

    weights = ts.get("weights") or {"w1": 0.5, "w2": 0.2, "w3": 0.3}
    engine = ScoringEngine(w1=weights.get("w1", 0.5), w2=weights.get("w2", 0.2), w3=weights.get("w3", 0.3))
    expected = engine.compute(
        s_fact=s_fact,
        b_ling=b_ling,
        v_consensus=v_consensus,
        agents_succeeded=agents_succeeded,
        total_agents=total_agents,
        fact_check_found=fact_check_found,
        verdict_found=verdict_found,
        v_refute=v_refute,
    )

    assert ts.get("c_total") == pytest.approx(expected.c_total, abs=1e-3), (
        f"{filename}: c_total mismatch fixture={ts.get('c_total')} expected={expected.c_total}"
    )
    assert ts.get("raw_c_total") == pytest.approx(expected.raw_c_total, abs=1e-3), (
        f"{filename}: raw_c_total mismatch fixture={ts.get('raw_c_total')} expected={expected.raw_c_total}"
    )
    assert ts.get("cap_rule") == expected.cap_rule, (
        f"{filename}: cap_rule mismatch fixture={ts.get('cap_rule')} expected={expected.cap_rule}"
    )
    assert conf.get("level") == expected.confidence_level, (
        f"{filename}: confidence.level mismatch fixture={conf.get('level')} expected={expected.confidence_level}"
    )
    assert conf.get("evidence_status") == expected.evidence_status, (
        f"{filename}: confidence.evidence_status mismatch fixture={conf.get('evidence_status')} expected={expected.evidence_status}"
    )


def test_recent_fixture_schema():
    path = FIXTURES_DIR / "recent.json"
    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    for item in data:
        validated = RecentResultModel.model_validate(item)
        assert validated is not None
        assert_no_extra_keys(item, RecentResultModel)


def test_errors_fixture_schema():
    path = FIXTURES_DIR / "errors.json"
    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict)
    for code, item in data.items():
        validated = ErrorResponse.model_validate(item)
        assert validated is not None
        assert_no_extra_keys(item, ErrorResponse)
