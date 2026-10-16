import json
import re
from pathlib import Path
from typing import Any, Literal, get_origin, get_args
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
from app.utils.errors import degradation_message

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "src" / "fixtures"
COPY_JS = Path(__file__).resolve().parent.parent.parent / "frontend" / "src" / "lib" / "copy.js"

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


def _load(filename: str) -> dict:
    with open(FIXTURES_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_legacy(data: dict) -> bool:
    return any(
        isinstance(w, str) and w.startswith(LEGACY_WARNING_PREFIX)
        for w in data.get("confidence", {}).get("warnings", [])
    )


LEGACY_WARNING_PREFIX = "Legacy result:"
AGENT_LABELS = {"factcheck": "Fact-check", "crossref": "Cross-reference", "linguistic": "Linguistic"}


def test_at_least_one_fixture_is_legacy():
    assert any(_is_legacy(_load(name)) for name in VERIFY_FIXTURE_NAMES)


@pytest.mark.parametrize("filename", VERIFY_FIXTURE_NAMES)
def test_legacy_fixture_matches_stored_formatter_shape(filename: str):
    """A legacy row (evidence_status NULL, no agent_outputs.scoring) can only reach the client via
    format_stored_result, which yields unverified / null raw_c_total / null cap_rule / no agents."""
    data = _load(filename)
    if not _is_legacy(data):
        pytest.skip("not a legacy fixture")
    assert data["confidence"]["evidence_status"] == "unverified"
    assert data["trust_score"]["raw_c_total"] is None
    assert data["trust_score"]["cap_rule"] is None
    assert data["agents"] == []


@pytest.mark.parametrize("filename", VERIFY_FIXTURE_NAMES)
def test_degraded_agents_carry_degradation_warning(filename: str):
    """score_node always appends degradation_message(label, code) for each degraded agent."""
    data = _load(filename)
    warnings = data["confidence"]["warnings"]
    for agent in data.get("agents", []):
        if agent["status"] != "degraded":
            continue
        expected = degradation_message(AGENT_LABELS[agent["name"]], agent.get("error_code"))
        assert expected in warnings, f"{filename}: missing {expected!r} in {warnings}"


def _literal_values(annotation: Any) -> set:
    """Flatten Optional[Literal[...]] into the set of literal values."""
    values: set = set()
    for arg in get_args(annotation):
        if get_origin(arg) is Literal:
            values.update(get_args(arg))
        elif arg is not type(None):
            values.update(_literal_values(arg))
    if get_origin(annotation) is Literal:
        values.update(get_args(annotation))
    return values


def _js_object_keys(source: str, const_name: str) -> set:
    match = re.search(rf"export\s+const\s+{const_name}\s*=\s*\{{(.*?)\}};", source, re.DOTALL)
    assert match, f"{const_name} not found in copy.js"
    return set(re.findall(r"^\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*:", match.group(1), re.MULTILINE))


def test_cap_rule_literal_matches_frontend_cap_copy():
    backend_rules = _literal_values(TrustScoreModel.model_fields["cap_rule"].annotation)
    assert backend_rules, "cap_rule Literal args could not be parsed"
    source = COPY_JS.read_text(encoding="utf-8")
    assert backend_rules == _js_object_keys(source, "CAP_COPY")


@pytest.mark.parametrize("filename", VERIFY_FIXTURE_NAMES)
def test_verify_fixture_producible_state(filename: str):
    data = _load(filename)
    if _is_legacy(data):
        pytest.skip("legacy rows were scored by the pre-stance engine; shape is checked separately")

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
    # Independent of the fixture's own evidence_status: a matched review surfaces as a factcheck source.
    fact_check_found = any(s.get("found_by") == "factcheck" for s in data.get("sources", []))

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
