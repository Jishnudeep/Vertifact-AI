import json
from pathlib import Path
from app.scoring.engine import ScoringEngine
from app.api.schemas import VerifyResponse, RecentResultModel

fixtures_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "src" / "fixtures"
engine = ScoringEngine()

def build_fixture(
    id_str, claim, s_fact, b_ling, v_consensus, v_refute,
    agents, sources, fact_check_found, verdict_found,
    profile, processing_time_ms, cached, search_suggestions_html,
    created_at, custom_warnings=None
):
    agents_succeeded = sum(1 for a in agents if a.get("status") == "ok") if agents else 3
    total_agents = len(agents) if agents else 3
    res = engine.compute(
        s_fact=s_fact,
        b_ling=b_ling,
        v_consensus=v_consensus,
        agents_succeeded=agents_succeeded,
        total_agents=total_agents,
        fact_check_found=fact_check_found,
        verdict_found=verdict_found,
        v_refute=v_refute,
    )
    warnings = list(res.warnings)
    if custom_warnings:
        warnings.extend(custom_warnings)
    
    data = {
        "id": id_str,
        "claim": claim,
        "trust_score": {
            "c_total": res.c_total,
            "s_fact": s_fact,
            "b_ling": b_ling,
            "v_consensus": v_consensus,
            "v_refute": v_refute,
            "weights": {"w1": 0.5, "w2": 0.2, "w3": 0.3},
            "raw_c_total": res.raw_c_total,
            "cap_rule": res.cap_rule,
        },
        "confidence": {
            "level": res.confidence_level,
            "warnings": warnings,
            "verdict_found": verdict_found,
            "evidence_status": res.evidence_status,
        },
        "linguistic_profile": {
            "clickbait_syntax": profile.get("clickbait_syntax", 0.0),
            "emotional_framing": profile.get("emotional_framing", 0.0),
            "sensationalism": profile.get("sensationalism", 0.0),
            "informational_density": profile.get("informational_density", 0.5),
        },
        "sources": sources,
        "processing_time_ms": processing_time_ms,
        "cached": cached,
        "search_suggestions_html": search_suggestions_html,
        "agents": agents,
        "created_at": created_at,
    }
    # Validate with VerifyResponse
    VerifyResponse.model_validate(data)
    return data

# 1. fact_checked.json
fc_agents = [
    {"name": "factcheck", "status": "ok", "duration_ms": 780, "error_code": None},
    {"name": "crossref", "status": "ok", "duration_ms": 950, "error_code": None},
    {"name": "linguistic", "status": "ok", "duration_ms": 120, "error_code": None},
]
fc_sources = [
    {
        "url": "https://www.snopes.com/fact-check/perseverance-rover-mars/",
        "title": "Did NASA Land Perseverance Rover on Mars in 2021?",
        "publisher": "Snopes",
        "bias_label": "Center",
        "type": "fact_check",
        "stance": None,
        "rating": "True",
        "found_by": "factcheck",
    },
    {
        "url": "https://www.reuters.com/article/factcheck-nasa-perseverance/",
        "title": "Reuters Fact Check: Perseverance rover touched down on Mars",
        "publisher": "Reuters",
        "bias_label": "Center",
        "type": "corroborating",
        "stance": "supports",
        "rating": None,
        "found_by": "crossref",
    },
]
fc_data = build_fixture(
    "11111111-1111-1111-1111-111111111111",
    "NASA successfully landed the Perseverance rover on Mars in 2021.",
    0.95, 0.05, 0.90, 0.0,
    fc_agents, fc_sources, True, True,
    {"clickbait_syntax": 0.02, "emotional_framing": 0.10, "sensationalism": 0.05, "informational_density": 0.88},
    1250, False, "<div id='google-suggestions'>Related searches: Mars Perseverance landing date, NASA rover 2021</div>",
    "2026-10-15T12:00:00Z"
)
(fixtures_dir / "fact_checked.json").write_text(json.dumps(fc_data, indent=2))

# 2. corroborated.json
corr_agents = [
    {"name": "factcheck", "status": "ok", "duration_ms": 820, "error_code": None},
    {"name": "crossref", "status": "ok", "duration_ms": 1100, "error_code": None},
    {"name": "linguistic", "status": "ok", "duration_ms": 95, "error_code": None},
]
corr_sources = [
    {
        "url": "https://apnews.com/article/science-jwst-exoplanet-atmosphere",
        "title": "Webb telescope reveals water vapor on distant planet",
        "publisher": "Associated Press",
        "bias_label": "Lean Left",
        "type": "corroborating",
        "stance": "supports",
        "rating": None,
        "found_by": "crossref",
    },
    {
        "url": "https://www.bbc.com/news/science-environment-123456",
        "title": "JWST finds clear signs of water vapour in exoplanet atmosphere",
        "publisher": "BBC News",
        "bias_label": "Center",
        "type": "corroborating",
        "stance": "supports",
        "rating": None,
        "found_by": "crossref",
    },
]
corr_data = build_fixture(
    "22222222-2222-2222-2222-222222222222",
    "The James Webb Space Telescope detected atmospheric water vapour on an exoplanet.",
    0.50, 0.10, 0.95, 0.0,
    corr_agents, corr_sources, False, False,
    {"clickbait_syntax": 0.05, "emotional_framing": 0.15, "sensationalism": 0.12, "informational_density": 0.82},
    1420, False, None,
    "2026-10-15T12:05:00Z"
)
(fixtures_dir / "corroborated.json").write_text(json.dumps(corr_data, indent=2))

# 3. disputed_capped.json
disp_agents = [
    {"name": "factcheck", "status": "ok", "duration_ms": 650, "error_code": None},
    {"name": "crossref", "status": "ok", "duration_ms": 890, "error_code": None},
    {"name": "linguistic", "status": "ok", "duration_ms": 80, "error_code": None},
]
disp_sources = [
    {
        "url": "https://www.factcheck.org/2023/lemon-water-diabetes-claim/",
        "title": "Lemon water does not cure diabetes, medical experts say",
        "publisher": "FactCheck.org",
        "bias_label": "Center",
        "type": "corroborating",
        "stance": "refutes",
        "rating": None,
        "found_by": "crossref",
    },
    {
        "url": "https://www.mayoclinic.org/diseases-conditions/diabetes/expert-answers/lemon-cure",
        "title": "Diabetes Treatment Myths: Lemon Water",
        "publisher": "Mayo Clinic",
        "bias_label": "Center",
        "type": "corroborating",
        "stance": "refutes",
        "rating": None,
        "found_by": "crossref",
    },
]
disp_data = build_fixture(
    "33333333-3333-3333-3333-333333333333",
    "Consuming lemon water cures diabetes within 48 hours.",
    0.50, 0.35, 0.15, 0.80,
    disp_agents, disp_sources, False, False,
    {"clickbait_syntax": 0.70, "emotional_framing": 0.60, "sensationalism": 0.85, "informational_density": 0.25},
    1150, False, None,
    "2026-10-15T12:10:00Z"
)
(fixtures_dir / "disputed_capped.json").write_text(json.dumps(disp_data, indent=2))

# 4. unverified.json
unv_agents = [
    {"name": "factcheck", "status": "ok", "duration_ms": 600, "error_code": None},
    {"name": "crossref", "status": "ok", "duration_ms": 750, "error_code": None},
    {"name": "linguistic", "status": "ok", "duration_ms": 70, "error_code": None},
]
unv_data = build_fixture(
    "44444444-4444-4444-4444-444444444444",
    "The local city council is planning to build a new monorail station in sector 4.",
    0.60, 0.0, 0.20, 0.05,
    unv_agents, [], False, False,
    {"clickbait_syntax": 0.00, "emotional_framing": 0.05, "sensationalism": 0.10, "informational_density": 0.50},
    980, False, None,
    "2026-10-15T12:15:00Z"
)
(fixtures_dir / "unverified.json").write_text(json.dumps(unv_data, indent=2))

# 5. partial.json
part_agents = [
    {"name": "factcheck", "status": "ok", "duration_ms": 950, "error_code": None},
    {"name": "crossref", "status": "ok", "duration_ms": 1200, "error_code": None},
    {"name": "linguistic", "status": "degraded", "duration_ms": 8000, "error_code": "TIMEOUT"},
]
part_sources = [
    {
        "url": "https://www.reuters.com/business/aerospace-defense/solar-storm-flight-delays",
        "title": "Airlines report minor reroutes during solar storm",
        "publisher": "Reuters",
        "bias_label": "Center",
        "type": "corroborating",
        "stance": "supports",
        "rating": None,
        "found_by": "crossref",
    }
]
part_data = build_fixture(
    "55555555-5555-5555-5555-555555555555",
    "A new solar flare disrupted transatlantic flights yesterday.",
    0.50, 0.50, 0.60, 0.05,
    part_agents, part_sources, False, False,
    {"clickbait_syntax": 0.50, "emotional_framing": 0.50, "sensationalism": 0.50, "informational_density": 0.50},
    8200, False, None,
    "2026-10-15T12:20:00Z"
)
(fixtures_dir / "partial.json").write_text(json.dumps(part_data, indent=2))

# 6. cached.json
cached_agents = [
    {"name": "factcheck", "status": "ok", "duration_ms": None, "error_code": None},
    {"name": "crossref", "status": "ok", "duration_ms": None, "error_code": None},
    {"name": "linguistic", "status": "ok", "duration_ms": None, "error_code": None},
]
cached_sources = [
    {
        "url": "https://www.snopes.com/fact-check/great-wall-of-china-space/",
        "title": "Is the Great Wall of China Visible from Space or the Moon?",
        "publisher": "Snopes",
        "bias_label": "Center",
        "type": "fact_check",
        "stance": None,
        "rating": "False",
        "found_by": "factcheck",
    }
]
cached_data = build_fixture(
    "66666666-6666-6666-6666-666666666666",
    "The Great Wall of China is visible from the Moon with the naked eye.",
    0.05, 0.15, 0.35, 0.0,
    cached_agents, cached_sources, True, True,
    {"clickbait_syntax": 0.15, "emotional_framing": 0.10, "sensationalism": 0.20, "informational_density": 0.70},
    15, True, None,
    "2026-10-15T11:00:00Z"
)
(fixtures_dir / "cached.json").write_text(json.dumps(cached_data, indent=2))

# 7. legacy.json
leg_sources = [
    {
        "url": "https://www.example.com/legacy-news",
        "title": "Legacy Report on Space Exploration",
        "publisher": "Example News",
        "bias_label": "Center",
        "type": "corroborating",
        "stance": "supports",
        "rating": None,
        "found_by": "crossref",
    }
]
leg_data = build_fixture(
    "77777777-7777-7777-7777-777777777777",
    "Historical claim verified prior to Plan 7 architecture upgrades.",
    0.80, 0.15, 0.65, 0.0,
    [], leg_sources, False, False,
    {"clickbait_syntax": 0.05, "emotional_framing": 0.15, "sensationalism": 0.10, "informational_density": 0.75},
    1300, False, None,
    "2026-10-15T10:00:00Z",
    custom_warnings=["Legacy result: scored before stance-aware consensus; treat corroboration as unverified"]
)
(fixtures_dir / "legacy.json").write_text(json.dumps(leg_data, indent=2))

# 8. long_claim.json
long_claim_text = (
    "According to extensive archaeological documentation and historical records spanning several centuries across the Mediterranean basin, ancient civilizations developed intricate methods of civil engineering and water management that allowed cities in arid regions to sustain populations numbering in the tens of thousands. These systems included subterranean aqueducts known as qanats, complex cistern reservoirs carved directly into bedrock, and terraced agricultural slopes designed to capture seasonal runoff. Scholars from various academic institutions have cross-examined these artifacts using carbon dating, stratigraphic analysis, and comparative textual analysis from contemporaneous administrative records. However, in recent years, alternative interpretations circulating across digital platforms and fringe publications have asserted that these architectural marvels could not possibly have been engineered using the copper and bronze metallurgy available during the third millennium before the common era. Proponents of these fringe narratives claim that anomalous stone carvings depict high-energy electrical infrastructure and precision laser apparatus, suggesting the involvement of extraterrestrial visitors or technologically advanced lost civilizations that disappeared during a sudden cataclysmic event at the conclusion of the Younger Dryas climatic oscillation. Mainstream geologists, Egyptologists, and material scientists have repeatedly analyzed the physical quarry markings and tool marks on surviving granite monuments, demonstrating conclusively that diorite pounders, copper saws with quartz abrasive slurries, and wooden expansion wedges were fully capable of producing the observed structural tolerances. Despite extensive peer-reviewed refutations published in respected journals, the speculative theories continue to achieve substantial viral reach across online discussion forums, documentary channels, and short-form video streaming services worldwide."
)
long_agents = [
    {"name": "factcheck", "status": "ok", "duration_ms": 900, "error_code": None},
    {"name": "crossref", "status": "ok", "duration_ms": 1100, "error_code": None},
    {"name": "linguistic", "status": "ok", "duration_ms": 150, "error_code": None},
]
long_data = build_fixture(
    "88888888-8888-8888-8888-888888888888",
    long_claim_text,
    0.50, 0.30, 0.10, 0.75,
    long_agents, [], False, False,
    {"clickbait_syntax": 0.10, "emotional_framing": 0.20, "sensationalism": 0.35, "informational_density": 0.85},
    1400, False, None,
    "2026-10-15T12:25:00Z"
)
(fixtures_dir / "long_claim.json").write_text(json.dumps(long_data, indent=2))

# 9. Update recent.json to reflect the updated fixtures
recent_data = [
    {
        "id": "11111111-1111-1111-1111-111111111111",
        "claim": "NASA successfully landed the Perseverance rover on Mars in 2021.",
        "c_total": fc_data["trust_score"]["c_total"],
        "evidence_status": fc_data["confidence"]["evidence_status"],
        "confidence_level": fc_data["confidence"]["level"],
        "created_at": "2026-10-15T12:00:00Z",
    },
    {
        "id": "22222222-2222-2222-2222-222222222222",
        "claim": "The James Webb Space Telescope detected atmospheric water vapour on an exoplanet.",
        "c_total": corr_data["trust_score"]["c_total"],
        "evidence_status": corr_data["confidence"]["evidence_status"],
        "confidence_level": corr_data["confidence"]["level"],
        "created_at": "2026-10-15T12:05:00Z",
    },
    {
        "id": "33333333-3333-3333-3333-333333333333",
        "claim": "Consuming lemon water cures diabetes within 48 hours.",
        "c_total": disp_data["trust_score"]["c_total"],
        "evidence_status": disp_data["confidence"]["evidence_status"],
        "confidence_level": disp_data["confidence"]["level"],
        "created_at": "2026-10-15T12:10:00Z",
    },
    {
        "id": "44444444-4444-4444-4444-444444444444",
        "claim": "The local city council is planning to build a new monorail station in sector 4.",
        "c_total": unv_data["trust_score"]["c_total"],
        "evidence_status": unv_data["confidence"]["evidence_status"],
        "confidence_level": unv_data["confidence"]["level"],
        "created_at": "2026-10-15T12:15:00Z",
    },
]
for item in recent_data:
    RecentResultModel.model_validate(item)
(fixtures_dir / "recent.json").write_text(json.dumps(recent_data, indent=2))

print("All fixtures successfully generated!")
