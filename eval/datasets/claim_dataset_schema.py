from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class EvalClaim(BaseModel):
    id: str
    claim_text: str
    ground_truth_score: float = Field(ge=0.0, le=1.0)
    category: str
    claim_type: Literal["verifiable", "opinion", "question", "adversarial"] = "verifiable"
    expected_confidence: Literal["high", "medium", "low", "insufficient"] = "high"
    expected_verdict: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    # C3: which benchmark the claim came from; metrics are reported per source_dataset.
    source_dataset: Literal["curated", "liar", "fever"] = "curated"
    # C3 leave-source-out: the site that produced the label (LIAR -> politifact.com). Evidence
    # from this domain is filtered out while the claim is evaluated, so the system cannot
    # grade a claim with the labeller's own article.
    origin_domain: Optional[str] = None
    # E2 label provenance: primary source URL or citation for verifiable claims
    label_source: Optional[str] = None
    label_note: Optional[str] = None


def load_claims(path) -> List[EvalClaim]:
    import json
    from pathlib import Path
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return [EvalClaim.model_validate(c) for c in data]
