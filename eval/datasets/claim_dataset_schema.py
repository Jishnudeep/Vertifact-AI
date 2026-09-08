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
