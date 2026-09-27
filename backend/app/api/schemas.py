from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, AliasChoices

class VerifyRequest(BaseModel):
    input: str = Field(..., max_length=10000, description="The claim text to verify", validation_alias=AliasChoices("input", "text"))
    input_type: Literal["text", "url", "image"] = Field(default="text", description="Type of input provided")

class TrustScoreModel(BaseModel):
    c_total: float = Field(..., ge=0.0, le=1.0)
    s_fact: float = Field(..., ge=0.0, le=1.0)
    b_ling: float = Field(..., ge=0.0, le=1.0)
    v_consensus: float = Field(..., ge=0.0, le=1.0)
    v_refute: float = Field(default=0.0, ge=0.0, le=1.0)
    weights: Dict[str, float]

class ConfidenceModel(BaseModel):
    level: Literal["high", "medium", "low", "insufficient"]
    warnings: List[str] = Field(default_factory=list)
    verdict_found: bool = Field(default=False)
    evidence_status: Literal["fact_checked", "disputed", "corroborated", "unverified"] = "unverified"

class LinguisticProfileModel(BaseModel):
    clickbait_syntax: float
    emotional_framing: float
    sensationalism: float
    informational_density: float

class SourceCitationModel(BaseModel):
    url: Optional[str] = None
    title: Optional[str] = None
    publisher: str
    rating: Optional[str] = None
    bias_label: str
    type: Literal["fact_check", "corroborating"]
    stance: Optional[Literal["supports", "refutes", "neutral"]] = None

class VerifyResponse(BaseModel):
    id: Optional[str] = None
    claim: str
    trust_score: TrustScoreModel
    confidence: ConfidenceModel
    linguistic_profile: LinguisticProfileModel
    sources: List[SourceCitationModel]
    processing_time_ms: int
    cached: bool = False

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
