from typing import List

from pydantic import BaseModel


class VerifyRequest(BaseModel):
    """Workflow input (START) and the input of every worker node. `deadline` is a time.monotonic() timestamp."""
    claim: str
    embedding: List[float]
    deadline: float
    bypass_cache: bool = False
