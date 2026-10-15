class EvidenceAPIError(RuntimeError):
    """
    Raised when an external evidence-retrieval service (Gemini grounded search, Tavily,
    Google Fact Check Tools) could not be reached, returned a non-success status, is not
    configured, or its daily budget is spent.

    This is deliberately distinct from "the API answered with zero results": callers
    must be able to tell an outage from an absence of evidence so the agent can report
    degraded=True and the pipeline can raise the "Partial Analysis" flag
    (CLAUDE.md §4.3 / §7). A missing API key IS an outage (R8.4): the search never ran,
    so "no sources found" would be a false statement.
    """
    def __init__(self, message: str, status_code: int | None = None, provider: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider


class EvidenceBudgetExceeded(EvidenceAPIError):
    """A daily evidence-search budget is spent (TDD §11). An outage, classified QUOTA_EXHAUSTED."""
