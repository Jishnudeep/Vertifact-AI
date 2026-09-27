class EvidenceAPIError(RuntimeError):
    """
    Raised when an external evidence-retrieval API (Tavily, Google Fact Check Tools)
    could not be reached or returned a non-success status.

    This is deliberately distinct from "the API answered with zero results": callers
    must be able to tell an outage from an absence of evidence so the agent can report
    degraded=True and the pipeline can raise the "Partial Analysis" flag
    (CLAUDE.md §4.3 / §7). A missing API key is NOT an outage — it is an unconfigured
    optional service and still yields an empty result list.
    """
    def __init__(self, message: str, status_code: int | None = None, provider: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider
