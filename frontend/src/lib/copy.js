export const ERROR_COPY = {
  invalid_input: "Please enter a valid statement to check.",
  non_verifiable_claim: "VeriFact checks factual claims. This reads as an opinion or a question. Try a statement that can be true or false.",
  request_validation: "Something about this input couldn't be read.",
  rate_limited: "Too many checks in a minute. Try again shortly.",
  all_agents_failed: "All three checks failed, so there is no score.",
  network: "VeriFact can't reach its checking service right now.",
  not_found: "Check not found.",
  server: "An unexpected server error occurred.",
};

export const CAP_COPY = {
  false_verdict: "A fact-check rated this false",
  disputed: "Reputable sources contradict this claim",
  partial_analysis: "Partial analysis: not every check finished",
  low_evidence: "No fact-check and little coverage",
};

export const CONFIDENCE_COPY = {
  high: "High confidence",
  medium: "Medium confidence",
  low: "Low confidence",
  insufficient: "Insufficient confidence",
};

export const AGENT_LABELS = {
  factcheck: "Fact-Check",
  crossref: "Cross-Ref",
  linguistic: "Linguistic",
};
