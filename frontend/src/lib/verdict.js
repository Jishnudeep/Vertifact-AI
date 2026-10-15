/**
 * Helper to scale a 0..1 c_total float to an integer 0..100 score.
 * Clamped strictly to [0, 100].
 */
export function score100(cTotal) {
  if (typeof cTotal !== 'number' || Number.isNaN(cTotal)) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round(cTotal * 100)));
}

/**
 * Maps a 0..100 score to the 3 TDD §9.1 verdict bands:
 * - 0..33 -> "false"
 * - 34..66 -> "mixed"
 * - 67..100 -> "true"
 */
export function bandFor(score) {
  if (score <= 33) return 'false';
  if (score <= 66) return 'mixed';
  return 'true';
}

/**
 * Produces the verdict presentation model from a VerifyResponse.
 * @param {object} resp - VerifyResponse object
 * @returns {{ label: string, colorToken: string, showScore: boolean, score: number|null, disputed: boolean }}
 */
export function verdictView(resp) {
  if (!resp) {
    return {
      label: 'Not enough evidence',
      colorToken: 'verdict-none',
      showScore: false,
      score: null,
      disputed: false,
    };
  }

  const isUnverified = resp.evidence_status === 'unverified';
  if (isUnverified) {
    return {
      label: 'Not enough evidence',
      colorToken: 'verdict-none',
      showScore: false,
      score: null,
      disputed: false,
    };
  }

  const cTotal = resp.trust_score?.c_total ?? 0;
  const score = score100(cTotal);
  const band = bandFor(score);

  const disputed =
    resp.evidence_status === 'disputed' ||
    resp.trust_score?.cap_rule === 'disputed';

  let label = 'Mixed';
  let colorToken = 'verdict-mixed';

  if (band === 'false') {
    label = 'Likely false';
    colorToken = 'verdict-false';
  } else if (band === 'true') {
    label = 'Likely true';
    colorToken = 'verdict-true';
  }

  return {
    label,
    colorToken,
    showScore: true,
    score,
    disputed,
  };
}
