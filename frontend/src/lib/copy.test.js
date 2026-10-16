import { describe, it, expect } from 'vitest';
import { CAP_COPY } from './copy';

describe('CAP_COPY', () => {
  it('has entries for all four API schema Literal values', () => {
    const apiCapRules = ['false_verdict', 'disputed', 'partial_analysis', 'low_evidence'];
    for (const rule of apiCapRules) {
      expect(CAP_COPY[rule], `Missing CAP_COPY entry for ${rule}`).toBeDefined();
      expect(typeof CAP_COPY[rule]).toBe('string');
      expect(CAP_COPY[rule].length).toBeGreaterThan(0);
    }
  });

  it('contains exactly the four API values with no obsolete keys', () => {
    expect(Object.keys(CAP_COPY).sort()).toEqual(
      ['disputed', 'false_verdict', 'low_evidence', 'partial_analysis'].sort()
    );
  });
});
