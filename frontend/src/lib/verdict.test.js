import { describe, it, expect } from 'vitest';
import { score100, bandFor, verdictView } from './verdict';
import unverifiedFixture from '../fixtures/unverified.json';
import disputedFixture from '../fixtures/disputed_capped.json';
import factCheckedFixture from '../fixtures/fact_checked.json';

describe('verdict logic', () => {
  describe('bandFor', () => {
    it('maps 0..33 to false', () => {
      expect(bandFor(0)).toBe('false');
      expect(bandFor(15)).toBe('false');
      expect(bandFor(33)).toBe('false');
    });

    it('maps 34..66 to mixed', () => {
      expect(bandFor(34)).toBe('mixed');
      expect(bandFor(50)).toBe('mixed');
      expect(bandFor(66)).toBe('mixed');
    });

    it('maps 67..100 to true', () => {
      expect(bandFor(67)).toBe('true');
      expect(bandFor(85)).toBe('true');
      expect(bandFor(100)).toBe('true');
    });
  });

  describe('score100', () => {
    it('scales and rounds properly between 0 and 100', () => {
      expect(score100(0.92)).toBe(92);
      expect(score100(0.334)).toBe(33);
      expect(score100(0.336)).toBe(34);
      expect(score100(1.05)).toBe(100);
      expect(score100(-0.05)).toBe(0);
    });
  });

  describe('verdictView', () => {
    it('handles unverified result without a score', () => {
      const view = verdictView(unverifiedFixture);
      expect(view.showScore).toBe(false);
      expect(view.score).toBeNull();
      expect(view.label).toBe('Not enough evidence');
      expect(view.colorToken).toBe('verdict-none');
      expect(view.disputed).toBe(false);
    });

    it('handles disputed capped result', () => {
      const view = verdictView(disputedFixture);
      expect(view.disputed).toBe(true);
      expect(view.showScore).toBe(true);
      expect(view.score).toBe(33);
      expect(view.label).toBe('Likely false');
      expect(view.colorToken).toBe('verdict-false');
    });

    it('handles fact-checked true result', () => {
      const view = verdictView(factCheckedFixture);
      expect(view.disputed).toBe(false);
      expect(view.showScore).toBe(true);
      expect(view.score).toBe(92);
      expect(view.label).toBe('Likely true');
      expect(view.colorToken).toBe('verdict-true');
    });
  });
});
