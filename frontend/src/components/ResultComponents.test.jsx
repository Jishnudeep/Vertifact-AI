import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';

import ScoreBreakdown from './ScoreBreakdown';
import SourceList from './SourceList';
import BiasChip from './BiasChip';
import StanceChip from './StanceChip';
import LanguageRadar from './LanguageRadar';
import SearchSuggestions from './SearchSuggestions';
import ResultFooter from './ResultFooter';
import ResultView from './ResultView';
import VerdictLine from './VerdictLine';
import SourceRow from './SourceRow';
import { RadarAxisTick } from './LanguageRadar';
import { score100, verdictView } from '../lib/verdict';

// Any verdict-band or alarm hue. Spec §3: verdict colour appears only on the
// interchange ring and the verdict word, so chips and notes must not use it.
const VERDICT_HUE = /amber|red-|rose|emerald|green|yellow|orange|#DC2626|#F59E0B|#10B981/i;

function baseSource(overrides = {}) {
  return {
    url: 'https://example.com/story',
    title: 'Example story',
    publisher: 'Example Pub',
    bias_label: 'Center',
    type: 'corroborating',
    stance: null,
    rating: null,
    found_by: 'crossref',
    ...overrides,
  };
}

import factCheckedFixture from '../fixtures/fact_checked.json';
import unverifiedFixture from '../fixtures/unverified.json';
import disputedFixture from '../fixtures/disputed_capped.json';
import partialFixture from '../fixtures/partial.json';
import legacyFixture from '../fixtures/legacy.json';
import cachedFixture from '../fixtures/cached.json';

describe('Result Components', () => {
  describe('ScoreBreakdown', () => {
    it('shows formula with weights and per-agent table', () => {
      render(
        <ScoreBreakdown
          trustScore={factCheckedFixture.trust_score}
          evidenceStatus={factCheckedFixture.confidence.evidence_status}
          agents={factCheckedFixture.agents}
        />
      );

      expect(screen.getAllByText(/0\.50/).length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText(/0\.20/).length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText(/0\.30/).length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('Fact-Check')).toBeInTheDocument();
      expect(screen.getByText('Cross-Ref')).toBeInTheDocument();
      expect(screen.getByText('Linguistic')).toBeInTheDocument();
    });

    it('hides total when unverified and shows "Inputs measured — no score given"', () => {
      const { container } = render(
        <ScoreBreakdown
          trustScore={unverifiedFixture.trust_score}
          evidenceStatus={unverifiedFixture.confidence.evidence_status}
          agents={unverifiedFixture.agents}
        />
      );

      expect(container.querySelector('[data-testid="total"]')).toBeNull();
      expect(screen.getByText('Inputs measured — no score given')).toBeInTheDocument();
    });

    it('shows cap for disputed_capped fixture', () => {
      render(
        <ScoreBreakdown
          trustScore={disputedFixture.trust_score}
          evidenceStatus={disputedFixture.confidence.evidence_status}
          agents={disputedFixture.agents}
        />
      );

      expect(screen.getByText(/Capped from 43 to 33/i)).toBeInTheDocument();
      expect(screen.getByText(/Reputable sources contradict this claim/i)).toBeInTheDocument();
    });

    it.each([
      ['disputed_capped', disputedFixture],
      ['unverified', unverifiedFixture],
      ['partial', partialFixture],
      ['cached (false_verdict)', cachedFixture],
    ])('cap note uses neutral styling, never a verdict hue (%s)', (_name, fixture) => {
      render(
        <ScoreBreakdown
          trustScore={fixture.trust_score}
          evidenceStatus={fixture.confidence.evidence_status}
          agents={fixture.agents}
        />
      );
      const capNote = screen.getByTestId('cap-note');
      expect(capNote.className).not.toMatch(VERDICT_HUE);
      capNote.querySelectorAll('*').forEach((el) => {
        expect(el.getAttribute('class') || '').not.toMatch(VERDICT_HUE);
      });
    });

    it('shows "—" (not "ok") for every agent status when agents is []', () => {
      render(
        <ScoreBreakdown
          trustScore={factCheckedFixture.trust_score}
          evidenceStatus={factCheckedFixture.confidence.evidence_status}
          agents={[]}
        />
      );
      ['factcheck', 'crossref', 'linguistic'].forEach((name) => {
        const cell = screen.getByTestId(`status-${name}`);
        expect(cell).toHaveTextContent('—');
        expect(cell).not.toHaveTextContent(/ok/i);
      });
      expect(screen.queryByText('ok')).toBeNull();
    });

    it('still shows "ok" and "Partial" for known statuses', () => {
      render(
        <ScoreBreakdown
          trustScore={partialFixture.trust_score}
          evidenceStatus={partialFixture.confidence.evidence_status}
          agents={partialFixture.agents}
        />
      );
      const statuses = ['factcheck', 'crossref', 'linguistic'].map(
        (name) => screen.getByTestId(`status-${name}`).textContent
      );
      expect(statuses).toContain('Partial');
      expect(statuses).toContain('ok');
    });

    it('does not invent weights when trust_score.weights lacks a key', () => {
      const trustScore = {
        ...factCheckedFixture.trust_score,
        weights: { w1: 0.5, w3: 0.3 },
      };
      const { container } = render(
        <ScoreBreakdown
          trustScore={trustScore}
          evidenceStatus={factCheckedFixture.confidence.evidence_status}
          agents={factCheckedFixture.agents}
        />
      );
      // The old fallback silently printed the default 0.20 for w2.
      expect(container.textContent).not.toContain('0.20');
      expect(container.querySelector('code').textContent).toContain('— · (1 − B_ling)');
    });
  });

  describe('SourceRow URL scheme guard', () => {
    it.each([
      ['https', 'https://example.com/story'],
      ['http', 'http://example.com/story'],
    ])('renders an %s URL as a new-tab link', (_scheme, url) => {
      render(<SourceRow source={baseSource({ url })} />);
      const link = screen.getByRole('link', { name: 'Example story' });
      expect(link).toHaveAttribute('href', url);
      expect(link).toHaveAttribute('target', '_blank');
      expect(link).toHaveAttribute('rel', 'noopener noreferrer');
    });

    it.each([
      ['javascript:', 'javascript:alert(document.cookie)'],
      ['javascript: (mixed case, padded)', '  JaVaScRiPt:alert(1)'],
      ['data:', 'data:text/html,<script>alert(1)</script>'],
      ['vbscript:', 'vbscript:msgbox(1)'],
      ['malformed', 'ht!tp://not a url'],
      ['relative', '/relative/path'],
      ['null', null],
    ])('renders %s URL as plain text with no href', (_name, url) => {
      const { container } = render(<SourceRow source={baseSource({ url })} />);
      expect(screen.queryByRole('link')).toBeNull();
      expect(container.querySelector('[href]')).toBeNull();
      expect(screen.getByText('Example story')).toBeInTheDocument();
    });

    it('shows the raw URL text (inert) when an unsafe URL has no title', () => {
      const url = 'javascript:alert(1)';
      const { container } = render(<SourceRow source={baseSource({ url, title: null })} />);
      expect(container.querySelector('[href]')).toBeNull();
      expect(screen.getByText(url)).toBeInTheDocument();
    });
  });

  describe('SourceList and Chips', () => {
    it('groups by found_by so a found_by="factcheck" source with type="corroborating" is under Fact-Check', () => {
      const customSources = [
        {
          url: 'https://example.com/fc',
          title: 'FactCheck Found Source',
          publisher: 'FactCheck Pub',
          bias_label: 'Center',
          type: 'corroborating',
          stance: 'refutes',
          rating: 'False',
          found_by: 'factcheck',
        },
        {
          url: 'https://example.com/cr',
          title: 'CrossRef Found Source',
          publisher: 'CrossRef Pub',
          bias_label: 'Lean Left',
          type: 'corroborating',
          stance: 'supports',
          rating: null,
          found_by: 'crossref',
        },
      ];

      render(<SourceList sources={customSources} />);

      const fcSection = screen.getByTestId('sources-factcheck');
      expect(fcSection).toHaveTextContent('FactCheck Found Source');

      const crSection = screen.getByTestId('sources-crossref');
      expect(crSection).toHaveTextContent('CrossRef Found Source');
    });

    it('renders AllSides attribution with exact text and link', () => {
      render(<SourceList sources={factCheckedFixture.sources} />);

      const link = screen.getByRole('link', {
        name: /Media bias ratings: AllSides\.com \(CC BY-NC 4\.0\)/i,
      });
      expect(link).toBeInTheDocument();
      expect(link).toHaveAttribute('href', 'https://www.allsides.com');
      expect(link).toHaveAttribute('target', '_blank');
      expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'));
    });

    it('BiasChip renders label verbatim, including Unrated and unknown labels', () => {
      const { rerender } = render(<BiasChip biasLabel="Unrated" />);
      expect(screen.getByText('Unrated')).toBeInTheDocument();

      rerender(<BiasChip biasLabel="Custom Unknown Bias" />);
      expect(screen.getByText('Custom Unknown Bias')).toBeInTheDocument();
    });

    it('StanceChip renders nothing when stance is null', () => {
      const { container } = render(<StanceChip stance={null} />);
      expect(container.firstChild).toBeNull();
    });

    it('StanceChip renders words outline-only without emerald or rose color tokens', () => {
      const { container } = render(<StanceChip stance="supports" />);
      expect(screen.getByText('Supports')).toBeInTheDocument();
      const chip = container.querySelector('span');
      expect(chip.className).not.toMatch(/emerald/i);
      expect(chip.className).not.toMatch(/rose/i);
    });
  });

  describe('VerdictLine', () => {
    it('shows band word and separate Disputed badge when disputed is true', () => {
      const verdict = {
        label: 'Likely false',
        colorToken: 'verdict-false',
        showScore: true,
        score: 33,
        disputed: true,
      };
      render(<VerdictLine verdict={verdict} confidence={{ level: 'medium' }} />);
      expect(screen.getByText('Likely false')).toBeInTheDocument();
      expect(screen.getByText('Disputed')).toBeInTheDocument();
    });

    it('Disputed chip is a neutral outline chip, not a verdict hue; verdict word keeps its band colour', () => {
      const verdict = verdictView(disputedFixture);
      render(
        <VerdictLine
          verdict={verdict}
          confidence={disputedFixture.confidence}
          warnings={disputedFixture.confidence.warnings}
        />
      );
      const chip = screen.getByText('Disputed');
      expect(chip.className).not.toMatch(VERDICT_HUE);
      expect(chip.className).toMatch(/\bborder\b/);
      expect(chip.className).toMatch(/bg-transparent/);
      // The first warning is also not a verdict-coloured box.
      const warning = screen.getByText(disputedFixture.confidence.warnings[0]);
      expect(warning.className).not.toMatch(VERDICT_HUE);
      // The verdict word is one of the two places the band colour is allowed.
      expect(screen.getByText('Likely false').className).toContain('#DC2626');
    });
  });

  describe('LanguageRadar', () => {
    it('scales values 0..1 to 0..100 and shows numbers', () => {
      const profile = {
        clickbait_syntax: 0.28,
        emotional_framing: 0.45,
        sensationalism: 0.15,
        informational_density: 0.85,
      };
      render(<LanguageRadar profile={profile} isDegraded={false} />);
      expect(screen.getByText(/28/)).toBeInTheDocument();
      expect(screen.getByText(/45/)).toBeInTheDocument();
      expect(screen.getByText(/Clickbait Syntax/i)).toBeInTheDocument();
      expect(screen.getByText(/Emotional Framing/i)).toBeInTheDocument();
    });

    it('shows degradation message and no chart when degraded', () => {
      const { container } = render(
        <LanguageRadar
          profile={partialFixture.linguistic_profile}
          isDegraded={true}
        />
      );

      expect(
        screen.getByText('Language analysis unavailable for this check')
      ).toBeInTheDocument();
      expect(container.querySelector('.recharts-responsive-container')).toBeNull();
    });

    it('reads emotional_framing only (no emotional_tone fallback)', () => {
      const profile = {
        clickbait_syntax: 0.11,
        emotional_tone: 0.93,
        sensationalism: 0.12,
        informational_density: 0.14,
      };
      const { container } = render(<LanguageRadar profile={profile} isDegraded={false} />);
      expect(container.textContent).not.toContain('93');
    });

    describe('RadarAxisTick (clipping guard at 375px)', () => {
      function renderTick(props) {
        return render(
          <svg>
            <RadarAxisTick x={100} y={100} fill="#F4F1EA" {...props} />
          </svg>
        );
      }

      it.each([
        ['Informational Density', 'end', ['Informational', 'Density']],
        ['Emotional Framing', 'start', ['Emotional', 'Framing']],
      ])('wraps the side label "%s" onto two lines', (value, textAnchor, lines) => {
        const { container } = renderTick({
          payload: { value },
          textAnchor,
          verticalAnchor: 'middle',
        });
        const text = container.querySelector('text');
        const tspans = [...container.querySelectorAll('tspan')].map((t) => t.textContent);
        expect(tspans).toEqual(lines);
        expect(text.getAttribute('text-anchor')).toBe(textAnchor);
        // Font never shrinks below the 11px station-label floor (spec §7).
        expect(Number(text.getAttribute('font-size'))).toBeGreaterThanOrEqual(11);
      });

      it('keeps top/bottom labels on one line', () => {
        const { container } = renderTick({
          payload: { value: 'Clickbait Syntax' },
          textAnchor: 'middle',
          verticalAnchor: 'end',
        });
        const tspans = [...container.querySelectorAll('tspan')].map((t) => t.textContent);
        expect(tspans).toEqual(['Clickbait Syntax']);
      });
    });
  });

  describe('SearchSuggestions', () => {
    it('uses exact sandbox attribute and renders iframe', () => {
      const html = '<div id="test-suggestions">Google Search Suggestions</div>';
      render(<SearchSuggestions html={html} />);

      const iframe = screen.getByTitle('Related Google searches');
      expect(iframe).toBeInTheDocument();
      const sandbox = iframe.getAttribute('sandbox');
      // Spec §5.1: exactly these two tokens, nothing else (no top navigation,
      // no scripts, no same-origin).
      expect(sandbox).toBe('allow-popups allow-popups-to-escape-sandbox');
      expect(sandbox).not.toContain('allow-top-navigation');
      expect(sandbox).not.toContain('allow-scripts');
      expect(sandbox).not.toContain('allow-same-origin');
      expect(iframe.getAttribute('srcdoc')).toBe(html);
    });

    it('renders nothing when html is null', () => {
      const { container } = render(<SearchSuggestions html={null} />);
      expect(container.firstChild).toBeNull();
    });
  });

  describe('ResultFooter', () => {
    it('shows "Cached result" when fresh and cached', () => {
      render(
        <ResultFooter
          processingTimeMs={cachedFixture.processing_time_ms}
          createdAt={cachedFixture.created_at}
          cached={true}
          fresh={true}
          id={cachedFixture.id}
        />
      );

      expect(screen.getByText(/Cached result/i)).toBeInTheDocument();
    });

    it('shows "Checked on" plus date when not fresh', () => {
      render(
        <ResultFooter
          processingTimeMs={factCheckedFixture.processing_time_ms}
          createdAt="2026-10-15T12:00:00Z"
          cached={true}
          fresh={false}
          id={factCheckedFixture.id}
        />
      );

      expect(screen.getByText(/Checked on/i)).toBeInTheDocument();
      expect(screen.queryByText(/^Cached result$/i)).toBeNull();
    });

    it('does not render share button when id is null', () => {
      const { container } = render(
        <ResultFooter
          processingTimeMs={120}
          createdAt="2026-10-15T12:00:00Z"
          cached={false}
          fresh={true}
          id={null}
        />
      );

      expect(container.querySelector('button[aria-label="Share this check"]')).toBeNull();
    });
  });

  describe('ResultView with Legacy Fixture', () => {
    it('renders legacy fixture without throwing', () => {
      expect(() => {
        render(<ResultView result={legacyFixture} fresh={false} />);
      }).not.toThrow();
      expect(screen.getByText(new RegExp(legacyFixture.claim))).toBeInTheDocument();
    });
  });

  describe('ResultView B1 and B3 contract tests', () => {
    it('renders unverified response with no score digits anywhere on the page', () => {
      const { container } = render(<ResultView result={unverifiedFixture} fresh={false} />);
      expect(screen.getAllByText('Not enough evidence').length).toBeGreaterThanOrEqual(1);

      // Interchange circle must not have the numeric score
      const interchangeScore = container.querySelector('[data-testid="interchange-score"]');
      expect(interchangeScore).toBeNull();

      // Verdict line must not render "/ 100"
      expect(screen.queryByText(/\/ 100/)).toBeNull();

      // Breakdown total must not render
      expect(container.querySelector('[data-testid="total"]')).toBeNull();
      expect(screen.getByText('Inputs measured — no score given')).toBeInTheDocument();

      // Breakdown table must not have Points column
      expect(screen.queryByText('Points')).toBeNull();

      // ScoreBreakdown cap disclosure must not print "Capped from"
      expect(screen.queryByText(/Capped from/i)).toBeNull();

      // Cap copy should still show in TransitDiagram and ScoreBreakdown
      expect(screen.getAllByText('No fact-check and little coverage').length).toBeGreaterThanOrEqual(1);

      // No composite score anywhere: neither the capped c_total (0.55 -> 55)
      // nor the pre-cap raw_c_total (0.56 -> 56).
      expect(score100(unverifiedFixture.trust_score.c_total)).toBe(55);
      expect(score100(unverifiedFixture.trust_score.raw_c_total)).toBe(56);
      expect(container.textContent).not.toMatch(/\b55\b|\b56\b/);
      const transitSummary = container.querySelector('#transit-summary');
      expect(transitSummary.textContent).not.toMatch(/\b55\b|\b56\b/);
    });

    it('ignores non-contract top-level warnings / evidence_status (no legacy fallback)', () => {
      // Real VerifyResponse shape carries these only under `confidence`.
      const { confidence, ...rest } = factCheckedFixture;
      const { warnings: _w, evidence_status: _e, ...confidenceWithout } = confidence;
      const offContract = {
        ...rest,
        confidence: confidenceWithout,
        warnings: ['Off-contract top-level warning'],
        evidence_status: 'unverified',
      };
      render(<ResultView result={offContract} fresh={false} />);
      expect(screen.queryByText('Off-contract top-level warning')).toBeNull();
      expect(screen.queryByText('Inputs measured — no score given')).toBeNull();
    });

    it('renders partial analysis warning on screen below the verdict', () => {
      render(<ResultView result={partialFixture} fresh={false} />);
      expect(screen.getByText('Partial analysis: 2/3 agents succeeded')).toBeInTheDocument();
    });

    it('clamps claim > 280 chars with toggle button in ResultView', async () => {
      const longClaimFixture = (await import('../fixtures/long_claim.json')).default;
      render(<ResultView result={longClaimFixture} fresh={false} />);
      const toggleBtn = screen.getByRole('button', { name: /Show full claim/i });
      expect(toggleBtn).toBeInTheDocument();
    });
  });

  describe('ScoreBreakdown responsive and neutral styling', () => {
    it('uses neutral status styling and hides Duration and Weight on mobile', () => {
      render(
        <ScoreBreakdown
          trustScore={partialFixture.trust_score}
          evidenceStatus={partialFixture.confidence.evidence_status}
          agents={partialFixture.agents}
        />
      );

      const partialBadge = screen.getByText('Partial');
      expect(partialBadge.className).not.toMatch(/amber/i);

      const durationHeader = screen.getByText('Duration');
      expect(durationHeader.className).toContain('hidden');
      expect(durationHeader.className).toContain('sm:table-cell');

      const weightHeader = screen.getByText('Weight');
      expect(weightHeader.className).toContain('hidden');
      expect(weightHeader.className).toContain('sm:table-cell');
    });
  });
});
