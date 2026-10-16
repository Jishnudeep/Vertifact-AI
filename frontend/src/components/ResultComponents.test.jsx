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

    it('StanceChip renders words first outline when stance is present', () => {
      render(<StanceChip stance="supports" />);
      expect(screen.getByText('Supports')).toBeInTheDocument();
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
  });

  describe('SearchSuggestions', () => {
    it('uses exact sandbox attribute and renders iframe', () => {
      const html = '<div id="test-suggestions">Google Search Suggestions</div>';
      render(<SearchSuggestions html={html} />);

      const iframe = screen.getByTitle('Related Google searches');
      expect(iframe).toBeInTheDocument();
      expect(iframe.getAttribute('sandbox')).toBe(
        'allow-popups allow-popups-to-escape-sandbox'
      );
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

      // Verify no score number is rendered
      expect(screen.queryByText(/55/)).toBeNull();
    });

    it('renders partial analysis warning on screen below the verdict', () => {
      render(<ResultView result={partialFixture} fresh={false} />);
      expect(screen.getByText('Partial analysis: 2/3 agents succeeded')).toBeInTheDocument();
    });
  });
});
