import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';
import TransitDiagram from './TransitDiagram';
import { verdictView } from '../lib/verdict';

import factCheckedFixture from '../fixtures/fact_checked.json';
import unverifiedFixture from '../fixtures/unverified.json';
import partialFixture from '../fixtures/partial.json';
import legacyFixture from '../fixtures/legacy.json';

describe('TransitDiagram', () => {
  it('renders fact_checked with "Likely true" and the score', () => {
    const verdict = verdictView(factCheckedFixture);
    render(
      <TransitDiagram
        mode="result"
        agents={factCheckedFixture.agents}
        verdict={verdict}
        capRule={factCheckedFixture.trust_score.cap_rule}
        elapsedS={1.2}
      />
    );

    expect(screen.getByText('Likely true')).toBeInTheDocument();
    expect(screen.getByText('94')).toBeInTheDocument();
  });

  it('renders unverified with "Not enough evidence" and no digits inside the circle', () => {
    const verdict = verdictView(unverifiedFixture);
    const { container } = render(
      <TransitDiagram
        mode="result"
        agents={unverifiedFixture.agents}
        verdict={verdict}
        capRule={unverifiedFixture.trust_score.cap_rule}
        elapsedS={1.0}
      />
    );

    expect(screen.getByText('Not enough evidence')).toBeInTheDocument();
    // Confirm no score digits inside the interchange text
    const scoreText = container.querySelector('[data-testid="interchange-score"]');
    expect(scoreText).toBeNull();
  });

  it('marks degraded line data-status="degraded" and summary contains Partial', () => {
    const verdict = verdictView(partialFixture);
    const { container } = render(
      <TransitDiagram
        mode="result"
        agents={partialFixture.agents}
        verdict={verdict}
        capRule={partialFixture.trust_score.cap_rule}
        elapsedS={8.2}
      />
    );

    const degradedLine = container.querySelector('[data-status="degraded"]');
    expect(degradedLine).not.toBeNull();
    const summary = container.querySelector('#transit-summary');
    expect(summary).toBeInTheDocument();
    expect(summary.textContent).toContain('Partial');
  });

  it('shows "Checking…" and no score in checking mode', () => {
    const verdict = {
      label: '',
      colorToken: 'verdict-none',
      showScore: false,
      score: null,
      disputed: false,
    };
    const { container } = render(
      <TransitDiagram
        mode="checking"
        agents={[]}
        verdict={verdict}
        capRule={null}
        elapsedS={3.4}
      />
    );

    expect(screen.getByText('Checking…')).toBeInTheDocument();
    const scoreText = container.querySelector('[data-testid="interchange-score"]');
    expect(scoreText).toBeNull();
  });

  it('renders no "· s" when agent durations are null (legacy fixture)', () => {
    const verdict = verdictView(legacyFixture);
    const { container } = render(
      <TransitDiagram
        mode="result"
        agents={legacyFixture.agents}
        verdict={verdict}
        capRule={legacyFixture.trust_score.cap_rule}
        elapsedS={1.3}
      />
    );

    expect(container.textContent).not.toContain('· s');
    expect(container.textContent).not.toMatch(/·\s*\d+(\.\d+)?\s*s/);
  });
});
