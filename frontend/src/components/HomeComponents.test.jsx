import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import { MemoryRouter } from 'react-router-dom';

import ClaimInput from './ClaimInput';
import RecentList from './RecentList';
import TransitDiagram from './TransitDiagram';
import { ERROR_COPY } from '../lib/copy';

import recentFixture from '../fixtures/recent.json';
import errorsFixture from '../fixtures/errors.json';

describe('Home Components', () => {
  describe('ClaimInput', () => {
    it('disables submit during check', () => {
      render(
        <ClaimInput
          onSubmit={vi.fn()}
          status="checking"
          value="Test claim"
          onChange={vi.fn()}
        />
      );

      const submitBtn = screen.getByRole('button', { name: /Verify Claim/i });
      expect(submitBtn).toBeDisabled();
    });

    it('keeps text after error', () => {
      const { rerender } = render(
        <ClaimInput
          onSubmit={vi.fn()}
          status="checking"
          value="My preserved claim"
          onChange={vi.fn()}
        />
      );

      // Now rerender with error status and an error object
      rerender(
        <ClaimInput
          onSubmit={vi.fn()}
          status="error"
          error={errorsFixture.invalid_input}
          value="My preserved claim"
          onChange={vi.fn()}
        />
      );

      const textarea = screen.getByRole('textbox');
      expect(textarea.value).toBe('My preserved claim');
    });

    it('counter blocks 2001 chars and disables submit', () => {
      const long2001 = 'a'.repeat(2001);
      const handleChange = vi.fn();
      render(
        <ClaimInput
          onSubmit={vi.fn()}
          status="idle"
          value={long2001}
          onChange={handleChange}
        />
      );

      expect(screen.getByText('2001 / 2000')).toBeInTheDocument();
      const submitBtn = screen.getByRole('button', { name: /Verify Claim/i });
      expect(submitBtn).toBeDisabled();
    });

    it('shows non_verifiable_claim ERROR_COPY text', () => {
      render(
        <ClaimInput
          onSubmit={vi.fn()}
          status="error"
          error={errorsFixture.non_verifiable_claim}
          value="Is vanilla ice cream good?"
          onChange={vi.fn()}
        />
      );

      expect(
        screen.getByText(ERROR_COPY.non_verifiable_claim)
      ).toBeInTheDocument();
    });

    it('shows all_agents_failed with no score and a retry button', () => {
      const handleRetry = vi.fn();
      render(
        <ClaimInput
          onSubmit={vi.fn()}
          status="error"
          error={errorsFixture.all_agents_failed}
          onRetry={handleRetry}
          value="Some claim"
          onChange={vi.fn()}
        />
      );

      expect(
        screen.getByText(ERROR_COPY.all_agents_failed)
      ).toBeInTheDocument();
      const retryBtn = screen.getByRole('button', { name: /Try again/i });
      expect(retryBtn).toBeInTheDocument();
      fireEvent.click(retryBtn);
      expect(handleRetry).toHaveBeenCalled();
    });

    it('shows network error copy when status is 0 or network error', () => {
      const netError = { status: 0, code: null, detail: 'Failed to fetch' };
      render(
        <ClaimInput
          onSubmit={vi.fn()}
          status="error"
          error={netError}
          value="Some claim"
          onChange={vi.fn()}
        />
      );

      expect(screen.getByText(ERROR_COPY.network)).toBeInTheDocument();
    });
  });

  describe('RecentList', () => {
    it('shows no score for unverified row and displays "Not enough evidence"', () => {
      render(
        <MemoryRouter>
          <RecentList recent={recentFixture} />
        </MemoryRouter>
      );

      // In recentFixture, the 4th item is unverified
      const unverifiedClaim = screen.getByText(
        /The local city council is planning to build a new monorail station in sector 4/i
      );
      expect(unverifiedClaim).toBeInTheDocument();

      const row = unverifiedClaim.closest('li') || unverifiedClaim.closest('a');
      expect(row).toBeInTheDocument();
      expect(row).toHaveTextContent('Not enough evidence');
      expect(row).not.toHaveTextContent('48');
    });

    it('is hidden completely when empty or null', () => {
      const { container } = render(
        <MemoryRouter>
          <RecentList recent={[]} />
        </MemoryRouter>
      );

      expect(container.firstChild).toBeNull();
    });
  });

  describe('checking state rendering', () => {
    it('renders TransitDiagram in checking mode', () => {
      render(
        <TransitDiagram
          mode="checking"
          agents={[]}
          verdict={{ label: '', colorToken: 'verdict-none', showScore: false, score: null, disputed: false }}
          elapsedS={4.5}
        />
      );

      expect(screen.getByText('Checking…')).toBeInTheDocument();
      expect(screen.getByText('4.5 s')).toBeInTheDocument();
    });
  });
});
