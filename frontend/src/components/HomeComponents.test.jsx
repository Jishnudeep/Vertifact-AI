import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import { MemoryRouter } from 'react-router-dom';

import ClaimInput from './ClaimInput';
import RecentList from './RecentList';
import TransitDiagram from './TransitDiagram';
import { ERROR_COPY } from '../lib/copy';
import { RATE_LIMIT_STORAGE_KEY } from '../hooks/useVerify';

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

    it('maps 503 with no code to network copy', () => {
      render(
        <ClaimInput
          onSubmit={vi.fn()}
          status="error"
          error={{ status: 503, code: null }}
          value="Some claim"
          onChange={vi.fn()}
        />
      );

      expect(screen.getByText(ERROR_COPY.network)).toBeInTheDocument();
    });

    it('after a 429, submit is disabled, rate-limited copy is shown and there is no retry button', () => {
      const handleRetry = vi.fn();
      render(
        <ClaimInput
          onSubmit={vi.fn()}
          status="error"
          error={{ status: 429, code: 'rate_limited' }}
          blockedUntil={Date.now() + 60000}
          onRetry={handleRetry}
          value="Some claim"
          onChange={vi.fn()}
        />
      );

      expect(screen.getByText(ERROR_COPY.rate_limited)).toBeInTheDocument();
      const submitBtn = screen.getByRole('button', { name: /Verify Claim/i });
      expect(submitBtn).toBeDisabled();
      expect(screen.queryByRole('button', { name: /Try again/i })).toBeNull();
    });

    it('shows rate-limited copy while blocked even with no error (lockout restored after remount)', () => {
      render(
        <ClaimInput
          onSubmit={vi.fn()}
          status="idle"
          error={null}
          blockedUntil={Date.now() + 30000}
          onRetry={vi.fn()}
          value="Some claim"
          onChange={vi.fn()}
        />
      );
      expect(screen.getByText(ERROR_COPY.rate_limited)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Verify Claim/i })).toBeDisabled();
      expect(screen.queryByRole('button', { name: /Try again/i })).toBeNull();
    });

    // Retry is an allow-list: only all_agents_failed and network/503 errors offer it.
    it.each([
      ['429 rate_limited', { status: 429, code: 'rate_limited' }],
      ['429 without code', { status: 429, code: null }],
      ['422 non_verifiable_claim', { status: 422, code: 'non_verifiable_claim' }],
      ['422 request_validation', { status: 422, code: 'request_validation' }],
      ['422 without code', { status: 422, code: null }],
      ['400 invalid_input', { status: 400, code: 'invalid_input', detail: 'Claim is empty' }],
      ['404 not found', { status: 404, code: null }],
      ['500 without code', { status: 500, code: null }],
      ['unknown future code', { status: 500, code: 'some_new_code' }],
    ])('shows no retry button for %s', (_name, error) => {
      render(
        <ClaimInput
          onSubmit={vi.fn()}
          status="error"
          error={error}
          onRetry={vi.fn()}
          value="Some claim"
          onChange={vi.fn()}
        />
      );
      expect(screen.queryByRole('button', { name: /Try again/i })).toBeNull();
    });

    it.each([
      ['500 all_agents_failed', { status: 500, code: 'all_agents_failed' }],
      ['network (status 0)', { status: 0, code: null, detail: 'Failed to fetch' }],
      ['503 without code', { status: 503, code: null }],
    ])('offers retry for %s', (_name, error) => {
      render(
        <ClaimInput
          onSubmit={vi.fn()}
          status="error"
          error={error}
          onRetry={vi.fn()}
          value="Some claim"
          onChange={vi.fn()}
        />
      );
      expect(screen.getByRole('button', { name: /Try again/i })).toBeEnabled();
    });

    it('shows error.detail for invalid_input when present and does not show retry button', () => {
      const handleRetry = vi.fn();
      render(
        <ClaimInput
          onSubmit={vi.fn()}
          status="error"
          error={{ status: 422, code: 'invalid_input', detail: 'Claim text exceeds 2000 characters' }}
          onRetry={handleRetry}
          value="Some claim"
          onChange={vi.fn()}
        />
      );

      expect(screen.getByText('Claim text exceeds 2000 characters')).toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /Try again/i })).toBeNull();
    });

    it('shows request_validation copy and does not show retry button', () => {
      const handleRetry = vi.fn();
      render(
        <ClaimInput
          onSubmit={vi.fn()}
          status="error"
          error={{ status: 422, code: 'request_validation' }}
          onRetry={handleRetry}
          value="Some claim"
          onChange={vi.fn()}
        />
      );

      expect(screen.getByText(ERROR_COPY.request_validation)).toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /Try again/i })).toBeNull();
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

    it.each([
      ['null', null],
      ['missing', undefined],
      ['unparseable', 'not-a-date'],
    ])('omits the time text when created_at is %s (no made-up recency)', (_name, createdAt) => {
      const row = {
        id: '55555555-5555-5555-5555-555555555555',
        claim: 'A claim whose row has no usable timestamp.',
        c_total: 0.8,
        evidence_status: 'corroborated',
        confidence_level: 'medium',
      };
      if (createdAt !== undefined) row.created_at = createdAt;
      render(
        <MemoryRouter>
          <RecentList recent={[row]} />
        </MemoryRouter>
      );
      const li = screen.getByText(/A claim whose row has no usable timestamp/).closest('li');
      expect(li).toHaveTextContent('Likely true');
      expect(li.textContent).not.toMatch(/just now|ago|·/);
    });

    it('still shows relative time for a valid created_at', () => {
      const row = {
        id: '66666666-6666-6666-6666-666666666666',
        claim: 'A claim with a real timestamp.',
        c_total: 0.8,
        evidence_status: 'corroborated',
        confidence_level: 'medium',
        created_at: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString(),
      };
      render(
        <MemoryRouter>
          <RecentList recent={[row]} />
        </MemoryRouter>
      );
      expect(screen.getByText('3h ago')).toBeInTheDocument();
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

  describe('HomePage recent feed failure resilience', () => {
    it('still renders input and examples when getRecent rejects', async () => {
      const api = await import('../lib/api');
      vi.spyOn(api, 'getRecent').mockRejectedValueOnce(new Error('Feed failed'));
      const HomePage = (await import('../pages/HomePage')).default;

      render(
        <MemoryRouter>
          <HomePage />
        </MemoryRouter>
      );

      expect(
        screen.getByPlaceholderText(/Paste or type a factual claim to verify/i)
      ).toBeInTheDocument();
      expect(screen.getByText(/Try an example claim/i)).toBeInTheDocument();
    });
  });

  describe('HomePage + useVerify rate-limit lockout', () => {
    beforeEach(() => {
      vi.restoreAllMocks();
      window.sessionStorage.clear();
    });

    afterEach(() => {
      window.sessionStorage.clear();
    });

    async function renderHome() {
      const HomePage = (await import('../pages/HomePage')).default;
      return render(
        <MemoryRouter>
          <HomePage />
        </MemoryRouter>
      );
    }

    it('a 429 disables submit, and the lockout survives navigating away and back', async () => {
      const api = await import('../lib/api');
      vi.spyOn(api, 'getRecent').mockResolvedValue([]);
      const verifySpy = vi.spyOn(api, 'verifyClaim').mockRejectedValue(
        new api.ApiError({ status: 429, code: 'rate_limited', detail: '10 per 1 minute', retryAfterS: 60 })
      );

      const first = await renderHome();
      fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Some claim' } });
      fireEvent.click(screen.getByRole('button', { name: /Verify Claim/i }));

      expect(await screen.findByText(ERROR_COPY.rate_limited)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Verify Claim/i })).toBeDisabled();
      expect(screen.queryByRole('button', { name: /Try again/i })).toBeNull();
      expect(verifySpy).toHaveBeenCalledTimes(1);

      // Navigate away (unmount) and come back (fresh mount of HomePage + useVerify).
      first.unmount();
      await renderHome();
      fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Another claim' } });

      const submit = screen.getByRole('button', { name: /Verify Claim/i });
      expect(submit).toBeDisabled();
      expect(screen.getByText(ERROR_COPY.rate_limited)).toBeInTheDocument();
      fireEvent.click(submit);
      expect(verifySpy).toHaveBeenCalledTimes(1);
    });

    it('an expired stored lockout does not block a fresh mount', async () => {
      const api = await import('../lib/api');
      vi.spyOn(api, 'getRecent').mockResolvedValue([]);
      window.sessionStorage.setItem(RATE_LIMIT_STORAGE_KEY, String(Date.now() - 1000));

      await renderHome();
      fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Some claim' } });
      expect(screen.getByRole('button', { name: /Verify Claim/i })).toBeEnabled();
      expect(screen.queryByText(ERROR_COPY.rate_limited)).toBeNull();
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
