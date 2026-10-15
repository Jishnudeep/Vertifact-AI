import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useVerify } from './useVerify';
import * as api from '../lib/api';

describe('useVerify hook', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('rate limit falls back to 60s when retryAfterS is null', async () => {
    const error429 = new api.ApiError({
      status: 429,
      code: 'rate_limited',
      detail: 'Rate limited',
      retryAfterS: null,
    });
    vi.spyOn(api, 'verifyClaim').mockRejectedValue(error429);

    const { result } = renderHook(() => useVerify());
    const before = Date.now();

    await act(async () => {
      await result.current.submit('Test claim');
    });

    expect(result.current.status).toBe('error');
    expect(result.current.error.status).toBe(429);
    expect(result.current.blockedUntil).toBeGreaterThanOrEqual(before + 59000);
    expect(result.current.blockedUntil).toBeLessThanOrEqual(before + 61000);
  });

  it('uses specific retryAfterS on 429 when provided', async () => {
    const error429 = new api.ApiError({
      status: 429,
      code: 'rate_limited',
      detail: 'Rate limited',
      retryAfterS: 30,
    });
    vi.spyOn(api, 'verifyClaim').mockRejectedValue(error429);

    const { result } = renderHook(() => useVerify());
    const before = Date.now();

    await act(async () => {
      await result.current.submit('Test claim');
    });

    expect(result.current.blockedUntil).toBeGreaterThanOrEqual(before + 29000);
    expect(result.current.blockedUntil).toBeLessThanOrEqual(before + 31000);
  });

  it('transitions to done on successful verify', async () => {
    const mockResult = { id: 'test-123', claim: 'Valid claim' };
    vi.spyOn(api, 'verifyClaim').mockResolvedValue(mockResult);

    const { result } = renderHook(() => useVerify());

    await act(async () => {
      await result.current.submit('Valid claim');
    });

    expect(result.current.status).toBe('done');
    expect(result.current.result).toEqual(mockResult);
    expect(result.current.error).toBeNull();
  });
});
