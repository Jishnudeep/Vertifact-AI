import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useVerify, RATE_LIMIT_STORAGE_KEY } from './useVerify';
import * as api from '../lib/api';

describe('useVerify hook', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.sessionStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    window.sessionStorage.clear();
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

  describe('timers and lockout persistence (fake timers)', () => {
    const rateLimited = (retryAfterS = 60) =>
      new api.ApiError({ status: 429, code: 'rate_limited', detail: 'Rate limited', retryAfterS });

    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('lifts the lockout when Retry-After elapses while mounted', async () => {
      vi.spyOn(api, 'verifyClaim').mockRejectedValue(rateLimited(30));
      const { result } = renderHook(() => useVerify());

      await act(async () => {
        await result.current.submit('Test claim');
      });
      expect(result.current.blockedUntil).not.toBeNull();

      await act(async () => {
        vi.advanceTimersByTime(29_000);
      });
      expect(result.current.blockedUntil).not.toBeNull();

      await act(async () => {
        vi.advanceTimersByTime(1_000);
      });
      expect(result.current.blockedUntil).toBeNull();
      expect(window.sessionStorage.getItem(RATE_LIMIT_STORAGE_KEY)).toBeNull();
    });

    it('after unmount no timer remains and no callback fires, but the lockout is kept for the next mount', async () => {
      vi.spyOn(api, 'verifyClaim').mockRejectedValue(rateLimited(60));
      const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
      let renders = 0;
      const { result, unmount } = renderHook(() => {
        renders += 1;
        return useVerify();
      });

      await act(async () => {
        await result.current.submit('Test claim');
      });
      const blockedUntil = result.current.blockedUntil;
      expect(vi.getTimerCount()).toBe(1); // the lockout-expiry timeout

      unmount();
      expect(vi.getTimerCount()).toBe(0);

      const rendersAtUnmount = renders;
      const setItem = vi.spyOn(Storage.prototype, 'setItem');
      const removeItem = vi.spyOn(Storage.prototype, 'removeItem');
      await act(async () => {
        vi.advanceTimersByTime(120_000);
      });
      expect(renders).toBe(rendersAtUnmount);
      expect(setItem).not.toHaveBeenCalled();
      expect(removeItem).not.toHaveBeenCalled();
      expect(consoleError).not.toHaveBeenCalled();
      // The stored deadline is untouched by any stray callback.
      expect(window.sessionStorage.getItem(RATE_LIMIT_STORAGE_KEY)).toBe(String(blockedUntil));
    });

    it('a remount inside the window restores blockedUntil; after expiry it does not', async () => {
      vi.spyOn(api, 'verifyClaim').mockRejectedValue(rateLimited(60));
      const first = renderHook(() => useVerify());
      await act(async () => {
        await first.result.current.submit('Test claim');
      });
      const blockedUntil = first.result.current.blockedUntil;
      first.unmount();

      await act(async () => {
        vi.advanceTimersByTime(20_000);
      });
      const second = renderHook(() => useVerify());
      expect(second.result.current.blockedUntil).toBe(blockedUntil);
      // The remaining 40 s are scheduled again on the new mount.
      await act(async () => {
        vi.advanceTimersByTime(40_000);
      });
      expect(second.result.current.blockedUntil).toBeNull();
      second.unmount();

      const third = renderHook(() => useVerify());
      expect(third.result.current.blockedUntil).toBeNull();
      third.unmount();
    });

    it('a 429 that lands after unmount schedules no timer but still persists the lockout', async () => {
      let rejectRequest;
      vi.spyOn(api, 'verifyClaim').mockImplementation(
        () => new Promise((_resolve, reject) => {
          rejectRequest = reject;
        })
      );
      const { result, unmount } = renderHook(() => useVerify());

      let pending;
      act(() => {
        pending = result.current.submit('Test claim');
      });
      expect(vi.getTimerCount()).toBe(1); // the elapsed-seconds interval

      unmount();
      expect(vi.getTimerCount()).toBe(0);

      await act(async () => {
        rejectRequest(rateLimited(60));
        await pending;
      });
      expect(vi.getTimerCount()).toBe(0);
      const stored = Number(window.sessionStorage.getItem(RATE_LIMIT_STORAGE_KEY));
      expect(stored).toBeGreaterThan(Date.now());

      const next = renderHook(() => useVerify());
      expect(next.result.current.blockedUntil).toBe(stored);
      next.unmount();
    });

    it('survives sessionStorage throwing (e.g. disabled storage)', async () => {
      vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
        throw new Error('SecurityError');
      });
      vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
        throw new Error('SecurityError');
      });
      vi.spyOn(api, 'verifyClaim').mockRejectedValue(rateLimited(60));

      const { result, unmount } = renderHook(() => useVerify());
      expect(result.current.blockedUntil).toBeNull();
      await act(async () => {
        await result.current.submit('Test claim');
      });
      expect(result.current.blockedUntil).not.toBeNull();
      unmount();
    });
  });
});
