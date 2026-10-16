import { useState, useRef, useEffect, useCallback } from 'react';
import { verifyClaim } from '../lib/api';

// The 429 lockout outlives the component: HomePage unmounts when the user opens
// a result, and the lockout must still hold when they come back. Stored in
// sessionStorage (per tab, also survives a reload); every access is guarded
// because storage can throw (disabled storage, privacy modes, quota).
export const RATE_LIMIT_STORAGE_KEY = 'verifact:blockedUntil';

function readStoredBlock() {
  try {
    const raw = window.sessionStorage.getItem(RATE_LIMIT_STORAGE_KEY);
    const until = raw == null ? NaN : Number(raw);
    if (Number.isFinite(until) && until > Date.now()) return until;
    if (raw != null) window.sessionStorage.removeItem(RATE_LIMIT_STORAGE_KEY);
  } catch {
    // Storage unavailable: fall back to in-memory lockout only.
  }
  return null;
}

function writeStoredBlock(until) {
  try {
    window.sessionStorage.setItem(RATE_LIMIT_STORAGE_KEY, String(until));
  } catch {
    // Storage unavailable: the in-memory lockout still applies on this mount.
  }
}

function clearStoredBlock() {
  try {
    window.sessionStorage.removeItem(RATE_LIMIT_STORAGE_KEY);
  } catch {
    // Storage unavailable: nothing to clear.
  }
}

export function useVerify() {
  const [status, setStatus] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [elapsedS, setElapsedS] = useState(0);
  const [blockedUntil, setBlockedUntil] = useState(readStoredBlock);

  const timerRef = useRef(null);
  const blockTimerRef = useRef(null);
  const startTimeRef = useRef(0);
  const mountedRef = useRef(false);

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const clearBlockTimer = useCallback(() => {
    if (blockTimerRef.current) {
      clearTimeout(blockTimerRef.current);
      blockTimerRef.current = null;
    }
  }, []);

  // Schedules the lockout's expiry on this mount. Only called while mounted.
  const scheduleUnblock = useCallback(
    (until) => {
      clearBlockTimer();
      blockTimerRef.current = setTimeout(() => {
        blockTimerRef.current = null;
        clearStoredBlock();
        setBlockedUntil(null);
      }, Math.max(0, until - Date.now()));
    },
    [clearBlockTimer]
  );

  useEffect(() => {
    mountedRef.current = true;
    // Resume a lockout restored from storage (set by an earlier mount).
    const restored = readStoredBlock();
    if (restored) scheduleUnblock(restored);
    return () => {
      mountedRef.current = false;
      clearTimer();
      clearBlockTimer();
    };
  }, [clearTimer, clearBlockTimer, scheduleUnblock]);

  const submit = useCallback(
    async (text) => {
      if (!text || !text.trim()) return;

      clearTimer();
      setStatus('checking');
      setError(null);
      setResult(null);
      setElapsedS(0);

      startTimeRef.current = Date.now();
      timerRef.current = setInterval(() => {
        const sec = (Date.now() - startTimeRef.current) / 1000;
        setElapsedS(Number(sec.toFixed(1)));
      }, 100);

      try {
        const data = await verifyClaim(text.trim());
        clearTimer();
        // Unmounted mid-check (the user left the page): no state update and no
        // result handed back, so the caller cannot navigate them away.
        if (!mountedRef.current) return undefined;
        setResult(data);
        setStatus('done');
        return data;
      } catch (err) {
        clearTimer();
        if (err.status === 429) {
          const waitS = err.retryAfterS || 60;
          const until = Date.now() + waitS * 1000;
          // Persist first so a lockout that lands after unmount still holds
          // on the next mount; no timer or state update once unmounted.
          writeStoredBlock(until);
          if (!mountedRef.current) return;
          setBlockedUntil(until);
          scheduleUnblock(until);
        }
        if (!mountedRef.current) return;
        setError(err);
        setStatus('error');
      }
    },
    [clearTimer, scheduleUnblock]
  );

  const reset = useCallback(() => {
    clearTimer();
    setStatus('idle');
    setResult(null);
    setError(null);
    setElapsedS(0);
  }, [clearTimer]);

  return {
    status,
    result,
    error,
    elapsedS,
    blockedUntil,
    submit,
    reset,
  };
}
