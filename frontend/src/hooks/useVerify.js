import { useState, useRef, useEffect, useCallback } from 'react';
import { verifyClaim } from '../lib/api';

export function useVerify() {
  const [status, setStatus] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [elapsedS, setElapsedS] = useState(0);
  const [blockedUntil, setBlockedUntil] = useState(null);

  const timerRef = useRef(null);
  const startTimeRef = useRef(0);

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      clearTimer();
    };
  }, [clearTimer]);

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
        setResult(data);
        setStatus('done');
        return data;
      } catch (err) {
        clearTimer();
        setError(err);
        setStatus('error');
        if (err.status === 429) {
          const waitS = err.retryAfterS || 60;
          setBlockedUntil(Date.now() + waitS * 1000);
          setTimeout(() => setBlockedUntil(null), waitS * 1000);
        }
      }
    },
    [clearTimer]
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
