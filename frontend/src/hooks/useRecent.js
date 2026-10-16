import { useState, useEffect, useCallback } from 'react';
import { getRecent } from '../lib/api';

export function useRecent(limit = 10) {
  const [recent, setRecent] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isCancelled = false;
    getRecent(limit)
      .then((data) => {
        if (!isCancelled) {
          setRecent(Array.isArray(data) ? data : []);
          setError(null);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!isCancelled) {
          setError(err);
          setRecent([]);
          setLoading(false);
        }
      });

    return () => {
      isCancelled = true;
    };
  }, [limit]);

  const reload = useCallback(() => {
    setLoading(true);
    getRecent(limit)
      .then((data) => {
        setRecent(Array.isArray(data) ? data : []);
        setError(null);
      })
      .catch((err) => {
        setError(err);
        setRecent([]);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [limit]);

  return { recent, loading, error, reload };
}
