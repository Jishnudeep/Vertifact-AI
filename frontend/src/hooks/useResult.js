import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { getResult } from '../lib/api';

export function useResult(id) {
  const location = useLocation();
  const stateResult = location.state?.result;

  const [state, setState] = useState(() => {
    if (!id) {
      return {
        status: 'error',
        result: null,
        error: { status: 404, message: 'Result not found' },
      };
    }
    if (stateResult && stateResult.id === id) {
      return { status: 'done', result: stateResult, error: null };
    }
    return { status: 'loading', result: null, error: null };
  });

  useEffect(() => {
    if (!id || (stateResult && stateResult.id === id)) {
      return;
    }

    let isCancelled = false;

    getResult(id)
      .then((data) => {
        if (!isCancelled) {
          setState({ status: 'done', result: data, error: null });
        }
      })
      .catch((err) => {
        if (!isCancelled) {
          setState({ status: 'error', result: null, error: err });
        }
      });

    return () => {
      isCancelled = true;
    };
  }, [id, stateResult]);

  return state;
}
