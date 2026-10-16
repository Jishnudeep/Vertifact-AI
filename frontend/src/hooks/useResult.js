import { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { getResult } from '../lib/api';

/**
 * Loads a result for /r/:id.
 *
 * A fresh submit hands its response over in navigation state
 * ({ result, fresh: true }) so the page renders without a refetch and can say
 * "Cached result" when the submit was answered from the cache. That hand-off
 * is read once, at mount, and then removed from the history entry: browsers
 * keep history state across reloads and back/forward, and a revisit must read
 * the stored copy ("Checked on <date>", spec §5.1), not the fresh-submit view.
 *
 * Callers should key the component by `id` so a new id starts clean.
 */
export function useResult(id) {
  const location = useLocation();
  const navigate = useNavigate();

  const [handoff] = useState(() => {
    const navState = location.state;
    if (id && navState?.result && navState.result.id === id) {
      return { result: navState.result, fresh: navState.fresh === true };
    }
    return null;
  });

  const [state, setState] = useState(() => {
    if (!id) {
      return {
        status: 'error',
        result: null,
        error: { status: 404, message: 'Result not found' },
      };
    }
    if (handoff) {
      return { status: 'done', result: handoff.result, error: null };
    }
    return { status: 'loading', result: null, error: null };
  });

  // Consume the hand-off: replace the current entry with the same URL and no
  // state, so a reload or a later revisit of this entry fetches the stored copy.
  const hasNavState = location.state != null;
  const { pathname, search, hash } = location;
  useEffect(() => {
    if (hasNavState) {
      navigate({ pathname, search, hash }, { replace: true, state: null });
    }
  }, [hasNavState, pathname, search, hash, navigate]);

  useEffect(() => {
    if (!id || handoff) {
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
  }, [id, handoff]);

  return { ...state, fresh: handoff?.fresh ?? false };
}
