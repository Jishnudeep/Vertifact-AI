import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { MemoryRouter, Routes, Route, useLocation } from 'react-router-dom';
import ResultPage from './ResultPage';
import * as api from '../lib/api';

import factCheckedFixture from '../fixtures/fact_checked.json';
import cachedFixture from '../fixtures/cached.json';

describe('ResultPage', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('shows not-found message on a 404', async () => {
    const error404 = new api.ApiError({
      status: 404,
      code: null,
      detail: 'Result not found.',
    });
    vi.spyOn(api, 'getResult').mockRejectedValue(error404);

    render(
      <MemoryRouter initialEntries={['/r/unknown-id']}>
        <Routes>
          <Route path="/r/:id" element={<ResultPage />} />
        </Routes>
      </MemoryRouter>
    );

    expect(await screen.findByText(/Check not found/i)).toBeInTheDocument();
    expect(screen.getAllByRole('link', { name: /New check/i }).length).toBeGreaterThanOrEqual(1);
  });

  it('renders result when fetch succeeds', async () => {
    vi.spyOn(api, 'getResult').mockResolvedValue(factCheckedFixture);

    render(
      <MemoryRouter initialEntries={['/r/11111111-1111-1111-1111-111111111111']}>
        <Routes>
          <Route path="/r/:id" element={<ResultPage />} />
        </Routes>
      </MemoryRouter>
    );

    expect(await screen.findByText(new RegExp(factCheckedFixture.claim))).toBeInTheDocument();
    expect(screen.getAllByText('Likely true').length).toBeGreaterThanOrEqual(1);
  });

  describe('fresh-submit hand-off vs reload/revisit', () => {
    function LocationSpy({ onLocation }) {
      const location = useLocation();
      onLocation(location);
      return null;
    }

    function renderAt(entry, onLocation = () => {}) {
      return render(
        <MemoryRouter initialEntries={[entry]}>
          <Routes>
            <Route
              path="/r/:id"
              element={
                <>
                  <ResultPage />
                  <LocationSpy onLocation={onLocation} />
                </>
              }
            />
          </Routes>
        </MemoryRouter>
      );
    }

    it('a fresh cached submit reads "Cached result" once; a reload of the same entry reads "Checked on"', async () => {
      // What GET /api/results/{id} returns for a stored row: cached is false (spec §8.4).
      const getResult = vi
        .spyOn(api, 'getResult')
        .mockResolvedValue({ ...cachedFixture, cached: false });

      let latest;
      const first = renderAt(
        { pathname: `/r/${cachedFixture.id}`, state: { result: cachedFixture, fresh: true } },
        (loc) => {
          latest = loc;
        }
      );

      // First render uses the hand-off: no fetch, "Cached result".
      expect(screen.getByText('Cached result')).toBeInTheDocument();
      expect(getResult).not.toHaveBeenCalled();

      // The hand-off is consumed: the history entry no longer carries it,
      // and the page keeps showing the fresh view without refetching.
      await waitFor(() => expect(latest.state).toBeNull());
      expect(screen.getByText('Cached result')).toBeInTheDocument();
      expect(getResult).not.toHaveBeenCalled();

      // Reload: the browser restores the *current* history entry (state now null).
      const reloadEntry = { pathname: latest.pathname, search: latest.search, hash: latest.hash, state: latest.state };
      first.unmount();
      renderAt(reloadEntry);

      expect(await screen.findByText(/Checked on/)).toBeInTheDocument();
      expect(screen.queryByText('Cached result')).toBeNull();
      expect(getResult).toHaveBeenCalledWith(cachedFixture.id);
    });

    it('ignores a hand-off whose result id does not match the route id', async () => {
      const getResult = vi.spyOn(api, 'getResult').mockResolvedValue(factCheckedFixture);
      renderAt({
        pathname: `/r/${factCheckedFixture.id}`,
        state: { result: cachedFixture, fresh: true },
      });

      expect(await screen.findByText(new RegExp(factCheckedFixture.claim))).toBeInTheDocument();
      expect(getResult).toHaveBeenCalledWith(factCheckedFixture.id);
      expect(screen.queryByText('Cached result')).toBeNull();
    });
  });
});
