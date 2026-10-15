import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import ResultPage from './ResultPage';
import * as api from '../lib/api';

import factCheckedFixture from '../fixtures/fact_checked.json';

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
});
