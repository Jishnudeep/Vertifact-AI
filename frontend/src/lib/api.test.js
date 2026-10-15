import { describe, it, expect, vi, beforeEach } from 'vitest';
import { verifyClaim, getResult, getRecent, ApiError } from './api';

describe('api client', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  describe('ApiError parsing', () => {
    it('parses a 422 ErrorResponse body with code', async () => {
      const mockResponse = {
        ok: false,
        status: 422,
        headers: new Headers(),
        json: async () => ({
          error: 'Not verifiable',
          code: 'non_verifiable_claim',
          detail: 'Try a factual statement.'
        }),
      };
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(mockResponse));

      await expect(verifyClaim('Is vanilla ice cream tasty?')).rejects.toSatisfy((err) => {
        return (
          err instanceof ApiError &&
          err.status === 422 &&
          err.code === 'non_verifiable_claim' &&
          err.detail === 'Try a factual statement.'
        );
      });
    });

    it('parses a 404 {detail} body with null code', async () => {
      const mockResponse = {
        ok: false,
        status: 404,
        headers: new Headers(),
        json: async () => ({ detail: 'Result not found.' }),
      };
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(mockResponse));

      await expect(getResult('00000000-0000-0000-0000-000000000000')).rejects.toMatchObject({
        name: 'ApiError',
        status: 404,
        code: null,
        detail: 'Result not found.',
      });
    });

    it('defaults retryAfterS to 60 on a 429 when no Retry-After header is present', async () => {
      const mockResponse = {
        ok: false,
        status: 429,
        headers: new Headers(),
        json: async () => ({ error: 'Too many requests' }),
      };
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(mockResponse));

      await expect(verifyClaim('Some claim')).rejects.toMatchObject({
        name: 'ApiError',
        status: 429,
        retryAfterS: 60,
      });
    });

    it('reads Retry-After header on 429 when present', async () => {
      const headers = new Headers();
      headers.set('Retry-After', '45');
      const mockResponse = {
        ok: false,
        status: 429,
        headers,
        json: async () => ({ error: 'Too many requests' }),
      };
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(mockResponse));

      await expect(verifyClaim('Some claim')).rejects.toMatchObject({
        name: 'ApiError',
        status: 429,
        retryAfterS: 45,
      });
    });

    it('sets status === 0 when fetch rejects with a network error', async () => {
      vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

      await expect(verifyClaim('Some claim')).rejects.toMatchObject({
        name: 'ApiError',
        status: 0,
        code: null,
        retryAfterS: null,
      });
    });
  });

  describe('successful calls', () => {
    it('verifyClaim posts to /api/verify', async () => {
      const mockResult = { id: 'test-id', claim: 'Hello world' };
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        headers: new Headers(),
        json: async () => mockResult,
      });
      vi.stubGlobal('fetch', fetchMock);

      const res = await verifyClaim('Hello world');
      expect(res).toEqual(mockResult);
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining('/api/verify'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ input: 'Hello world', input_type: 'text' }),
        })
      );
    });

    it('getResult gets /api/results/:id', async () => {
      const mockResult = { id: 'test-id' };
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        headers: new Headers(),
        json: async () => mockResult,
      });
      vi.stubGlobal('fetch', fetchMock);

      const res = await getResult('test-id');
      expect(res).toEqual(mockResult);
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining('/api/results/test-id'),
        expect.anything()
      );
    });

    it('getRecent queries /api/results/recent?limit=X', async () => {
      const mockRecent = [{ id: '1' }, { id: '2' }];
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        headers: new Headers(),
        json: async () => mockRecent,
      });
      vi.stubGlobal('fetch', fetchMock);

      const res = await getRecent(5);
      expect(res).toEqual(mockRecent);
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining('/api/results/recent?limit=5'),
        expect.anything()
      );
    });
  });
});
