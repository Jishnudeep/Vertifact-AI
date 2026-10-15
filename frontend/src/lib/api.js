const BASE_URL =
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE_URL) ||
  'http://localhost:8000';

export class ApiError extends Error {
  constructor({ status, code = null, detail = null, retryAfterS = null, message }) {
    super(message || detail || `API Error: ${status}`);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.detail = detail;
    this.retryAfterS = retryAfterS;
  }
}

async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`;
  let response;

  try {
    response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
      },
    });
  } catch (err) {
    throw new ApiError({
      status: 0,
      code: null,
      detail: err.message,
      retryAfterS: null,
      message: err.message,
    });
  }

  if (!response.ok) {
    let data = null;
    try {
      data = await response.json();
    } catch {
      // non-JSON body
    }

    const status = response.status;
    const headerRetry = response.headers?.get?.('Retry-After');
    let retryAfterS = headerRetry ? parseInt(headerRetry, 10) : null;
    if (status === 429 && retryAfterS === null) {
      retryAfterS = 60;
    }

    const code = data?.code || null;
    const detail = data?.detail || data?.error || null;

    throw new ApiError({
      status,
      code,
      detail,
      retryAfterS,
      message: detail || `HTTP ${status}`,
    });
  }

  return await response.json();
}

/**
 * Verify a claim through the main verification endpoint.
 * @param {string} input - Claim text
 * @param {string} [inputType="text"] - Type of input
 * @returns {Promise<object>} VerifyResponse
 */
export async function verifyClaim(input, inputType = 'text') {
  return request('/api/verify', {
    method: 'POST',
    body: JSON.stringify({ input, input_type: inputType }),
  });
}

/**
 * Retrieve a stored verification result by result ID or claim ID.
 * @param {string} id - UUID
 * @returns {Promise<object>} VerifyResponse
 */
export async function getResult(id) {
  return request(`/api/results/${id}`, {
    method: 'GET',
  });
}

/**
 * Retrieve recent featured results for the landing feed.
 * @param {number} [limit=10]
 * @returns {Promise<Array<object>>} RecentResultModel[]
 */
export async function getRecent(limit = 10) {
  return request(`/api/results/recent?limit=${encodeURIComponent(limit)}`, {
    method: 'GET',
  });
}
