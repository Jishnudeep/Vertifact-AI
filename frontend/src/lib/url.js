/**
 * Returns the canonical href for an absolute http(s) URL, or null for anything
 * else (javascript:, data:, vbscript:, relative paths, malformed strings, null).
 * Source URLs come from third-party search results, so they are never used as
 * link targets without this check.
 */
export function safeHttpUrl(url) {
  if (typeof url !== 'string' || !url.trim()) return null;
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:' ? parsed.href : null;
  } catch {
    return null;
  }
}
