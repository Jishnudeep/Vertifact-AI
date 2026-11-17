// What the user typed: a bare article link is checked as a URL (the backend scrapes its headline);
// anything else, including text that merely contains a link, is checked as a claim.
export function detectInputType(text) {
  const value = (text || '').trim();
  if (!/^https?:\/\/\S+$/i.test(value)) return 'text';
  try {
    const { hostname } = new URL(value);
    return hostname.includes('.') ? 'url' : 'text';
  } catch {
    return 'text';
  }
}

export const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
export const IMAGE_TYPES = ['image/png', 'image/jpeg'];

// Client-side screen before upload; the server checks again (magic bytes, pixel cap).
export function imageProblem(file) {
  if (!file) return 'Choose a screenshot to check.';
  if (!IMAGE_TYPES.includes(file.type) || file.size > MAX_IMAGE_BYTES) {
    return 'Upload a PNG or JPEG under 5 MB.';
  }
  return null;
}
