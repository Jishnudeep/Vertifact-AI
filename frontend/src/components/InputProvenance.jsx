import React from 'react';

function domainOf(url) {
  try {
    const parsed = new URL(url);
    if (!/^https?:$/.test(parsed.protocol)) return null; // only ever link http(s)
    return parsed.hostname.replace(/^www\./, '');
  } catch {
    return null;
  }
}

// Where the checked claim came from: an article headline or text read from a screenshot.
// Shown only on a fresh response; a stored result does not keep the source URL or OCR details.
export default function InputProvenance({ result, onEditClaim = null }) {
  const sourceUrl = result?.source_url;
  const domain = sourceUrl ? domainOf(sourceUrl) : null;
  if (domain) {
    return (
      <p className="mt-2 text-xs text-slate-400">
        Headline from{' '}
        <a
          href={sourceUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="underline underline-offset-2 hover:text-porcelain"
        >
          {domain}
        </a>
        {result.scrape_degraded ? ' (page partly blocked; headline only)' : null}
      </p>
    );
  }

  const confidence = result?.ocr_confidence;
  if (typeof confidence === 'number') {
    const text = result.ocr_extracted_text || result.claim || '';
    return (
      <details className="mt-2 text-xs text-slate-400">
        <summary className="cursor-pointer hover:text-porcelain min-h-[44px] flex items-center">
          Text read from your screenshot · {Math.round(confidence * 100)}% confidence
        </summary>
        <p className="mt-2 whitespace-pre-wrap text-slate-300">{text}</p>
        {onEditClaim && (
          <button
            type="button"
            onClick={() => onEditClaim(text)}
            className="mt-2 min-h-[44px] px-3 rounded-lg border border-slate-600 font-semibold hover:border-porcelain/70 cursor-pointer"
          >
            Edit and re-check
          </button>
        )}
      </details>
    );
  }
  return null;
}
