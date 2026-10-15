import React, { useState } from 'react';

export default function ResultFooter({
  processingTimeMs,
  createdAt,
  cached = false,
  fresh = false,
  id = null,
}) {
  const [copied, setCopied] = useState(false);

  const handleShare = async () => {
    const url = window.location.href;
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'VeriFact Credibility Assessment',
          url,
        });
        return;
      } catch {
        // Fall back to clipboard if share was cancelled or failed
      }
    }

    if (navigator.clipboard) {
      try {
        await navigator.clipboard.writeText(url);
        setCopied(true);
        setTimeout(() => setCopied(false), 2500);
      } catch {
        // ignore
      }
    }
  };

  let dateDisplay = null;
  if (fresh && cached) {
    dateDisplay = 'Cached result';
  } else if (!fresh && createdAt) {
    const formattedDate = new Date(createdAt).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
    dateDisplay = `Checked on ${formattedDate}`;
  } else if (createdAt) {
    const formattedDate = new Date(createdAt).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
    dateDisplay = `Checked on ${formattedDate}`;
  }

  return (
    <footer className="pt-4 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400 flex-wrap gap-4">
      <div className="flex items-center gap-3">
        {processingTimeMs != null && (
          <span className="font-mono">{processingTimeMs} ms</span>
        )}
        {dateDisplay && (
          <>
            <span>·</span>
            <span>{dateDisplay}</span>
          </>
        )}
      </div>

      {id !== null && id !== undefined && (
        <button
          type="button"
          onClick={handleShare}
          aria-label="Share this check"
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface border border-slate-700 hover:border-slate-500 text-porcelain font-medium transition-colors cursor-pointer"
        >
          {copied ? 'Link copied' : 'Share'}
        </button>
      )}
    </footer>
  );
}
