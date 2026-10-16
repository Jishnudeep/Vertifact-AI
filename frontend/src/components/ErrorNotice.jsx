import React from 'react';
import { ERROR_COPY } from '../lib/copy';

export default function ErrorNotice({ error, onRetry }) {
  if (!error) return null;

  const code = error.code || (error.status === 0 ? 'network' : error.status === 404 ? 'not_found' : null);
  const message = (code && ERROR_COPY[code]) || error.detail || ERROR_COPY.server;

  return (
    <div className="p-4 rounded-xl bg-red-950/30 border border-red-800/50 text-red-200 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
      <div className="flex flex-col gap-1">
        <span className="font-bold text-sm">Verification Notice</span>
        <p className="text-xs text-red-200/90">{message}</p>
      </div>

      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="px-3 py-1.5 rounded-lg bg-red-800/40 hover:bg-red-800/60 border border-red-700/60 text-xs font-semibold text-white transition-colors cursor-pointer shrink-0"
        >
          Try again
        </button>
      )}
    </div>
  );
}
