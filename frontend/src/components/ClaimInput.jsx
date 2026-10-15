import React from 'react';
import { ERROR_COPY } from '../lib/copy';

export default function ClaimInput({
  value = '',
  onChange,
  onSubmit,
  status = 'idle',
  error = null,
  onRetry = null,
  blockedUntil = null,
  inputRef = null,
}) {
  const isChecking = status === 'checking';
  const isOverLimit = value.length > 2000;
  const isBlocked = Boolean(blockedUntil);
  const isSubmitDisabled =
    !value.trim() || isOverLimit || isChecking || isBlocked;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (isSubmitDisabled) return;
    onSubmit?.(value);
  };

  // Determine error message if error is set
  let errorMessage = null;
  if (error) {
    const code = error.code || (error.status === 0 ? 'network' : error.status === 404 ? 'not_found' : null);
    errorMessage = (code && ERROR_COPY[code]) || error.detail || ERROR_COPY.server;
  }

  return (
    <form onSubmit={handleSubmit} className="w-full flex flex-col gap-3">
      <div className="relative w-full rounded-2xl bg-surface/70 border border-slate-700/80 focus-within:border-porcelain/60 transition-colors shadow-lg">
        <textarea
          ref={inputRef}
          value={value}
          onChange={(e) => onChange?.(e.target.value)}
          placeholder="Paste or type a factual claim to verify (e.g. news headline, health statement, statistic)…"
          rows={4}
          disabled={isChecking}
          className="w-full p-4 pb-12 bg-transparent text-porcelain placeholder:text-slate-400 text-base sm:text-lg resize-y focus:outline-none rounded-2xl disabled:opacity-60"
        />

        <div className="absolute bottom-3 left-4 right-4 flex items-center justify-between pointer-events-none">
          <span
            className={`text-xs font-mono font-medium ${
              isOverLimit ? 'text-red-400 font-bold' : 'text-slate-400'
            }`}
          >
            {value.length} / 2000
          </span>

          <button
            type="submit"
            aria-label="Verify Claim"
            disabled={isSubmitDisabled}
            className="pointer-events-auto px-4 py-2 rounded-xl bg-porcelain text-midnight font-bold text-xs uppercase tracking-wider hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition-all cursor-pointer shadow"
          >
            {isChecking ? 'Checking…' : 'Verify Claim'}
          </button>
        </div>
      </div>

      {/* Error state */}
      {errorMessage && (
        <div className="p-4 rounded-xl bg-red-950/30 border border-red-800/50 text-red-200 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs">
          <p className="leading-relaxed">{errorMessage}</p>
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="px-3 py-1 rounded bg-red-800/40 hover:bg-red-800/60 border border-red-700/60 font-semibold transition-colors shrink-0 cursor-pointer"
            >
              Try again
            </button>
          )}
        </div>
      )}
    </form>
  );
}
