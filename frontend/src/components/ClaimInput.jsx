import React, { useState, useEffect } from 'react';
import { ERROR_COPY } from '../lib/copy';
import { imageProblem } from '../lib/inputType';
import ImagePicker from './ImagePicker';

const RETRYABLE_CODES = new Set(['all_agents_failed', 'network']);

export default function ClaimInput({
  value = '',
  onChange,
  onSubmit,
  status = 'idle',
  error = null,
  onRetry = null,
  blockedUntil = null,
  inputRef = null,
  onSubmitImage = null,
  inputKind = 'text',
}) {
  // A file refused before upload (wrong type or too large); cleared by the next action.
  const [pickError, setPickError] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  // Any change to the text (typing, an example claim, a prefill) makes a stale file error irrelevant.
  useEffect(() => {
    setPickError(null);
  }, [value]);

  const isChecking = status === 'checking';
  const isOverLimit = value.length > 2000;
  const isBlocked = Boolean(blockedUntil);
  const isSubmitDisabled =
    !value.trim() || isOverLimit || isChecking || isBlocked;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (isSubmitDisabled) return;
    setPickError(null);
    onSubmit?.(value);
  };

  const pickImage = (file) => {
    setPickError(null);
    onSubmitImage?.(file);
  };

  // Drag-and-drop is an enhancement; the "Add screenshot" button always works.
  const canTakeImage = Boolean(onSubmitImage) && !isChecking && !isBlocked;
  const isFileDrag = (e) => Boolean(e.dataTransfer?.types?.includes('Files'));
  // File drags are always prevented (else the browser opens the image and leaves the app);
  // text drags fall through to the textarea.
  const handleDragOver = (e) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    if (canTakeImage) setIsDragOver(true);
  };
  const handleDragLeave = (e) => {
    if (e.currentTarget.contains(e.relatedTarget)) return;
    setIsDragOver(false);
  };
  const handleDrop = (e) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    setIsDragOver(false);
    if (!canTakeImage) return;
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    const problem = imageProblem(file);
    if (problem) setPickError(problem);
    else pickImage(file);
  };

  // Determine error message if error is set
  let errorMessage = null;
  let isRetryAllowed = false;

  if (!error && isBlocked) {
    // Lockout restored from a previous mount (useVerify persists it): explain
    // why submit is disabled even though this mount saw no error.
    errorMessage = ERROR_COPY.rate_limited;
  }

  if (error) {
    const code =
      error.code ||
      (error.status === 0 || error.status === 503
        ? 'network'
        : error.status === 404
        ? 'not_found'
        : error.status === 429
        ? 'rate_limited'
        : null);

    if (code === 'invalid_input' && error.detail) {
      errorMessage = error.detail;
      if (inputKind === 'url') errorMessage += ' ' + ERROR_COPY.url_fallback;
    } else {
      errorMessage = (code && ERROR_COPY[code]) || error.detail || ERROR_COPY.server;
    }

    // Allow-list (spec §6): only a total agent failure and an unreachable
    // service offer "Try again". Everything else (input errors, 429, unknown
    // codes) is fixed by changing the input or waiting, not by resending.
    isRetryAllowed = RETRYABLE_CODES.has(code);
  }

  if (pickError) {
    errorMessage = pickError;
    isRetryAllowed = false;
  }

  return (
    <form onSubmit={handleSubmit} className="w-full flex flex-col gap-3">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative w-full rounded-2xl bg-surface/70 border focus-within:border-porcelain/60 transition-colors shadow-lg ${
          isDragOver ? 'border-porcelain border-dashed' : 'border-slate-700/80'
        }`}
      >
        <textarea
          ref={inputRef}
          value={value}
          onChange={(e) => {
            setPickError(null);
            onChange?.(e.target.value);
          }}
          aria-label="Claim or article link"
          placeholder="Paste a claim or an article link, or add a screenshot…"
          rows={4}
          disabled={isChecking}
          className="w-full p-4 pb-2 bg-transparent text-porcelain placeholder:text-slate-400 text-base sm:text-lg resize-y focus:outline-none rounded-2xl disabled:opacity-60"
        />

        {/* Normal flow under the text (not overlaid), wrapping on narrow screens. */}
        <div className="flex flex-wrap items-center justify-between gap-2 px-4 pb-3">
          <span
            className={`text-xs font-mono font-medium ${
              isOverLimit ? 'text-red-400 font-bold' : 'text-slate-400'
            }`}
          >
            {value.length} / 2000
          </span>

          <div className="flex flex-wrap items-center justify-end gap-2">
          {onSubmitImage && (
            <ImagePicker
              onPick={pickImage}
              onReject={setPickError}
              disabled={isChecking || isBlocked}
            />
          )}
          <button
            type="submit"
            aria-label="Verify Claim"
            disabled={isSubmitDisabled}
            className="min-h-[44px] px-4 py-2 rounded-xl bg-porcelain text-midnight font-bold text-xs uppercase tracking-wider hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed transition-all cursor-pointer shadow"
          >
            {isChecking ? 'Checking…' : 'Verify Claim'}
          </button>
          </div>
        </div>
      </div>

      {/* Error state */}
      {errorMessage && (
        <div role="alert" className="p-4 rounded-xl bg-red-950/30 border border-red-800/50 text-red-200 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs">
          <p className="leading-relaxed">{errorMessage}</p>
          {onRetry && isRetryAllowed && (
            <button
              type="button"
              onClick={onRetry}
              disabled={isBlocked}
              className="px-3 py-1 rounded bg-red-800/40 hover:bg-red-800/60 border border-red-700/60 font-semibold transition-colors shrink-0 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Try again
            </button>
          )}
        </div>
      )}
    </form>
  );
}
