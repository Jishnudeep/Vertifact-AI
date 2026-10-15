import React from 'react';
import { CONFIDENCE_COPY } from '../lib/copy';

export default function VerdictLine({ verdict, confidence, warnings = [] }) {
  if (!verdict) return null;

  const label = verdict.disputed ? 'Disputed' : verdict.label;
  const confidenceText = confidence?.level ? CONFIDENCE_COPY[confidence.level] : null;

  let colorClass = 'text-slate-400';
  if (verdict.colorToken === 'verdict-false') colorClass = 'text-[#DC2626]';
  if (verdict.colorToken === 'verdict-mixed') colorClass = 'text-[#F59E0B]';
  if (verdict.colorToken === 'verdict-true') colorClass = 'text-[#10B981]';

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <h2 className={`text-3xl font-extrabold tracking-tight ${colorClass}`}>
          {label}
        </h2>
        {confidenceText && (
          <span className="text-sm font-semibold text-slate-400">
            {confidenceText}
          </span>
        )}
      </div>

      {warnings && warnings.length > 0 && (
        <div className="mt-1 p-3 rounded-lg bg-amber-950/30 border border-amber-800/40 text-amber-200 text-xs">
          {warnings[0]}
        </div>
      )}
    </div>
  );
}
