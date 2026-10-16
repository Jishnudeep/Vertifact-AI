import React from 'react';
import { CONFIDENCE_COPY } from '../lib/copy';

export default function VerdictLine({ verdict, confidence, warnings = [] }) {
  if (!verdict) return null;

  const confidenceText = confidence?.level ? CONFIDENCE_COPY[confidence.level] : null;

  let colorClass = 'text-slate-400';
  if (verdict.colorToken === 'verdict-false') colorClass = 'text-[#DC2626]';
  if (verdict.colorToken === 'verdict-mixed') colorClass = 'text-[#F59E0B]';
  if (verdict.colorToken === 'verdict-true') colorClass = 'text-[#10B981]';

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-3 flex-wrap">
          <h2 className={`text-3xl font-extrabold tracking-tight ${colorClass}`}>
            {verdict.label}
          </h2>
          {/* Neutral outline chip: verdict colour is reserved for the
              interchange ring and the verdict word (spec §3). */}
          {verdict.disputed && (
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase tracking-wider border border-porcelain/60 text-porcelain bg-transparent">
              Disputed
            </span>
          )}
        </div>
        {confidenceText && (
          <span className="text-sm font-semibold text-slate-400">
            {confidenceText}
          </span>
        )}
      </div>

      {warnings && warnings.length > 0 && (
        <p className="mt-1 p-3 rounded-lg bg-midnight/60 border border-slate-600 border-l-4 border-l-porcelain text-porcelain font-semibold text-xs">
          {warnings[0]}
        </p>
      )}
    </div>
  );
}
