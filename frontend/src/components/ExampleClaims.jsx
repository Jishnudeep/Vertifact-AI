import React from 'react';

const EXAMPLES = [
  'NASA successfully landed the Perseverance rover on Mars in 2021.',
  'The James Webb Space Telescope detected atmospheric water vapour on an exoplanet.',
  'Consuming lemon water cures diabetes within 48 hours.',
];

export default function ExampleClaims({ onSelect }) {
  return (
    <div className="flex flex-col gap-2.5">
      <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
        Try an example claim
      </span>
      <div className="flex flex-col gap-2">
        {EXAMPLES.map((claim, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => onSelect?.(claim)}
            className="text-left p-3 rounded-xl bg-surface/40 hover:bg-surface/80 border border-slate-800 hover:border-slate-700 text-xs text-slate-300 hover:text-porcelain transition-all cursor-pointer flex items-center justify-between group"
          >
            <span className="line-clamp-2">“{claim}”</span>
            <span className="text-slate-400 group-hover:text-porcelain text-xs shrink-0 ml-3">
              Fill ↗
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
