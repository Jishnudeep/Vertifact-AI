import React from 'react';

const STANCE_STYLES = {
  supports: 'border-emerald-500/50 text-emerald-300 bg-emerald-950/20',
  refutes: 'border-rose-500/50 text-rose-300 bg-rose-950/20',
  neutral: 'border-slate-500/50 text-slate-300 bg-slate-900/20',
};

export default function StanceChip({ stance }) {
  if (!stance) return null;

  const normalized = String(stance).toLowerCase();
  const label = normalized.charAt(0).toUpperCase() + normalized.slice(1);
  const style = STANCE_STYLES[normalized] || STANCE_STYLES.neutral;

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold border ${style}`}
    >
      {label}
    </span>
  );
}
