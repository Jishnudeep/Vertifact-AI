import React from 'react';

const STANCE_STYLES = {
  supports: 'border-slate-600/70 text-slate-200 bg-transparent',
  refutes: 'border-slate-600/70 text-slate-200 bg-transparent',
  neutral: 'border-slate-700/70 text-slate-400 bg-transparent',
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
