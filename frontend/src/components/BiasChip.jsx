import React from 'react';

const BIAS_STYLES = {
  Left: 'text-[#3B6FD8] border-[#3B6FD8]/40 bg-[#3B6FD8]/10',
  'Lean Left': 'text-[#8FB0F0] border-[#8FB0F0]/40 bg-[#8FB0F0]/10',
  Center: 'text-[#9AA3B5] border-[#9AA3B5]/40 bg-[#9AA3B5]/10',
  'Lean Right': 'text-[#F0A39B] border-[#F0A39B]/40 bg-[#F0A39B]/10',
  Right: 'text-[#D9544A] border-[#D9544A]/40 bg-[#D9544A]/10',
  Unrated: 'text-slate-400 border-slate-600 bg-slate-800/20',
};

export default function BiasChip({ biasLabel = 'Unrated' }) {
  const label = biasLabel || 'Unrated';
  const style = BIAS_STYLES[label] || BIAS_STYLES.Unrated;

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold border ${style}`}
    >
      {label}
    </span>
  );
}
