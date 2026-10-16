import React from 'react';
import BiasChip from './BiasChip';
import StanceChip from './StanceChip';
import { safeHttpUrl } from '../lib/url';

export default function SourceRow({ source }) {
  if (!source) return null;

  const href = safeHttpUrl(source.url);
  const titleText = source.title || source.url || 'Untitled source';
  const titleClass = 'text-sm font-medium line-clamp-2 break-words';

  return (
    <div className="py-3 border-b border-surface/80 last:border-b-0 flex flex-col gap-1.5">
      <div className="flex items-center justify-between gap-2 flex-wrap text-xs">
        <span className="font-bold text-porcelain/90">{source.publisher || 'Unknown Publisher'}</span>
        <div className="flex items-center gap-1.5">
          <BiasChip biasLabel={source.bias_label} />
          <StanceChip stance={source.stance} />
          {source.rating && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-surface border border-slate-700 text-slate-300 font-medium">
              Rating: {source.rating}
            </span>
          )}
        </div>
      </div>
      {href ? (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className={`${titleClass} text-sky-300 hover:text-sky-200 hover:underline transition-colors`}
        >
          {titleText}
        </a>
      ) : (
        <span className={`${titleClass} text-slate-300`}>{titleText}</span>
      )}
    </div>
  );
}
