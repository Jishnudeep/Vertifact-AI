import React from 'react';
import { Link } from 'react-router-dom';
import { score100, bandFor } from '../lib/verdict';

const DOT_COLORS = {
  false: 'bg-[#DC2626]',
  mixed: 'bg-[#F59E0B]',
  true: 'bg-[#10B981]',
  unverified: 'bg-[#8A94B0]',
};

function formatRelativeTime(dateString) {
  if (!dateString) return '';
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now - date;
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHours = Math.floor(diffMin / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffDays > 0) return `${diffDays}d ago`;
  if (diffHours > 0) return `${diffHours}h ago`;
  if (diffMin > 0) return `${diffMin}m ago`;
  return 'just now';
}

export default function RecentList({ recent = [] }) {
  if (!recent || recent.length === 0) {
    return null;
  }

  const items = recent.slice(0, 10);

  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
          Recently Checked
        </h3>
      </div>

      <ul className="flex flex-col gap-2">
        {items.map((item) => {
          const isUnverified = item.evidence_status === 'unverified';
          let dotColor = DOT_COLORS.unverified;
          let verdictText = 'Not enough evidence';

          if (!isUnverified) {
            const score = score100(item.c_total ?? 0);
            const band = bandFor(score);
            dotColor = DOT_COLORS[band] || DOT_COLORS.mixed;
            if (band === 'false') verdictText = 'Likely false';
            else if (band === 'true') verdictText = 'Likely true';
            else verdictText = 'Mixed';
          }

          const relativeTime = formatRelativeTime(item.created_at);

          return (
            <li key={item.id}>
              <Link
                to={`/r/${item.id}`}
                className="p-3 rounded-xl bg-surface/30 hover:bg-surface/80 border border-slate-800/80 hover:border-slate-700 transition-all flex items-start gap-3 group"
              >
                <span
                  className={`w-2.5 h-2.5 rounded-full mt-1.5 shrink-0 ${dotColor}`}
                />
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-slate-200 group-hover:text-porcelain line-clamp-2 leading-relaxed">
                    “{item.claim}”
                  </p>
                  <div className="flex items-center gap-2 mt-1 text-[11px] text-slate-400">
                    <span className="font-semibold text-slate-300">
                      {verdictText}
                    </span>
                    {relativeTime && (
                      <>
                        <span>·</span>
                        <span>{relativeTime}</span>
                      </>
                    )}
                  </div>
                </div>
              </Link>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
