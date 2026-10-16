import React from 'react';
import SourceRow from './SourceRow';

export default function SourceList({ sources = [] }) {
  const safeSources = Array.isArray(sources) ? sources : [];

  const fcSources = safeSources.filter(
    (s) => s.found_by === 'factcheck' || (!s.found_by && s.type === 'fact_check')
  );
  const crSources = safeSources.filter(
    (s) => s.found_by === 'crossref' || (!s.found_by && s.type !== 'fact_check')
  );

  return (
    <section className="bg-surface/50 border border-slate-800 rounded-xl p-5 flex flex-col gap-5">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
          Sources & Citations ({safeSources.length})
        </h3>
      </div>

      {safeSources.length === 0 ? (
        <p className="text-sm text-slate-400 italic">No external sources cited for this claim.</p>
      ) : (
        <div className="flex flex-col gap-6">
          {fcSources.length > 0 && (
            <div data-testid="sources-factcheck" className="flex flex-col gap-1">
              <h4 className="text-xs font-semibold text-[#B9A2F2] flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-[#B9A2F2] inline-block" />
                Fact-Check Sources ({fcSources.length})
              </h4>
              <div className="flex flex-col">
                {fcSources.map((s, idx) => (
                  <SourceRow key={s.url || idx} source={s} />
                ))}
              </div>
            </div>
          )}

          {crSources.length > 0 && (
            <div data-testid="sources-crossref" className="flex flex-col gap-1">
              <h4 className="text-xs font-semibold text-[#2BB3A3] flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-[#2BB3A3] inline-block" />
                Cross-Reference Coverage ({crSources.length})
              </h4>
              <div className="flex flex-col">
                {crSources.map((s, idx) => (
                  <SourceRow key={s.url || idx} source={s} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="pt-3 border-t border-slate-800/80 text-xs text-slate-400">
        <a
          href="https://www.allsides.com"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:text-slate-200 underline transition-colors"
        >
          Media bias ratings: AllSides.com (CC BY-NC 4.0)
        </a>
      </div>
    </section>
  );
}
