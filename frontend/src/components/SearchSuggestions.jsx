import React from 'react';

export default function SearchSuggestions({ html }) {
  if (!html) return null;

  return (
    <section className="bg-surface/50 border border-slate-800 rounded-xl p-5 flex flex-col gap-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
          Related Google searches
        </h3>
      </div>
      <div className="w-full min-h-[140px] rounded-lg overflow-hidden bg-white/5">
        <iframe
          title="Related Google searches"
          sandbox="allow-popups allow-popups-to-escape-sandbox"
          srcDoc={html}
          className="w-full min-h-[140px] border-none"
        />
      </div>
    </section>
  );
}
