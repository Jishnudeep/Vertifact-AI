import React from 'react';
import { score100 } from '../lib/verdict';
import { CAP_COPY } from '../lib/copy';

export default function ScoreBreakdown({ trustScore, evidenceStatus, agents = [] }) {
  if (!trustScore) return null;

  const isUnverified = evidenceStatus === 'unverified';
  const weights = trustScore.weights || { w1: 0.5, w2: 0.2, w3: 0.3 };
  const w1Str = (weights.w1 ?? 0.5).toFixed(2);
  const w2Str = (weights.w2 ?? 0.2).toFixed(2);
  const w3Str = (weights.w3 ?? 0.3).toFixed(2);

  const rawScore = trustScore.raw_c_total != null ? score100(trustScore.raw_c_total) : null;
  const finalScore = score100(trustScore.c_total ?? 0);
  const capRule = trustScore.cap_rule || null;

  const agentRows = [
    {
      name: 'Fact-Check',
      key: 'factcheck',
      componentVal: (trustScore.s_fact ?? 0).toFixed(2),
      weight: w1Str,
      points: ((trustScore.s_fact ?? 0) * (weights.w1 ?? 0.5) * 100).toFixed(1),
    },
    {
      name: 'Linguistic',
      key: 'linguistic',
      componentVal: (1 - (trustScore.b_ling ?? 0)).toFixed(2),
      weight: w2Str,
      points: ((1 - (trustScore.b_ling ?? 0)) * (weights.w2 ?? 0.2) * 100).toFixed(1),
    },
    {
      name: 'Cross-Ref',
      key: 'crossref',
      componentVal: (trustScore.v_consensus ?? 0).toFixed(2),
      weight: w3Str,
      points: ((trustScore.v_consensus ?? 0) * (weights.w3 ?? 0.3) * 100).toFixed(1),
    },
  ];

  return (
    <section className="bg-surface/50 border border-slate-800 rounded-xl p-5 flex flex-col gap-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
          How the score was built
        </h3>
        {isUnverified ? (
          <span className="text-xs font-bold text-slate-400">
            Inputs measured — no score given
          </span>
        ) : (
          <span data-testid="total" className="text-xs font-bold text-porcelain">
            Total Score: {finalScore} / 100
          </span>
        )}
      </div>

      {/* Formula view */}
      <div className="font-mono text-xs text-slate-300 bg-midnight/60 p-3 rounded-lg border border-slate-800/80">
        <code>
          C = {w1Str} · S_fact + {w2Str} · (1 − B_ling) + {w3Str} · V_consensus
        </code>
      </div>

      {/* Cap disclosure */}
      {capRule && rawScore !== null && (
        <div className="p-3 rounded-lg bg-red-950/20 border border-red-900/40 text-xs text-red-200 flex flex-col gap-1">
          {!isUnverified && (
            <span className="font-bold">
              Capped from {rawScore} to {finalScore}
            </span>
          )}
          {CAP_COPY[capRule] && <span>{CAP_COPY[capRule]}</span>}
        </div>
      )}

      {/* Agent components table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs text-slate-300">
          <thead className="text-[11px] uppercase tracking-wider text-slate-400 border-b border-slate-800">
            <tr>
              <th className="py-2 pr-3">Agent</th>
              <th className="py-2 pr-3">Status</th>
              <th className="py-2 pr-3 hidden sm:table-cell">Duration</th>
              <th className="py-2 pr-3">Input</th>
              <th className="py-2 pr-3 hidden sm:table-cell">Weight</th>
              {!isUnverified && <th className="py-2 text-right">Points</th>}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono">
            {agentRows.map((row) => {
              const a = agents.find((item) => item.name === row.key);
              const status = a?.status || 'ok';
              const duration =
                a?.duration_ms != null ? `${(a.duration_ms / 1000).toFixed(1)}s` : '—';

              return (
                <tr key={row.key} className="hover:bg-slate-800/20">
                  <td className="py-2 pr-3 font-sans font-semibold text-porcelain">
                    {row.name}
                  </td>
                  <td className="py-2 pr-3 font-sans">
                    {status === 'degraded' ? (
                      <span className="text-slate-300 font-semibold">Partial</span>
                    ) : (
                      <span className="text-slate-400">ok</span>
                    )}
                  </td>
                  <td className="py-2 pr-3 text-slate-400 hidden sm:table-cell">{duration}</td>
                  <td className="py-2 pr-3">{row.componentVal}</td>
                  <td className="py-2 pr-3 text-slate-400 hidden sm:table-cell">{row.weight}</td>
                  {!isUnverified && (
                    <td className="py-2 text-right font-bold text-porcelain">
                      {row.points}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
