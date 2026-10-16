import React from 'react';
import { score100 } from '../lib/verdict';
import { CAP_COPY } from '../lib/copy';

// Prints a measured number, or "—" when the API did not send it. Never
// substitutes a default: every number shown must come from the response.
function fmt(value, digits) {
  return Number.isFinite(value) ? value.toFixed(digits) : '—';
}

function points(component, weight) {
  return Number.isFinite(component) && Number.isFinite(weight)
    ? fmt(component * weight * 100, 1)
    : '—';
}

export default function ScoreBreakdown({ trustScore, evidenceStatus, agents = [] }) {
  if (!trustScore) return null;

  const isUnverified = evidenceStatus === 'unverified';
  const { w1, w2, w3 } = trustScore.weights || {};

  const rawScore = trustScore.raw_c_total != null ? score100(trustScore.raw_c_total) : null;
  const finalScore = score100(trustScore.c_total);
  const capRule = trustScore.cap_rule || null;

  const sFact = trustScore.s_fact;
  const lingInput = Number.isFinite(trustScore.b_ling) ? 1 - trustScore.b_ling : null;
  const vConsensus = trustScore.v_consensus;

  const agentRows = [
    { name: 'Fact-Check', key: 'factcheck', component: sFact, weight: w1 },
    { name: 'Linguistic', key: 'linguistic', component: lingInput, weight: w2 },
    { name: 'Cross-Ref', key: 'crossref', component: vConsensus, weight: w3 },
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
          C = {fmt(w1, 2)} · S_fact + {fmt(w2, 2)} · (1 − B_ling) + {fmt(w3, 2)} · V_consensus
        </code>
      </div>

      {/* Cap disclosure: neutral styling for every cap rule. Verdict colour is
          reserved for the interchange ring and the verdict word (spec §3). */}
      {capRule && rawScore !== null && (
        <div
          data-testid="cap-note"
          className="p-3 rounded-lg bg-midnight/60 border border-slate-600 text-xs text-slate-200 flex flex-col gap-1"
        >
          {!isUnverified && (
            <span className="font-bold text-porcelain">
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
              const status = a?.status ?? null;
              const duration =
                a?.duration_ms != null ? `${(a.duration_ms / 1000).toFixed(1)}s` : '—';

              return (
                <tr key={row.key} className="hover:bg-slate-800/20">
                  <td className="py-2 pr-3 font-sans font-semibold text-porcelain">
                    {row.name}
                  </td>
                  <td data-testid={`status-${row.key}`} className="py-2 pr-3 font-sans">
                    {status === 'degraded' ? (
                      <span className="text-slate-300 font-semibold">Partial</span>
                    ) : status === 'ok' ? (
                      <span className="text-slate-400">ok</span>
                    ) : (
                      // No status from the API (e.g. an older stored result):
                      // never presented as healthy.
                      <>
                        <span aria-hidden="true" className="text-slate-500">—</span>
                        <span className="sr-only">Status not recorded</span>
                      </>
                    )}
                  </td>
                  <td className="py-2 pr-3 text-slate-400 hidden sm:table-cell">{duration}</td>
                  <td className="py-2 pr-3">{fmt(row.component, 2)}</td>
                  <td className="py-2 pr-3 text-slate-400 hidden sm:table-cell">{fmt(row.weight, 2)}</td>
                  {!isUnverified && (
                    <td className="py-2 text-right font-bold text-porcelain">
                      {points(row.component, row.weight)}
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
