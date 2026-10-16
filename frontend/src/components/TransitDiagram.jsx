import React from 'react';
import { AGENT_LABELS, CAP_COPY } from '../lib/copy';

const VERDICT_COLORS = {
  'verdict-false': '#DC2626',
  'verdict-mixed': '#F59E0B',
  'verdict-true': '#10B981',
  'verdict-none': '#8A94B0',
};

export default function TransitDiagram({
  mode = 'result',
  agents = [],
  verdict = { label: '', colorToken: 'verdict-none', showScore: false, score: null, disputed: false },
  capRule = null,
  elapsedS = 0,
}) {
  const isChecking = mode === 'checking';

  const fcAgent = agents?.find?.((a) => a.name === 'factcheck') || null;
  const crAgent = agents?.find?.((a) => a.name === 'crossref') || null;
  const lingAgent = agents?.find?.((a) => a.name === 'linguistic') || null;

  function formatAgentLabel(name, agent) {
    const base = AGENT_LABELS[name] || name;
    if (isChecking) return base;
    let label = base;
    if (agent?.duration_ms != null) {
      label += ` · ${(agent.duration_ms / 1000).toFixed(1)} s`;
    }
    if (agent?.status === 'degraded') {
      label += ' · Partial';
    }
    return label;
  }

  const ringColor = isChecking
    ? '#8A94B0'
    : VERDICT_COLORS[verdict?.colorToken] || '#8A94B0';

  // Build screen-reader summary
  const summaryParts = [];
  if (isChecking) {
    summaryParts.push(`Verification in progress. Elapsed time: ${elapsedS.toFixed(1)} seconds.`);
  } else {
    summaryParts.push(`Verification outcome: ${verdict.label}.`);
    if (verdict.showScore && verdict.score !== null) {
      summaryParts.push(`Composite trust score: ${verdict.score} out of 100.`);
    }
    if (capRule && CAP_COPY[capRule]) {
      summaryParts.push(`Score note: ${CAP_COPY[capRule]}.`);
    }
    ['factcheck', 'crossref', 'linguistic'].forEach((name) => {
      const a = agents?.find?.((item) => item.name === name);
      const label = AGENT_LABELS[name];
      if (a?.status === 'degraded') {
        summaryParts.push(`${label}: degraded (Partial analysis).`);
      } else if (a?.status === 'ok') {
        summaryParts.push(`${label}: completed successfully.`);
      }
    });
  }
  const summaryText = summaryParts.join(' ');

  return (
    <div className="relative w-full max-w-[640px] mx-auto select-none">
      <p id="transit-summary" className="sr-only">
        {summaryText}
      </p>

      <svg
        viewBox="0 0 390 300"
        className="w-full h-auto block"
        role="img"
        aria-describedby="transit-summary"
      >
        <style>{`
          @keyframes transitDash {
            to { stroke-dashoffset: -22; }
          }
          .transit-checking {
            stroke-dasharray: 14 8;
            animation: transitDash 1.2s linear infinite;
          }
          @media (prefers-reduced-motion: reduce) {
            .transit-checking {
              animation: none !important;
            }
          }
        `}</style>

        {/* --- Fact-Check Line (Lilac #B9A2F2) --- */}
        <path
          d="M20 40 H150 L230 120"
          stroke="#B9A2F2"
          strokeWidth="8"
          fill="none"
          strokeLinecap="round"
          className={isChecking ? 'transit-checking' : ''}
          strokeDasharray={!isChecking && fcAgent?.status === 'degraded' ? '8 6' : undefined}
          opacity={!isChecking && fcAgent?.status === 'degraded' ? 0.5 : 1}
          data-status={fcAgent?.status || 'ok'}
        />
        {/* Fact-Check Station dot */}
        <circle cx="80" cy="40" r="6" fill="#F4F1EA" stroke="#B9A2F2" strokeWidth="3" />
        <text
          x="20"
          y="26"
          style={{ font: '700 11px "Hanken Grotesk", sans-serif', fill: '#F4F1EA' }}
        >
          {formatAgentLabel('factcheck', fcAgent)}
        </text>

        {/* --- Cross-Ref Line (Teal #2BB3A3) --- */}
        <path
          d="M20 150 H195"
          stroke="#2BB3A3"
          strokeWidth="8"
          fill="none"
          strokeLinecap="round"
          className={isChecking ? 'transit-checking' : ''}
          strokeDasharray={!isChecking && crAgent?.status === 'degraded' ? '8 6' : undefined}
          opacity={!isChecking && crAgent?.status === 'degraded' ? 0.5 : 1}
          data-status={crAgent?.status || 'ok'}
        />
        {/* Cross-Ref Station dots */}
        <circle cx="70" cy="150" r="6" fill="#F4F1EA" stroke="#2BB3A3" strokeWidth="3" />
        <circle cx="130" cy="150" r="6" fill="#F4F1EA" stroke="#2BB3A3" strokeWidth="3" />
        <text
          x="20"
          y="136"
          style={{ font: '700 11px "Hanken Grotesk", sans-serif', fill: '#F4F1EA' }}
        >
          {formatAgentLabel('crossref', crAgent)}
        </text>

        {/* --- Linguistic Line (Porcelain with Navy casing) --- */}
        <path
          d="M20 260 H150 L210 200"
          stroke="#0F1B3D"
          strokeWidth="12"
          fill="none"
          strokeLinecap="round"
        />
        <path
          d="M20 260 H150 L210 200"
          stroke="#F4F1EA"
          strokeWidth="8"
          fill="none"
          strokeLinecap="round"
          className={isChecking ? 'transit-checking' : ''}
          strokeDasharray={!isChecking && lingAgent?.status === 'degraded' ? '8 6' : undefined}
          opacity={!isChecking && lingAgent?.status === 'degraded' ? 0.5 : 1}
          data-status={lingAgent?.status || 'ok'}
        />
        {/* Linguistic Station dot */}
        <circle cx="90" cy="260" r="6" fill="#F4F1EA" stroke="#0F1B3D" strokeWidth="3" />
        <text
          x="20"
          y="246"
          style={{ font: '700 11px "Hanken Grotesk", sans-serif', fill: '#F4F1EA' }}
        >
          {formatAgentLabel('linguistic', lingAgent)}
        </text>

        {/* --- Interchange Hub --- */}
        {/* Inner white porcelain circle */}
        <circle cx="250" cy="150" r="58" fill="#F4F1EA" stroke="#0F1B3D" strokeWidth="6" />

        {/* Outer verdict ring */}
        <circle
          cx="250"
          cy="150"
          r="64"
          fill="none"
          stroke={ringColor}
          strokeWidth="4"
          style={{
            transition: 'stroke 0.4s ease-out',
          }}
        />

        {/* --- Interchange Content --- */}
        {isChecking ? (
          <>
            <text
              x="250"
              y="146"
              textAnchor="middle"
              style={{
                font: '800 16px "Hanken Grotesk", sans-serif',
                fill: '#0F1B3D',
                letterSpacing: '0.04em',
              }}
            >
              Checking…
            </text>
            <text
              x="250"
              y="168"
              textAnchor="middle"
              style={{
                font: '600 12px "Hanken Grotesk", sans-serif',
                fill: '#475569',
              }}
            >
              {elapsedS.toFixed(1)} s
            </text>
          </>
        ) : verdict.showScore && verdict.score !== null ? (
          <>
            <text
              x="250"
              y="140"
              textAnchor="middle"
              data-testid="interchange-score"
              style={{
                font: '800 40px "Hanken Grotesk", sans-serif',
                fill: '#0F1B3D',
              }}
            >
              {verdict.score}
            </text>
            <text
              x="250"
              y="166"
              textAnchor="middle"
              style={{
                font: '800 12px "Hanken Grotesk", sans-serif',
                fill: '#0F1B3D',
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
              }}
            >
              {verdict.label}
            </text>
          </>
        ) : (
          <text
            x="250"
            y="154"
            textAnchor="middle"
            style={{
              font: '700 11px "Hanken Grotesk", sans-serif',
              fill: '#0F1B3D',
              letterSpacing: '0.02em',
            }}
          >
            Not enough evidence
          </text>
        )}

        {/* Cap note below the interchange circle if present, placed clear of tracks */}
        {!isChecking && capRule && CAP_COPY[capRule] && (
          <text
            x="250"
            y="280"
            textAnchor="middle"
            style={{
              font: '600 11px "Hanken Grotesk", sans-serif',
              fill: '#F4F1EA',
              opacity: 0.9,
            }}
          >
            {CAP_COPY[capRule]}
          </text>
        )}
      </svg>
    </div>
  );
}
