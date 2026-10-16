import React from 'react';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  ResponsiveContainer,
} from 'recharts';

// Station-label floor from spec §7: axis text never drops below 11px.
const TICK_FONT_SIZE = 11;
const TICK_LINE_HEIGHT = 13;

// Side axes (left/right) have the least room on a 375px screen, so their
// two-word names stack onto two lines instead of running off the chart.
const SIDE_LABEL_LINES = {
  'Emotional Framing': ['Emotional', 'Framing'],
  'Informational Density': ['Informational', 'Density'],
};

// Chart margins reserve room for the tick labels outside the polar grid:
// top/bottom hold one 11px line, left/right hold "Informational" (the widest
// stacked word, ~67px at 11px/600) plus Recharts' 8px tick offset.
const CHART_MARGIN = { top: 24, bottom: 24, left: 80, right: 80 };

/**
 * Axis label for the radar. Recharts passes x/y (the tick point), textAnchor
 * ("start" | "middle" | "end") and verticalAnchor ("start" | "middle" | "end").
 * Its `fill` prop (the axis stroke) is ignored: labels are always porcelain.
 */
export function RadarAxisTick({
  x,
  y,
  payload,
  textAnchor = 'middle',
  verticalAnchor = 'middle',
}) {
  const label = String(payload?.value ?? '');
  const lines =
    textAnchor !== 'middle' && SIDE_LABEL_LINES[label] ? SIDE_LABEL_LINES[label] : [label];
  const extra = (lines.length - 1) * TICK_LINE_HEIGHT;

  // Baseline offset of the first line: the block sits above the point
  // ("end"), below it ("start") or centred on it ("middle").
  let firstDy;
  if (verticalAnchor === 'end') firstDy = -extra;
  else if (verticalAnchor === 'start') firstDy = TICK_FONT_SIZE * 0.8;
  else firstDy = -extra / 2 + TICK_FONT_SIZE * 0.35;

  return (
    <text
      x={x}
      y={y}
      textAnchor={textAnchor}
      fill="#F4F1EA"
      fontSize={TICK_FONT_SIZE}
      fontWeight={600}
    >
      {lines.map((line, i) => (
        <tspan key={line} x={x} dy={i === 0 ? firstDy : TICK_LINE_HEIGHT}>
          {line}
        </tspan>
      ))}
    </text>
  );
}

export default function LanguageRadar({ profile, isDegraded = false }) {
  if (isDegraded || !profile) {
    return (
      <section className="bg-surface/50 border border-slate-800 rounded-xl p-5 flex flex-col gap-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
          Language Analysis
        </h3>
        <p className="text-sm text-slate-400 italic">
          Language analysis unavailable for this check
        </p>
      </section>
    );
  }

  const clickbaitVal = Math.round((profile.clickbait_syntax ?? 0) * 100);
  const emotionalVal = Math.round((profile.emotional_framing ?? 0) * 100);
  const sensationalVal = Math.round((profile.sensationalism ?? 0) * 100);
  const densityVal = Math.round((profile.informational_density ?? 0) * 100);

  const data = [
    { subject: 'Clickbait Syntax', value: clickbaitVal },
    { subject: 'Emotional Framing', value: emotionalVal },
    { subject: 'Sensationalism', value: sensationalVal },
    { subject: 'Informational Density', value: densityVal },
  ];

  return (
    <section className="bg-surface/50 border border-slate-800 rounded-xl p-5 flex flex-col gap-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
          Language Analysis
        </h3>
      </div>

      <div className="grid grid-cols-2 gap-3 text-xs">
        {data.map((d) => (
          <div
            key={d.subject}
            className="flex justify-between gap-2 p-2 rounded bg-midnight/50 border border-slate-800/80"
          >
            <span className="text-slate-400">{d.subject}</span>
            <span className="font-mono font-bold text-porcelain">{d.value}</span>
          </div>
        ))}
      </div>

      <div className="w-full h-44 sm:h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart
            cx="50%"
            cy="50%"
            outerRadius="100%"
            margin={CHART_MARGIN}
            data={data}
          >
            <PolarGrid stroke="#24325C" />
            <PolarAngleAxis dataKey="subject" tick={<RadarAxisTick />} />
            <Radar
              name="Profile"
              dataKey="value"
              stroke="#F4F1EA"
              fill="#F4F1EA"
              fillOpacity={0.3}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
