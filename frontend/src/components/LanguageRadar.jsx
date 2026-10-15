import React from 'react';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  ResponsiveContainer,
} from 'recharts';

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
  const emotionalVal = Math.round((profile.emotional_tone ?? 0) * 100);
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

      <div className="grid grid-cols-2 gap-3 text-xs mb-2">
        <div className="flex justify-between p-2 rounded bg-midnight/50 border border-slate-800/80">
          <span className="text-slate-400">Clickbait Syntax</span>
          <span className="font-mono font-bold text-porcelain">{clickbaitVal}</span>
        </div>
        <div className="flex justify-between p-2 rounded bg-midnight/50 border border-slate-800/80">
          <span className="text-slate-400">Emotional Framing</span>
          <span className="font-mono font-bold text-porcelain">{emotionalVal}</span>
        </div>
        <div className="flex justify-between p-2 rounded bg-midnight/50 border border-slate-800/80">
          <span className="text-slate-400">Sensationalism</span>
          <span className="font-mono font-bold text-porcelain">{sensationalVal}</span>
        </div>
        <div className="flex justify-between p-2 rounded bg-midnight/50 border border-slate-800/80">
          <span className="text-slate-400">Informational Density</span>
          <span className="font-mono font-bold text-porcelain">{densityVal}</span>
        </div>
      </div>

      <div className="w-full h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart cx="50%" cy="50%" outerRadius="75%" data={data}>
            <PolarGrid stroke="#24325C" />
            <PolarAngleAxis
              dataKey="subject"
              tick={{ fill: '#F4F1EA', fontSize: 11, fontWeight: 600 }}
            />
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
