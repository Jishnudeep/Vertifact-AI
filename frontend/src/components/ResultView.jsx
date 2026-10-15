import React from 'react';
import TransitDiagram from './TransitDiagram';
import VerdictLine from './VerdictLine';
import ScoreBreakdown from './ScoreBreakdown';
import SourceList from './SourceList';
import LanguageRadar from './LanguageRadar';
import SearchSuggestions from './SearchSuggestions';
import ResultFooter from './ResultFooter';
import { verdictView } from '../lib/verdict';

export default function ResultView({ result, fresh = false }) {
  if (!result) return null;

  const verdict = verdictView(result);
  const lingAgent = result.agents?.find?.((a) => a.name === 'linguistic');
  const isLingDegraded = lingAgent?.status === 'degraded';

  return (
    <article className="w-full max-w-5xl mx-auto flex flex-col gap-6 py-6 px-4 sm:px-6">
      {/* Claim Title */}
      <div className="border-b border-slate-800 pb-4">
        <blockquote className="text-xl sm:text-2xl font-bold text-porcelain leading-snug">
          “{result.claim}”
        </blockquote>
      </div>

      {/* Transit Diagram signature */}
      <TransitDiagram
        mode="result"
        agents={result.agents || []}
        verdict={verdict}
        capRule={result.trust_score?.cap_rule || null}
      />

      {/* Two Column Section on lg, stacked on small */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 items-start">
        {/* Left Column (3/5) */}
        <div className="lg:col-span-3 flex flex-col gap-6">
          <VerdictLine
            verdict={verdict}
            confidence={result.confidence}
            warnings={result.warnings || []}
          />
          <ScoreBreakdown
            trustScore={result.trust_score}
            evidenceStatus={result.evidence_status}
            agents={result.agents || []}
          />
        </div>

        {/* Right Column (2/5) */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          <SourceList sources={result.sources || []} />
          <LanguageRadar
            profile={result.linguistic_profile}
            isDegraded={isLingDegraded}
          />
          <SearchSuggestions html={result.search_suggestions_html} />
          <ResultFooter
            processingTimeMs={result.processing_time_ms}
            createdAt={result.created_at}
            cached={result.cached}
            fresh={fresh}
            id={result.id}
          />
        </div>
      </div>
    </article>
  );
}
