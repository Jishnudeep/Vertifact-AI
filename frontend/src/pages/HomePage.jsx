import React, { useState, useRef, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import AppHeader from '../components/AppHeader';
import ClaimInput from '../components/ClaimInput';
import ExampleClaims from '../components/ExampleClaims';
import RecentList from '../components/RecentList';
import TransitDiagram from '../components/TransitDiagram';
import ResultView from '../components/ResultView';
import { useVerify } from '../hooks/useVerify';
import { useRecent } from '../hooks/useRecent';

import factCheckedFix from '../fixtures/fact_checked.json';
import corroboratedFix from '../fixtures/corroborated.json';
import disputedFix from '../fixtures/disputed_capped.json';
import unverifiedFix from '../fixtures/unverified.json';
import partialFix from '../fixtures/partial.json';
import cachedFix from '../fixtures/cached.json';
import legacyFix from '../fixtures/legacy.json';
import longClaimFix from '../fixtures/long_claim.json';

const DEV_FIXTURES = {
  fact_checked: factCheckedFix,
  corroborated: corroboratedFix,
  disputed_capped: disputedFix,
  unverified: unverifiedFix,
  partial: partialFix,
  cached: cachedFix,
  legacy: legacyFix,
  long_claim: longClaimFix,
};

// Lead line while checking; plain text needs none (there is no per-agent progress yet).
const CHECKING_LEAD = { url: 'Reading the article…', image: 'Reading your screenshot…' };

export default function HomePage() {
  const location = useLocation();
  const [claimText, setClaimText] = useState(() =>
    typeof location.state?.prefill === 'string' ? location.state.prefill.slice(0, 2000) : ''
  );
  const inputRef = useRef(null);
  const navigate = useNavigate();

  const { status, result, error, elapsedS, blockedUntil, inputKind, submit, submitImage, retry, reset } =
    useVerify();

  // Consume a prefill once: a reload or back/forward must not bring it back.
  useEffect(() => {
    if (location.state?.prefill != null) {
      navigate({ pathname: location.pathname, search: location.search }, { replace: true, state: null });
    }
  }, [location.state, location.pathname, location.search, navigate]);
  const { recent } = useRecent(10);

  const fixtureParam = import.meta.env.DEV
    ? new URLSearchParams(window.location.search).get('fixture')
    : null;
  const activeFixture = fixtureParam ? DEV_FIXTURES[fixtureParam] : null;

  const openResult = (data) => {
    if (data?.id) {
      navigate(`/r/${data.id}`, { state: { result: data, fresh: true } });
    }
  };

  // useVerify never rethrows: errors land in its state.
  const handleSubmitImage = async (file) => openResult(await submitImage(file));
  const handleSubmit = async (text) => openResult(await submit(text));

  const handleSelectExample = (text) => {
    setClaimText(text);
    inputRef.current?.focus();
  };

  const handleNewCheck = () => {
    reset();
    setClaimText('');
    if (import.meta.env.DEV && fixtureParam) {
      window.history.pushState({}, '', window.location.pathname);
      window.location.reload();
      return;
    }
    inputRef.current?.focus();
  };

  const isChecking = status === 'checking';
  const showInPlaceResult = status === 'done' && result && !result.id;

  return (
    <div className="min-h-screen bg-midnight text-porcelain flex flex-col">
      <AppHeader onNewCheck={handleNewCheck} />

      <main className="flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 py-8 flex flex-col gap-8">
        {activeFixture ? (
          <ResultView result={activeFixture} fresh={fixtureParam === 'cached'} />
        ) : showInPlaceResult ? (
          <ResultView result={result} fresh={true} onEditClaim={(text) => { reset(); setClaimText(text.slice(0, 2000)); }} />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-10 items-start">
            {/* Left Column (3/5 on lg) */}
            <div className="lg:col-span-3 flex flex-col gap-6">
              <div>
                <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight mb-2">
                  Verify any factual claim in seconds.
                </h1>
                <p className="text-sm text-slate-400">
                  Autonomous multi-agent consensus pipeline with transparent source citations and AllSides media bias ratings.
                </p>
              </div>

              <ClaimInput
                inputRef={inputRef}
                value={claimText}
                onChange={setClaimText}
                onSubmit={handleSubmit}
                status={status}
                error={error}
                // A screenshot is replayed as sent; text retries what is in the box now.
                onRetry={async () =>
                  openResult(await (inputKind === 'image' ? retry() : submit(claimText)))
                }
                blockedUntil={blockedUntil}
                onSubmitImage={handleSubmitImage}
                inputKind={inputKind}
              />

              {isChecking ? (
                <div className="p-4 rounded-2xl bg-surface/40 border border-slate-800 flex flex-col items-center">
                  {CHECKING_LEAD[inputKind] && (
                    <p className="text-sm text-slate-300 mb-2" role="status">
                      {CHECKING_LEAD[inputKind]}
                    </p>
                  )}
                  <TransitDiagram
                    mode="checking"
                    agents={[]}
                    verdict={{ label: '', colorToken: 'verdict-none', showScore: false, score: null, disputed: false }}
                    elapsedS={elapsedS}
                  />
                </div>
              ) : (
                <ExampleClaims onSelect={handleSelectExample} />
              )}
            </div>

            {/* Right Column (2/5 on lg): Recently checked */}
            <div className="lg:col-span-2 flex flex-col gap-6">
              <RecentList recent={recent} />
            </div>
          </div>
        )}
      </main>

      {/* Dev-only fixture switcher toolbar */}
      {import.meta.env.DEV && (
        <aside
          aria-label="Dev fixture switcher"
          className="fixed bottom-3 right-3 bg-midnight/95 border border-slate-700 p-2.5 rounded-xl text-xs z-50 flex items-center gap-2 shadow-2xl max-w-full overflow-x-auto"
        >
          <span className="font-bold text-slate-400 uppercase text-[10px] tracking-wider">
            Fixture:
          </span>
          <select
            value={fixtureParam || ''}
            onChange={(e) => {
              const val = e.target.value;
              const url = new URL(window.location.href);
              if (val) {
                url.searchParams.set('fixture', val);
              } else {
                url.searchParams.delete('fixture');
              }
              window.location.href = url.toString();
            }}
            className="bg-surface border border-slate-600 rounded px-2 py-1 text-xs text-porcelain focus:outline-none"
          >
            <option value="">Default (Home)</option>
            {Object.keys(DEV_FIXTURES).map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </aside>
      )}
    </div>
  );
}
