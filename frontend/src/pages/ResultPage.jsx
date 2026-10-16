import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { useResult } from '../hooks/useResult';
import AppHeader from '../components/AppHeader';
import ResultView from '../components/ResultView';
import ErrorNotice from '../components/ErrorNotice';

export default function ResultPage() {
  const { id } = useParams();
  // Keyed by id: a different result always starts from a clean hand-off/fetch.
  return <ResultScreen key={id} id={id} />;
}

function ResultScreen({ id }) {
  // `fresh` is true only for the render that follows a submit; reloads and
  // revisits get false, so the footer reads "Checked on", not "Cached result".
  const { status, result, error, fresh } = useResult(id);

  return (
    <div className="min-h-screen bg-midnight text-porcelain flex flex-col">
      {/* Top Navigation */}
      <AppHeader />

      {/* Main Content */}
      <main className="flex-1 flex flex-col">
        {status === 'loading' && (
          <div className="flex-1 flex flex-col items-center justify-center p-12 text-slate-400">
            <div className="w-8 h-8 rounded-full border-2 border-slate-600 border-t-porcelain animate-spin mb-4" />
            <p className="text-sm font-semibold">Loading verification result…</p>
          </div>
        )}

        {status === 'error' && (
          <div className="max-w-md mx-auto my-auto p-6 text-center flex flex-col items-center gap-4">
            {error?.status === 404 ? (
              <>
                <h1 className="text-2xl font-bold text-porcelain">Check not found</h1>
                <p className="text-sm text-slate-400">
                  This verification result may have expired under our 90-day retention policy or never existed.
                </p>
                <Link
                  to="/"
                  className="mt-2 inline-flex items-center px-4 py-2 rounded-lg bg-porcelain text-midnight font-bold text-xs hover:bg-white transition-colors"
                >
                  New check
                </Link>
              </>
            ) : (
              <ErrorNotice error={error} onRetry={() => window.location.reload()} />
            )}
          </div>
        )}

        {status === 'done' && result && (
          <ResultView result={result} fresh={fresh} />
        )}
      </main>
    </div>
  );
}
