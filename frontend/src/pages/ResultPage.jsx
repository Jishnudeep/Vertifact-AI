import React from 'react';
import { useParams, Link, useLocation } from 'react-router-dom';
import { useResult } from '../hooks/useResult';
import ResultView from '../components/ResultView';
import ErrorNotice from '../components/ErrorNotice';

export default function ResultPage() {
  const { id } = useParams();
  const location = useLocation();
  const { status, result, error } = useResult(id);

  const fresh = location.state?.fresh ?? false;

  return (
    <div className="min-h-screen bg-midnight text-porcelain flex flex-col">
      {/* Top Navigation */}
      <header className="border-b border-slate-800 bg-midnight/90 backdrop-blur sticky top-0 z-20">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
          <Link
            to="/"
            className="text-lg font-extrabold tracking-tight hover:opacity-90 transition-opacity"
          >
            VeriFact AI
          </Link>
          <Link
            to="/"
            className="text-xs font-bold px-3.5 py-1.5 rounded-full bg-porcelain text-midnight hover:bg-white transition-colors"
          >
            New check
          </Link>
        </div>
      </header>

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
