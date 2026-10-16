import React from 'react';
import { Link } from 'react-router-dom';
import AppHeader from '../components/AppHeader';

export default function NotFoundPage() {
  return (
    <div className="min-h-screen bg-midnight text-porcelain flex flex-col">
      <AppHeader />
      <main className="flex-1 flex flex-col items-center justify-center p-6 text-center max-w-md mx-auto">
        <h1 className="text-3xl font-extrabold mb-2">Page not found</h1>
        <p className="text-sm text-slate-400 mb-6">
          The requested page does not exist or has been moved.
        </p>
        <Link
          to="/"
          className="px-4 py-2 rounded-lg bg-porcelain text-midnight font-bold text-xs hover:bg-white transition-colors"
        >
          New check
        </Link>
      </main>
    </div>
  );
}
