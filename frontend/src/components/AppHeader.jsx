import React from 'react';
import { Link } from 'react-router-dom';

export default function AppHeader({ onNewCheck }) {
  return (
    <header className="border-b border-slate-800 bg-midnight/90 backdrop-blur sticky top-0 z-20">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
        <Link
          to="/"
          className="text-lg font-extrabold tracking-tight text-porcelain hover:opacity-90 transition-opacity"
        >
          VeriFact AI
        </Link>
        <Link
          to="/"
          onClick={onNewCheck}
          className="text-xs font-bold px-3.5 py-1.5 rounded-full bg-porcelain text-midnight hover:bg-white transition-colors"
        >
          New check
        </Link>
      </div>
    </header>
  );
}
