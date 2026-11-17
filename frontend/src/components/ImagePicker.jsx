import React, { useRef } from 'react';
import { IMAGE_TYPES, imageProblem } from '../lib/inputType';

// "Add screenshot" button. Rejects other types and files over 5 MB before any request.
export default function ImagePicker({ onPick, onReject, disabled = false }) {
  const fileRef = useRef(null);

  const handleChange = (e) => {
    const file = e.target.files?.[0];
    e.target.value = ''; // picking the same file again still fires onChange
    if (!file) return;
    const problem = imageProblem(file);
    if (problem) onReject?.(problem);
    else onPick?.(file);
  };

  return (
    <>
      <input
        ref={fileRef}
        type="file"
        accept={IMAGE_TYPES.join(',')}
        onChange={handleChange}
        className="hidden"
        data-testid="image-input"
        tabIndex={-1}
      />
      <button
        type="button"
        onClick={() => fileRef.current?.click()}
        disabled={disabled}
        className="min-h-[44px] px-4 py-2 rounded-xl border border-slate-600 text-porcelain font-semibold text-xs uppercase tracking-wider hover:border-porcelain/70 disabled:opacity-40 disabled:cursor-not-allowed transition-colors cursor-pointer"
      >
        Add screenshot
      </button>
    </>
  );
}
