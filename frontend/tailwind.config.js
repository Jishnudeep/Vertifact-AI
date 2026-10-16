/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        midnight: '#0F1B3D',
        surface: '#16244D',
        porcelain: '#F4F1EA',
        'line-factcheck': '#B9A2F2',
        'line-crossref': '#2BB3A3',
        'line-linguistic': '#F4F1EA',
        'verdict-false': '#DC2626',
        'verdict-mixed': '#F59E0B',
        'verdict-true': '#10B981',
        'verdict-none': '#8A94B0',
        'bias-left': '#3B6FD8',
        'bias-leanleft': '#8FB0F0',
        'bias-center': '#9AA3B5',
        'bias-leanright': '#F0A39B',
        'bias-right': '#D9544A',
      },
      fontFamily: {
        sans: ['"Hanken Grotesk"', 'system-ui', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
