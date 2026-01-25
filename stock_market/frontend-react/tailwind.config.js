/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'trading-dark': '#0a0e27',
        'trading-card': '#141b2d',
        'trading-border': '#1e2742',
        'trading-green': '#00d4aa',
        'trading-red': '#ff4757',
      }
    },
  },
  plugins: [],
}
