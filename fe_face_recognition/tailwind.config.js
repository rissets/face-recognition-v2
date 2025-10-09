/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'cyber-blue': '#00f5ff',
        'cyber-purple': '#7c3aed',
        'cyber-pink': '#ec4899',
        'cyber-green': '#10b981',
        'cyber-orange': '#f97316',
        'dark-bg': '#0a0a0a',
        'dark-surface': '#1a1a1a',
        'dark-card': '#2a2a2a',
      },
      fontFamily: {
        'cyber': ['Orbitron', 'monospace'],
        'futura': ['Futura', 'Arial', 'sans-serif'],
      },
      animation: {
        'glow': 'glow 2s ease-in-out infinite alternate',
        'pulse-glow': 'pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-up': 'slide-up 0.5s ease-out',
        'fade-in': 'fade-in 0.3s ease-in',
      },
      keyframes: {
        glow: {
          'from': { boxShadow: '0 0 5px #00f5ff, 0 0 10px #00f5ff, 0 0 15px #00f5ff' },
          'to': { boxShadow: '0 0 10px #00f5ff, 0 0 20px #00f5ff, 0 0 30px #00f5ff' }
        },
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 5px rgba(0, 245, 255, 0.5)' },
          '50%': { boxShadow: '0 0 20px rgba(0, 245, 255, 0.8), 0 0 30px rgba(0, 245, 255, 0.6)' }
        },
        'slide-up': {
          'from': { transform: 'translateY(20px)', opacity: '0' },
          'to': { transform: 'translateY(0)', opacity: '1' }
        },
        'fade-in': {
          'from': { opacity: '0' },
          'to': { opacity: '1' }
        }
      },
      backdropBlur: {
        'xs': '2px',
      },
      boxShadow: {
        'glow': '0 0 20px rgba(0, 245, 255, 0.5)',
        'glow-lg': '0 0 40px rgba(0, 245, 255, 0.6)',
        'cyber': '0 0 10px rgba(124, 58, 237, 0.5), inset 0 0 10px rgba(0, 0, 0, 0.3)',
      }
    },
  },
  plugins: [],
}