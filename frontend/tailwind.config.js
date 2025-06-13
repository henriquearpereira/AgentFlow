/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
      "./src/**/*.{js,jsx,ts,tsx}",
      "./public/index.html"
    ],
    theme: {
      extend: {
        fontFamily: {
          sans: ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'sans-serif'],
          mono: ['Monaco', 'Menlo', 'Ubuntu Mono', 'monospace'],
        },
        colors: {
          primary: {
            50: '#eff6ff',
            500: '#3b82f6',
            600: '#2563eb',
            700: '#1d4ed8',
            900: '#1e3a8a',
          },
          secondary: {
            500: '#8b5cf6',
            600: '#7c3aed',
            700: '#6d28d9',
          }
        },
        animation: {
          'spin-slow': 'spin 3s linear infinite',
          'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
          'bounce-slow': 'bounce 2s infinite',
        },
        backdropBlur: {
          xs: '2px',
        },
        boxShadow: {
          'glow': '0 0 20px rgba(59, 130, 246, 0.5)',
          'glow-lg': '0 0 40px rgba(59, 130, 246, 0.3)',
          'inner-glow': 'inset 0 0 20px rgba(255, 255, 255, 0.1)',
        },
        backgroundImage: {
          'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
          'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        },
      },
    },
    plugins: [],
  }