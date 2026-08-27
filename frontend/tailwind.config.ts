import type { Config } from "tailwindcss";

export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { DEFAULT: "#111827", soft: "#374151", muted: "#6b7280" },
        surface: { DEFAULT: "#ffffff", sunken: "#f9fafb", raised: "#f3f4f6" },
        accent: { DEFAULT: "#4f46e5", soft: "#eef2ff", ring: "#c7d2fe" },
      },
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;
