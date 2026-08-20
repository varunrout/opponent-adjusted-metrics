import type { Config } from "tailwindcss";

// Design tokens centralised here per docs/dashboard_design_spec.md.
// Never hardcode these hex values / font stacks inline in components —
// consume them via Tailwind utility classes (bg-bg, text-text2, font-ui, ...)
// or the CSS variables defined once in app/globals.css.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        surface: "var(--surface)",
        card: "var(--card)",
        "card-hi": "var(--card-hi)",
        border: "var(--border)",
        text: "var(--text)",
        text2: "var(--text2)",
        muted: "var(--muted)",
        teal: "var(--teal)",
        violet: "var(--violet)",
        amber: "var(--amber)",
        green: "var(--green)",
        red: "var(--red)",
        "home-team": "var(--home-team)",
        "away-team": "var(--away-team)",
      },
      fontFamily: {
        ui: ["var(--font-montserrat)", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "Helvetica", "Arial", "sans-serif"],
        data: ["var(--font-inconsolata)", "SFMono-Regular", "Consolas", "Liberation Mono", "Menlo", "monospace"],
      },
      borderRadius: {
        DEFAULT: "10px",
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "100% 50%" },
          "100%": { backgroundPosition: "0 50%" },
        },
      },
      animation: {
        shimmer: "shimmer 1.4s ease infinite",
      },
    },
  },
  plugins: [],
};

export default config;
