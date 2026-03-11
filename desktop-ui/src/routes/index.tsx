import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/")({ component: Overview });

const STATS = [
  { label: "TOTAL WORDS", value: "15,262", sub: "Words transcribed",   icon: "⚡", color: "oklch(0.68 0.19 275)" },
  { label: "SPEAKING TIME", value: "1h 27m", sub: "Total duration",    icon: "⏱", color: "oklch(0.68 0.16 200)" },
  { label: "SESSIONS", value: "490", sub: "Total recordings",         icon: "🎙", color: "oklch(0.68 0.16 160)" },
  { label: "AVG PACE", value: "174 wpm", sub: "Words per minute",      icon: "〰", color: "oklch(0.72 0.14 55)"  },
];

const RECENT = [
  { time: "9:01 AM", words: 47, secs: 15, text: "I want you to navigate the website, particularly the blog page, and I want you to review the blog page and come up with improvements to the UI." },
  { time: "8:44 AM", words: 4,  secs: 2,  text: "Okay, execute the plan." },
  { time: "8:22 AM", words: 23, secs: 8,  text: "In order to learn more about each of these products, you can go to the website and check each product page accordingly." },
  { time: "7:55 AM", words: 11, secs: 4,  text: "Please generate a summary of the current session." },
];

function Overview() {
  return (
    <div className="page-enter" style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
      {/* Header */}
      <div>
        <h1 style={{ fontSize: "24px", fontWeight: 700, letterSpacing: "-0.02em", color: "var(--fg)", margin: 0 }}>
          Overview
        </h1>
        <p style={{ marginTop: "6px", color: "var(--fg-2)", fontSize: "14px" }}>
          Your Lune Voice performance snapshot, recent activity, and trends.
        </p>
      </div>

      {/* Stat cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px" }}>
        {STATS.map(s => (
          <div key={s.label} className="card card-highlight" style={{ padding: "18px 20px" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
              <span style={{ fontSize: "11px", fontWeight: 600, letterSpacing: "0.06em", color: "var(--fg-3)" }}>
                {s.label}
              </span>
              <span style={{ fontSize: "16px", opacity: 0.7 }}>{s.icon}</span>
            </div>
            <div className="stat-value" style={{ color: "var(--fg)" }}>{s.value}</div>
            <div style={{ marginTop: "6px", fontSize: "12px", color: "var(--fg-3)" }}>{s.sub}</div>
          </div>
        ))}
      </div>

      {/* Recent activity */}
      <div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px" }}>
          <div>
            <h2 style={{ fontSize: "15px", fontWeight: 600, color: "var(--fg)", margin: 0 }}>Recent Activity</h2>
            <p style={{ margin: "4px 0 0", fontSize: "13px", color: "var(--fg-3)" }}>
              Your latest transcriptions from this device.
            </p>
          </div>
          <a href="/history" style={{ fontSize: "13px", color: "var(--accent-light)", textDecoration: "none", opacity: 0.8 }}>
            View Full History →
          </a>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          {RECENT.map((r, i) => (
            <div key={i} className="card" style={{ padding: "16px 20px", cursor: "pointer" }}>
              <p style={{ margin: "0 0 10px", fontSize: "14px", color: "var(--fg)", lineHeight: 1.5 }}>
                {r.text}
              </p>
              <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                <span className="badge" style={{ background: "oklch(1 0 0 / 4%)", borderColor: "var(--border)", color: "var(--fg-3)" }}>
                  {r.time}
                </span>
                <span style={{ fontSize: "12px", color: "var(--fg-3)" }}>{r.words} words</span>
                <span style={{ fontSize: "12px", color: "var(--fg-3)" }}>{r.secs}s</span>
                <button style={{
                  marginLeft: "auto",
                  fontSize: "12px", color: "var(--fg-3)",
                  background: "none", border: "1px solid var(--border)",
                  borderRadius: "var(--r-sm)", padding: "2px 8px",
                  cursor: "pointer", transition: "color 0.15s, border-color 0.15s"
                }}>
                  + Add to Dictionary
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
