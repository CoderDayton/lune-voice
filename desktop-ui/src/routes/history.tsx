import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";

export const Route = createFileRoute("/history")({ component: History });

const SESSIONS = [
  { date: "Today", items: [
    { time: "9:01 AM", text: "I want you to navigate the website, particularly the blog page, and I want you to review the blog page and come up with improvements to the UI for the blog page as well as the blog ID pages. It needs to be very modern and unique.", words: 47, secs: 15 },
    { time: "8:44 AM", text: "Okay, execute the plan.", words: 4, secs: 2 },
    { time: "8:22 AM", text: "In order to learn more about each of these products, you can go to the website and check each product page accordingly to learn everything that you know, need to know.", words: 32, secs: 11 },
    { time: "7:55 AM", text: "Please generate a summary of the current session and export it to markdown.", words: 14, secs: 5 },
  ]},
  { date: "Yesterday", items: [
    { time: "11:32 PM", text: "Fix the authentication middleware to handle expired tokens gracefully and return proper error codes.", words: 16, secs: 6 },
    { time: "3:14 PM", text: "Schedule a meeting with the design team for next Monday at two PM.", words: 14, secs: 5 },
    { time: "10:07 AM", text: "Open the project dashboard and filter by active tasks assigned to me.", words: 13, secs: 5 },
  ]},
  { date: "Mar 9", items: [
    { time: "8:30 PM", text: "Wow, this is kind of amazing.", words: 6, secs: 2 },
    { time: "2:45 PM", text: "Transcribe anything locally without sending audio to the cloud.", words: 11, secs: 4 },
  ]},
];

function History() {
  const [search, setSearch] = useState("");

  const filtered = SESSIONS.map(g => ({
    ...g,
    items: g.items.filter(i =>
      !search || i.text.toLowerCase().includes(search.toLowerCase())
    )
  })).filter(g => g.items.length > 0);

  const total = SESSIONS.reduce((n, g) => n + g.items.length, 0);

  return (
    <div className="page-enter" style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
        <div>
          <h1 style={{ fontSize: "24px", fontWeight: 700, letterSpacing: "-0.02em", color: "var(--fg)", margin: 0 }}>
            History
          </h1>
          <p style={{ marginTop: "6px", color: "var(--fg-2)", fontSize: "14px" }}>
            All your transcriptions from this device.
          </p>
        </div>
        {/* Search */}
        <div style={{ position: "relative" }}>
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search..."
            style={{
              background: "var(--bg-2)",
              border: "1px solid var(--border)",
              borderRadius: "var(--r)",
              padding: "7px 12px 7px 32px",
              color: "var(--fg)",
              fontSize: "13px",
              outline: "none",
              width: "200px",
              transition: "border-color 0.15s",
            }}
          />
          <svg style={{ position: "absolute", left: "10px", top: "50%", transform: "translateY(-50%)", color: "var(--fg-3)" }} width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
            <circle cx="6" cy="6" r="4.5" /><line x1="9.5" y1="9.5" x2="13" y2="13" />
          </svg>
        </div>
      </div>

      {/* List */}
      <div className="card" style={{ overflow: "hidden", padding: 0 }}>
        {filtered.map((group, gi) => (
          <div key={gi}>
            <div style={{ padding: "10px 20px", borderBottom: "1px solid var(--border)", background: "oklch(1 0 0 / 1.5%)" }}>
              <span style={{ fontSize: "11px", fontWeight: 600, letterSpacing: "0.07em", color: "var(--fg-3)", textTransform: "uppercase" }}>
                {group.date}
              </span>
            </div>
            {group.items.map((item, ii) => (
              <div
                key={ii}
                style={{
                  padding: "14px 20px",
                  borderBottom: gi === filtered.length - 1 && ii === group.items.length - 1 ? "none" : "1px solid var(--border)",
                  display: "flex",
                  gap: "16px",
                  alignItems: "flex-start",
                  cursor: "pointer",
                  transition: "background 0.15s",
                }}
                onMouseEnter={e => (e.currentTarget.style.background = "oklch(1 0 0 / 2%)")}
                onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
              >
                <div style={{ position: "relative", marginTop: "5px", flexShrink: 0 }}>
                  <div style={{ width: "7px", height: "7px", borderRadius: "50%", background: "var(--accent-light)", opacity: 0.7 }} />
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <p style={{ margin: "0 0 6px", fontSize: "14px", color: "var(--fg)", lineHeight: 1.5 }}>
                    {item.text}
                  </p>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <span style={{ fontSize: "12px", color: "var(--fg-3)" }}>{item.time}</span>
                    <span style={{ color: "var(--fg-3)", opacity: 0.4 }}>·</span>
                    <span style={{ fontSize: "12px", color: "var(--fg-3)" }}>{item.words} words</span>
                    <span style={{ color: "var(--fg-3)", opacity: 0.4 }}>·</span>
                    <span style={{ fontSize: "12px", color: "var(--fg-3)" }}>{item.secs}s audio</span>
                    <span style={{ color: "var(--fg-3)", opacity: 0.4 }}>·</span>
                    <span style={{ fontSize: "12px", color: "var(--fg-3)" }}>Parakeet TDT</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ))}

        {/* Footer */}
        <div style={{ padding: "10px 20px", borderTop: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{ fontSize: "11px", fontWeight: 600, letterSpacing: "0.06em", color: "var(--fg-3)", textTransform: "uppercase" }}>
            {total} TRANSCRIPTIONS
          </span>
          <button style={{ fontSize: "11px", fontWeight: 600, letterSpacing: "0.06em", color: "var(--fg-3)", background: "none", border: "none", cursor: "pointer", textTransform: "uppercase" }}>
            CLEAR ALL
          </button>
        </div>
      </div>
    </div>
  );
}
