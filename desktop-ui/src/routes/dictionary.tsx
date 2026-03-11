import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";

export const Route = createFileRoute("/dictionary")({ component: Dictionary });

const ENTRIES = [
  { phrase: "github", replacement: "GitHub", priority: 5 },
  { phrase: "kubernetes", replacement: "Kubernetes", priority: 8 },
  { phrase: "lune voice", replacement: "Lune Voice", priority: 10 },
  { phrase: "rust lang", replacement: "Rust", priority: 5 },
  { phrase: "api", replacement: "API", priority: 6 },
  { phrase: "llm", replacement: "LLM", priority: 7 },
];

function Dictionary() {
  const [entries, setEntries] = useState(ENTRIES);
  const [phrase, setPhrase] = useState("");
  const [replacement, setReplacement] = useState("");

  const add = () => {
    if (!phrase.trim() || !replacement.trim()) return;
    setEntries(e => [...e, { phrase: phrase.trim(), replacement: replacement.trim(), priority: 5 }]);
    setPhrase(""); setReplacement("");
  };

  return (
    <div className="page-enter" style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
        <div>
          <h1 style={{ fontSize: "24px", fontWeight: 700, letterSpacing: "-0.02em", color: "var(--fg)", margin: 0 }}>
            Dictionary
          </h1>
          <p style={{ marginTop: "6px", color: "var(--fg-2)", fontSize: "14px" }}>
            Custom corrections applied after every transcription.
          </p>
        </div>
      </div>

      {/* Add entry */}
      <div className="card" style={{ padding: "20px", display: "flex", gap: "10px", alignItems: "flex-end" }}>
        <div style={{ flex: 1 }}>
          <label style={{ display: "block", fontSize: "12px", color: "var(--fg-3)", marginBottom: "6px", fontWeight: 500 }}>Spoken phrase</label>
          <input value={phrase} onChange={e => setPhrase(e.target.value)} placeholder="e.g. github"
            style={{ width: "100%", background: "var(--bg-2)", border: "1px solid var(--border)", borderRadius: "var(--r)", padding: "8px 12px", color: "var(--fg)", fontSize: "14px", outline: "none" }} />
        </div>
        <div style={{ color: "var(--fg-3)", paddingBottom: "10px" }}>→</div>
        <div style={{ flex: 1 }}>
          <label style={{ display: "block", fontSize: "12px", color: "var(--fg-3)", marginBottom: "6px", fontWeight: 500 }}>Replacement</label>
          <input value={replacement} onChange={e => setReplacement(e.target.value)} placeholder="e.g. GitHub"
            style={{ width: "100%", background: "var(--bg-2)", border: "1px solid var(--border)", borderRadius: "var(--r)", padding: "8px 12px", color: "var(--fg)", fontSize: "14px", outline: "none" }} />
        </div>
        <button onClick={add} style={{
          background: "var(--accent)", color: "white", border: "none",
          borderRadius: "var(--r)", padding: "9px 18px", fontSize: "14px", fontWeight: 600,
          cursor: "pointer", flexShrink: 0, transition: "opacity 0.15s"
        }}>
          Add
        </button>
      </div>

      {/* Entries table */}
      <div className="card" style={{ overflow: "hidden", padding: 0 }}>
        <div style={{ padding: "12px 20px", borderBottom: "1px solid var(--border)", display: "grid", gridTemplateColumns: "1fr 1fr auto", gap: "12px" }}>
          {["SPOKEN PHRASE", "REPLACEMENT", ""].map(h => (
            <span key={h} style={{ fontSize: "11px", fontWeight: 600, letterSpacing: "0.07em", color: "var(--fg-3)", textTransform: "uppercase" }}>{h}</span>
          ))}
        </div>
        {entries.length === 0 && (
          <div style={{ padding: "48px", textAlign: "center", color: "var(--fg-3)", fontSize: "14px" }}>
            No entries yet. Add your first correction above.
          </div>
        )}
        {entries.map((e, i) => (
          <div key={i} style={{
            padding: "12px 20px",
            borderBottom: i < entries.length - 1 ? "1px solid var(--border)" : "none",
            display: "grid", gridTemplateColumns: "1fr 1fr auto", gap: "12px", alignItems: "center",
            transition: "background 0.15s",
          }}
            onMouseEnter={ev => (ev.currentTarget.style.background = "oklch(1 0 0 / 2%)")}
            onMouseLeave={ev => (ev.currentTarget.style.background = "transparent")}
          >
            <span style={{ fontFamily: "monospace", fontSize: "13px", color: "var(--fg-2)" }}>{e.phrase}</span>
            <span style={{ fontFamily: "monospace", fontSize: "13px", color: "var(--fg)", fontWeight: 500 }}>{e.replacement}</span>
            <button onClick={() => setEntries(prev => prev.filter((_, j) => j !== i))}
              style={{ background: "none", border: "none", color: "var(--fg-3)", cursor: "pointer", fontSize: "16px", padding: "0 4px", opacity: 0.6, transition: "opacity 0.15s" }}
              onMouseEnter={ev => (ev.currentTarget.style.opacity = "1")}
              onMouseLeave={ev => (ev.currentTarget.style.opacity = "0.6")}
            >×</button>
          </div>
        ))}
      </div>
    </div>
  );
}
