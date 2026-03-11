import { createFileRoute } from "@tanstack/react-router";
import type { ReactNode } from "react";

export const Route = createFileRoute("/settings")({ component: Settings });

function Section({ title, description, children }: { title: string; description?: string; children: ReactNode }) {
  return (
    <div className="card card-highlight" style={{ padding: "20px 24px" }}>
      <div style={{ marginBottom: "16px" }}>
        <h3 style={{ margin: 0, fontSize: "15px", fontWeight: 600, color: "var(--fg)" }}>{title}</h3>
        {description && <p style={{ margin: "4px 0 0", fontSize: "13px", color: "var(--fg-3)" }}>{description}</p>}
      </div>
      {children}
    </div>
  );
}

function Row({ label, description, control }: { label: string; description?: string; control: ReactNode }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid var(--border)" }}>
      <div>
        <div style={{ fontSize: "14px", color: "var(--fg)", fontWeight: 500 }}>{label}</div>
        {description && <div style={{ fontSize: "12px", color: "var(--fg-3)", marginTop: "2px" }}>{description}</div>}
      </div>
      {control}
    </div>
  );
}

function Toggle({ on }: { on?: boolean }) {
  return (
    <div style={{
      width: "38px", height: "22px", borderRadius: "99px",
      background: on ? "var(--accent)" : "oklch(1 0 0 / 10%)",
      position: "relative", cursor: "pointer", transition: "background 0.2s",
      border: "1px solid var(--border)",
    }}>
      <div style={{
        width: "16px", height: "16px", borderRadius: "50%",
        background: "white", position: "absolute", top: "2px",
        left: on ? "18px" : "2px", transition: "left 0.2s ease",
      }} />
    </div>
  );
}

function Settings() {
  return (
    <div className="page-enter" style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      <div>
        <h1 style={{ fontSize: "24px", fontWeight: 700, letterSpacing: "-0.02em", color: "var(--fg)", margin: 0 }}>
          Settings
        </h1>
        <p style={{ marginTop: "6px", color: "var(--fg-2)", fontSize: "14px" }}>
          Configure Lune Voice to your workflow.
        </p>
      </div>

      <Section title="Activation" description="How you trigger dictation.">
        <Row label="Hotkey" description="Ctrl+Space to toggle, Escape to cancel" control={
          <span className="badge" style={{ background: "oklch(1 0 0 / 5%)", borderColor: "var(--border)", color: "var(--fg-2)", fontFamily: "monospace" }}>Ctrl+Space</span>
        } />
        <Row label="Activation mode" description="How the hotkey starts and stops" control={
          <select style={{ background: "var(--bg-2)", border: "1px solid var(--border)", borderRadius: "var(--r)", padding: "5px 10px", color: "var(--fg)", fontSize: "13px", outline: "none" }}>
            <option>Toggle</option>
            <option>Hold</option>
          </select>
        } />
      </Section>

      <Section title="ASR Backend" description="Speech recognition engine and model.">
        <Row label="Backend" control={
          <select style={{ background: "var(--bg-2)", border: "1px solid var(--border)", borderRadius: "var(--r)", padding: "5px 10px", color: "var(--fg)", fontSize: "13px", outline: "none" }}>
            <option>Parakeet TDT (recommended)</option>
            <option>Qwen3-ASR-0.6B</option>
            <option>OpenAI Compatible</option>
          </select>
        } />
        <Row label="Use GPU (CUDA)" description="Requires NVIDIA GPU and CUDA toolkit" control={<Toggle on />} />
        <Row label="Word confidence threshold" description="Words below this threshold are flagged for LLM correction" control={
          <span style={{ fontFamily: "monospace", fontSize: "13px", color: "var(--fg-2)", background: "var(--bg-2)", border: "1px solid var(--border)", borderRadius: "var(--r-sm)", padding: "3px 8px" }}>0.65</span>
        } />
      </Section>

      <Section title="LLM Post-Processing" description="AI correction for uncertain words.">
        <Row label="Enable LLM correction" control={<Toggle on />} />
        <Row label="Model" control={
          <select style={{ background: "var(--bg-2)", border: "1px solid var(--border)", borderRadius: "var(--r)", padding: "5px 10px", color: "var(--fg)", fontSize: "13px", outline: "none" }}>
            <option>Qwen2.5-1.5B-Instruct (local)</option>
          </select>
        } />
      </Section>

      <Section title="Text Delivery" description="How transcribed text is inserted.">
        <Row label="Delivery method" control={
          <select style={{ background: "var(--bg-2)", border: "1px solid var(--border)", borderRadius: "var(--r)", padding: "5px 10px", color: "var(--fg)", fontSize: "13px", outline: "none" }}>
            <option>Clipboard paste</option>
            <option>uinput keystrokes</option>
          </select>
        } />
        <Row label="Style preset" control={
          <select style={{ background: "var(--bg-2)", border: "1px solid var(--border)", borderRadius: "var(--r)", padding: "5px 10px", color: "var(--fg)", fontSize: "13px", outline: "none" }}>
            <option>Conversational</option>
            <option>Professional</option>
            <option>Code</option>
          </select>
        } />
      </Section>

      <Section title="VAD" description="Voice activity detection tuning.">
        <Row label="Adaptive VAD" description="Learns your pause rhythm over time" control={<Toggle />} />
        <Row label="Speech threshold" control={
          <span style={{ fontFamily: "monospace", fontSize: "13px", color: "var(--fg-2)", background: "var(--bg-2)", border: "1px solid var(--border)", borderRadius: "var(--r-sm)", padding: "3px 8px" }}>0.5</span>
        } />
      </Section>
    </div>
  );
}
