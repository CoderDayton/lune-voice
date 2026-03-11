import { Pill } from "../../../components/ui/pill";

const authProof = [
  { label: "People onboarded", value: "400K+" },
  { label: "Words corrected", value: "50M+" },
  { label: "Device-first privacy", value: "On" },
];

export function AuthHeroPanel() {
  return (
    <section className="auth-hero-panel">
      <div className="auth-hero-copy">
        <Pill className="auth-hero-pill">Private desktop dictation</Pill>
        <h2 className="auth-headline">
          Speak naturally. Keep every correction local and every shortcut close.
        </h2>
        <p className="body-copy-soft auth-hero-text">
          Lune Voice is built for people who want faster writing, cleaner transcripts, and a workspace that feels calm instead of technical.
        </p>
      </div>

      <div className="auth-proof-grid">
        {authProof.map((item) => (
          <div key={item.label} className="auth-proof-card">
            <div className="stat-label">{item.label}</div>
            <div className="auth-proof-value">{item.value}</div>
          </div>
        ))}
      </div>

      <div className="auth-orbit-stage" aria-hidden="true">
        <div className="auth-orbit-ring" />
        <div className="auth-orbit-arrow" />
        <div className="auth-cube-glow" />
        <div className="auth-cube">
          <span />
          <span />
          <span />
        </div>
      </div>
    </section>
  );
}
