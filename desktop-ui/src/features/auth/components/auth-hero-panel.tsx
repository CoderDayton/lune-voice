import { Pill } from "../../../components/ui/pill";

const authProof = [
  { label: "Audio processing", value: "On-device" },
  { label: "Cloud dependency", value: "None" },
  { label: "Data shared", value: "Zero" },
];

export function AuthHeroPanel() {
  return (
    <section className="card bg-base-200/80 border border-base-300/60 surface-hero p-4 sm:p-6 flex min-h-[24rem] lg:min-h-[38rem] flex-col justify-between overflow-hidden">
      <div className="stack-copy max-w-full sm:max-w-[34rem]">
        <Pill className="w-fit">Private desktop dictation</Pill>
        <h2 className="auth-headline">
          Speak naturally. Keep every correction local and every shortcut close.
        </h2>
        <p className="body-copy-soft max-w-full sm:max-w-[31rem]">
          Lune Voice is built for people who want faster writing, cleaner transcripts, and a workspace that feels calm instead of technical.
        </p>
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        {authProof.map((item) => (
          <div key={item.label} className="card bg-base-300/50 border border-base-300/30 p-4 stack-copy-tight">
            <div className="label-text-custom">{item.label}</div>
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
