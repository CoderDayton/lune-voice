# Modular correction pipeline — remaining framework (deferred)

Picks up the post-ASR correction refactor (Option A — text-layer only; NO ASR-decoder
change, NO N-best, NO fine-tune). The high-value, low-risk steps are **done and
committed** on branch `voice-casing-vad-fixes`; this doc captures the **remaining**
framework work plus a design finding that changes how it should be built.

## Already shipped (committed)
- `e89655d` — removed the dead `Arc<NGramModel>` from the corrector; renamed
  `HeuristicCorrector` → `LexicalGate` (it does no spelling repair — it applies the
  user dictionary + brand canonical casing and *flags* uncertain words).
- `55a4480` — `lexicon::WordInfo` now carries `p_cap` (from `caprate`); the lexical
  gate's casing decision routes through `p_cap` (falls back to dictionary membership
  only when the cap-rate table doesn't know the word).
- `2407638` — the selective-correction gate is a first-class tested function
  `crate::gate::decide(GateInputs) -> GateOutcome` (`crates/lune-app/src/gate.rs`),
  replacing the inline `if`s in `process_and_deliver`. Arithmetic is byte-identical.

## Design finding (why the rest is bigger than the original plan assumed)
The pipeline's stages are **not uniform**:
- `punctuation::defuse_inconsistent`, `decase_common_words`, `capitalize_sentence_starts`
  are **pure** `text → text` — trivial to reify as passes.
- The **`LexicalGate`** (`correct_with_risk_threshold`) and **`punctuation::normalize`**
  each produce **gate signals** *and* text: `all_resolved` / `word_risk_positions` /
  `unresolved_words` (heuristic) and `needs_llm` (normalize).

So a clean "**everything is a `CorrectionPass`**" framework requires the `Transcript`
IR to **carry the gate signals**, so the signal-producing stages are passes too and the
gate reads the signals off the transcript. That is the substantive remaining change.

## Target design (remaining)

### IR + trait (in `lune-core`, new `crates/lune-core/src/correction.rs`)
Plain-data types only (no lune-dict/lune-llm deps), registered via `pub mod correction;`:
```rust
pub enum PassId { Dictionary, BrandCasing, Defusion, PCapCasing, Punctuation, Llm, Other }
pub enum PassCost { Cheap, Expensive }
pub struct Edit { pub by: PassId, pub before: String, pub after: String, pub rationale: &'static str }

/// Gate signals produced by the heuristic + punctuation passes; read by `gate::decide`.
#[derive(Default)] pub struct GateSignals {
    pub all_resolved: bool,
    pub needs_punct: bool,
    pub word_risk_positions: Vec<(usize, f32)>,
    pub unresolved: Vec<(String, usize)>,
}

#[derive(Default)] pub struct Transcript {
    text: String,
    provenance: Vec<Edit>,
    pub signals: GateSignals,
    pub nbest: Vec<String>,        // empty seam — future GER
    pub phonetic: Option<String>,  // empty seam — future G2P
}
impl Transcript {
    pub fn new(text: impl Into<String>) -> Self
    pub fn text(&self) -> &str
    pub fn assemble(&self) -> &str            // == text(); the testable round-trip
    pub fn provenance(&self) -> &[Edit]
    /// The wrap seam for pure-text passes: run an existing fn, record the edit.
    pub fn rewrite(&mut self, by: PassId, rationale: &'static str, f: impl FnOnce(&str) -> String)
}

pub trait CorrectionPass {
    fn id(&self) -> PassId;
    fn cost(&self) -> PassCost;
    fn apply(&self, t: &mut Transcript);   // NO &Lexicon param — see crate-layering note
}
```
**Crate-layering note:** the trait must NOT take `&Lexicon` (lexicon lives in lune-dict,
*above* lune-core). Concrete passes that need the lexicon are defined **in lune-dict**
and hold their own session context (dict replacements, risk scores, lang, threshold);
they read the bundled static lexicon directly, exactly as the existing fns do.

### Concrete passes
- **lune-dict** (new `crates/lune-dict/src/pass.rs`):
  - `LexicalGatePass` — holds `(lang, dict_replacements, word_risk_scores, flag_threshold)`;
    wraps `LexicalGate::correct_with_risk_threshold`; sets `t.text` + populates
    `t.signals.{all_resolved, word_risk_positions, unresolved}`.
  - `DefusionPass` → `t.rewrite(Defusion, _, defuse_inconsistent)`.
  - `PCapCasingPass` → `decase_common_words` then `capitalize_sentence_starts` (after the LLM
    today; check ordering — see Step 8).
  - `PunctuationPass` → runs `normalize`, sets `t.text` + `t.signals.needs_punct`.
- **lune-app** (LLM is Expensive, lives where the orchestration is):
  - `LlmPass{cost=Expensive}` — builds `PostProcessInput{ raw_text: t.assemble().into(), .. }`
    (== today's `text_after_heuristic`), calls `post_processor.transform`, then
    `capitalize_sentence_starts`. **HARD CONSTRAINT: full utterance in, `validate_and_merge`
    stays internal to `CorrectionPipeline` (pipeline.rs:140) — do not touch it.**

### Orchestrator swap (`process_and_deliver`, `crates/lune-app/src/session.rs`)
Replace the inline sequence with, in **today's exact order**:
`build Transcript(text_after_commands)` → `LexicalGatePass` → `DefusionPass` →
`PunctuationPass` → `PCapCasingPass` → `gate::decide(transcript.signals + config + word_risk_scores)`
→ if `needs_llm` run `LlmPass` → finalize/deliver. `word_risk_scores` (ASR per-word, for the
min-confidence band) stays orchestrator-local and is passed to `gate::decide`.

## Remaining steps (each its own commit; behavior-preserving)
- **4** — add the IR + trait in lune-core (dead code). Verify: `cargo test -p lune-core` +
  an `assemble(new(s)) == s` round-trip test (whitespace single-space join, matching
  `corrector.rs` assembly).
- **6** — reify the cheap stages as passes in lune-dict (wrap existing fns). Verify per-pass:
  `pass.apply(&mut tx); tx.assemble() == original_fn(input)` over existing fixtures.
  Orchestrator untouched → all current tests still pass.
- **7** — reify the `LlmPass` adapter in lune-app. Verify: `cargo test -p lune-llm`; e2e Echo
  mock asserts full-utterance passthrough; `validate_and_merge` unchanged.
- **8** — swap `process_and_deliver` to drive the passes + gate. **Validate with a
  zero-output-diff** across `benchmarks/{pipeline_real,bench_domain,llm_quality}.rs` vs a
  pre-swap run (the only acceptable diff is none). `cargo test -p lune-app` green.

## Validation discipline
Every commit: `cargo fmt --check`, `cargo clippy --workspace --all-targets`,
`cargo test -p lune-core -p lune-dict -p lune-llm -p lune-app`, `cargo build -p lune-llm`
(no-default lune-dict edge), `cargo check -p lune-ui` (tauri). The repo `lefthook`
pre-commit runs the full CI. The Step-8 benchmark diff is the behavior-identity proof.

## Deferred SOTA path (seams, implement none here)
- **N-best / GER:** `Transcript.nbest` stays empty (needs an ASR beam-decode change;
  `WordConfidence` carries no alternatives). Later: an `NBestLlmPass`.
- **Phonetic G2P (anti-over-correction, arXiv:2505.17410):** `Transcript.phonetic` empty;
  later a `PhoneticGuardPass` + wiring the (currently calibration-only) homophone resolver.
- `provenance: Vec<Edit>` + `PassId` are the observability spine those passes plug into
  with no orchestrator re-plumbing — and turn the case/spacing benchmark metrics into
  per-pass scorecards.
