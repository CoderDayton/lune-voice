# Design: consolidate model management (Models = library, Transcription = chooser)

## Problem

Two Settings surfaces both render Whisper download/status UI:
- **Transcription** tab: engine cards (Whisper / Cloud) + Whisper size cards with inline selection *and* Download.
- **Models** tab: a generic "models on your device" list that *also* shows the same Whisper model with Download.

They share `useModelStatus` (download/delete/status by `local_name`), so the *logic* doesn't drift — but the duplicated *visual* download/status UI is confusing and won't scale to the styling LLM + VAD + future models.

## Decision

- **Models tab = the single model library / manager.** Shows the full installable set always (all 3 Whisper sizes + styling LLM), grouped by role, with status, size, download/delete, an **In use** badge on models the current config uses, and a **total on-disk usage** line.
- **Transcription tab = the chooser.** Engine cards + a compact Whisper **size selector** with a per-size **readiness badge**. Picking a not-installed size sets it active and shows **one** inline Download (shared flow + progress) plus a quiet **"Manage in Models →"** link. The redundant full download cards are removed.
- **VAD model is bundled with Whisper**, not shown as its own row: downloading any Whisper size also fetches the ~1 MB silero VAD; it never appears in the library.

## Backend changes (`lune-persist` + tauri)

1. `ModelSpec`: add a `role` (`SpeechToText` | `Styling`); Whisper + VAD = SpeechToText, style LLM = Styling.
2. `ModelStatus`: add `in_use: bool`, `role` (serialized), and `size_bytes: Option<u64>` (actual on-disk dir size when installed; `None` otherwise).
3. `check_model_status`: return **all** specs **except the VAD** (bundled/hidden), each with `ready`, `in_use` (the former `spec_needed` predicate), `role`, `size_bytes`. No longer filters out non-selected models.
4. `download_model_cmd`: when the requested `local_name` is a Whisper size, also download the VAD model if absent (so Whisper always has its VAD). VAD stays out of the listed library.
5. Mirror the new `ModelStatus` fields in `dto.rs`; regenerate ts-rs bindings.

## Frontend changes (`desktop-ui/src`)

- `useModelStatus.ModelStatus`: add `in_use`, `role`, `size_bytes`.
- **`models-section.tsx`**: group rows by `role` (Speech-to-text, Styling); per row name · size · status (Installed ✓ / Not installed / Downloading + progress) · Download/Delete; **In use** badge when `in_use`; total disk usage = Σ `size_bytes` of installed. Drop config-based hiding.
- **`transcription-section.tsx` / `asr-engine-cards.tsx`**: Whisper sizes become a selector with a readiness badge per size (from `check_model_status` `ready`); not-installed active size → inline `download()` + progress + "Manage in Models →" link; remove the full download cards.
- `settings-data.ts`: ensure a size label exists for every listed model (Whisper sizes already have them; add the styling LLM's).

## Out of scope / non-goals

- No change to `useModelStatus` download/delete wiring beyond the new fields (single source of truth preserved).
- No new model formats or engines; the design simply scales by adding `ModelSpec`s.

## Verification

`cargo test -p lune-ui` (ts-rs regen, no unintended drift) · `cargo check -p lune-ui` · `cargo test -p lune-persist --lib` · `cd desktop-ui && bunx @biomejs/biome check src/ && bun run test`. Manual: Models tab lists all sizes + LLM with correct badges/disk; Transcription size pick shows readiness + inline download; downloading a size also fetches VAD; deleting a non-active size works.
