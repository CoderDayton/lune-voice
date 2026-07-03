# Fine-Tuning Dataset Spec — Lune Voice Transform Model

Goal: fine-tune a small local model that takes a short (usually spoken/ASR)
transcript plus a **style/transform instruction** and rewrites the text
accordingly — from conservative restyles (casual, formal) to arbitrary
user-authored transforms (e.g. "Enhance Prompt"). It must **follow the
instruction as a transformation of the input**, never *execute* it (asked to
"write a function", it enhances the prompt; it does not emit code).

This document has two parts:

- **Part I — Overview:** the three datasets that feed one multi-task training
  run, the schema, licensing, blend, and build order.
- **Part II — Style Matrix (dataset #2) in full:** the turnkey generation spec —
  verbatim teacher prompts, style axis, column schemas, QA rules, worked rows.

It is a data spec, not training code.

---
---

# Part I — Overview: the three-dataset blend

## 0. Target & deploy constraints (drive every format decision)

| Item | Decision |
|---|---|
| Base model | **`meta-llama/Llama-3.2-1B`** (base, not `-Instruct`) — clean canvas, we teach our own behavior, no assistant reflexes to suppress |
| Adapter | Start LoRA `r=16–32`, all linear modules; escalate to high-rank/full-FT only if eval plateaus |
| Train context | **4096 tokens** (matches runtime `N_CTX=4096` in `llama_cpp.rs`) |
| Deploy | merge → `convert_hf_to_gguf.py` → quantize **Q5_K_M** (or Q8_0) → drop in `~/.local/share/lune-voice/models/` → eval with `cgs bench --env TYPE=styling` |
| **Eval target** | the **quantized GGUF**, never the fp16 checkpoint — a 1B loses gains under quant; this is the #1 silent failure |

### Commercial licensing ground rules (Lune Voice ships commercially)

Every input to the model must be commercially clean:

- **Datasets:** Apache-2.0 / MIT / CC-BY / CC-BY-SA / public domain only. **No
  CC-BY-NC** (rules out CoEdIT, No Robots, Alpaca, LIMA).
- **Teacher model for the synthetic set (#2):** must permit using its outputs
  to train another commercial model. **Use `Qwen2.5-32B-Instruct` (Apache-2.0)
  or `Llama-3.3-70B-Instruct` (Llama 3.1+ license explicitly allows using
  outputs to improve other models).** **Do NOT distill GPT-4o / Claude** — their
  ToS forbids using outputs to build competing models.
- CC-BY-SA sources (Dolly, Wikipedia): keep an attribution/provenance file.
  Trained weights are not generally treated as a share-alike derivative, and
  Databricks clears Dolly for commercial use, but retain provenance.

---

## 1. Unified training schema (all three normalize to this)

Every row from every dataset is normalized to one intermediate JSON record:

```json
{
  "instruction": "Rewrite the text in a professional tone. Keep the meaning; do not add facts.",
  "input": "so yeah the deploy's basically done i just gotta double check the numbers",
  "output": "The deployment is essentially complete. I will double-check the figures.",
  "source": "style_matrix",      // style_matrix | dolly | oasst | gutenberg | wiki | edit_synth | negatives
  "meta": { "style": "professional", "preserves_wording": true }
}
```

Then rendered into the **Llama-3 chat template** (the Llama-3.2 tokenizer ships
these special tokens even on the base model, so llama.cpp's `llama3` template
works at deploy time). Instruction → `system`, input → `user`, output →
`assistant`:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>
```

Loss is masked to the **assistant span only**. There is no `<think>` block —
starting from base lets us drop the entire thinking-suppression hack currently
in `llama_cpp.rs`.

> Integration note: the app's prompt builder (`build_styling_prompt` /
> `build_chatml`) must be updated to emit **exactly** this format, or you get
> train/inference skew. The style rules move into `system`, the transcript into
> `user`. (The `(system, user)` content split is already implemented in
> `build_styling_prompt`; only the token rendering in `build_chatml` — ChatML →
> Llama-3 — remains to swap when the fine-tune ships.)

---

## 2. Dataset 1 — Instruction-Following & Editing (real, from HuggingFace)

**Purpose:** teach the model to *listen to an instruction* and rewrite/edit/
summarize/restructure — the general capability under the styles. This is where
"follow the directive, apply it to the text" is learned.

**Use — commercial-safe, human-authored (no model-generation taint):**

| Dataset | HF id | License | Size | Use |
|---|---|---|---|---|
| Dolly 15k | `databricks/databricks-dolly-15k` | CC-BY-SA-3.0 (commercial ✅) | 15k | Keep `summarization`, `creative_writing`, `brainstorming`, `information_extraction`, `general_qa`→rewrite. **Drop** `closed_qa`/`open_qa` factual (that's the "answer, don't transform" behavior we must avoid). |
| OpenAssistant | `OpenAssistant/oasst1` | Apache-2.0 ✅ | ~88k msgs | English, `role=prompter → assistant` first turns only; filter to rewrite/summarize/write-in-style/edit intents. Skip multi-turn chat, code, refusals. |

**Dropped (non-commercial):** `grammarly/coedit`, `HuggingFaceH4/no_robots`,
Alpaca, LIMA — all CC-BY-NC.

**CoEdIT replacement (editing pairs):** since no permissive instruction-*editing*
corpus is clean, **synthesize** it (folds into the #2 pipeline): take permissive
source sentences (#3), have the OSS teacher produce `{instruction: "fix grammar
/ simplify / make concise", input, output}`. ~5k rows is plenty.

**Preparation:** map each record into the unified schema; the `instruction` is
the Dolly/OASST prompt, `input` the provided context (may be empty), `output`
the human response. Deduplicate; cap any single category so factual-QA never
dominates.

---

## 3. Dataset 2 — Synthetic "Style Matrix" (you build this — the core, ~55%)

**Purpose:** teach the *concept of stylistic shifting* across many styles so it
generalizes to unseen, user-authored styles (custom styles are a shipped
feature). This is the largest and most important set.

> **Full turnkey spec is Part II below** — teacher prompts (verbatim), style
> axis, column schemas, QA rules, worked examples. The summary here is the
> overview.

### 3.1 What to build

A matrix of **base text × style → styled output**. For each base text, generate
several stylistic variations with the permissive teacher.

- **Base texts (~8–12k), matched to real input distribution:**
  - Your **1,139 real transcripts** (`data/finetune/base_transcripts.jsonl`,
    extracted from `correction_pairs`) — highest value, real spoken/ASR
    disfluency.
  - **Synthesized spoken-style utterances** from the teacher to scale up: short,
    lowercase, disfluent, filler words ("um", "like"), no punctuation — must
    look like ASR output, **not** clean prose.
  - A minority of clean-prose base texts from #3 for the longer/writing styles.
- **Style axis (~8–12 styles):** the 5 shipped ones with their exact production
  wording (casual, professional, formal, technical, writing) **plus** a rotating
  long tail of synthetic instructions (academic, punchy, narrative, concise,
  friendly, "enhance this prompt", "make it a bulleted summary", …). Diversity
  here is what makes custom styles work.
- **Result:** ~8–12k base × ~4–5 styles ≈ **30–45k pairs**.

### 3.2 Teacher generation (per base text × style)

Prompt the teacher (Qwen2.5-32B / Llama-3.3-70B) to produce the `output`,
preserving wording for restyle styles and transforming (never executing) for
transform styles, then QA-filter. The exact prompts, params, and filter are in
Part II §D and §H.

### 3.3 Transform-not-execute negatives (~10% of #2, ~3–5k) — critical

The measured failure is the model *executing* the input. Explicitly teach the
boundary:

- `input` = a task-shaped utterance ("write a function that sorts a list",
  "make me a landing page", "explain photosynthesis").
- `instruction` = a transform ("Enhance this prompt: make it more specific and
  actionable", or "Rewrite this in a professional tone").
- `output` = the **transformed instruction/text**, NOT the executed result
  (an improved *prompt*, not code; a restyled *request*, not an essay).

Without these, a base model has no prior for which reading you mean. Full spec
(seed generation + gold anchors) in Part II §F.

---

## 4. Dataset 3 — Vocabulary & Syntactic Variety (real, from HuggingFace, ~15%)

**Purpose:** richer vocabulary and varied sentence structure for the elevated
styles (formal, writing, academic), so "bad" can become "detrimental / flawed"
*when the style calls for it*.

**Use — commercial-safe prose sources:**

| Dataset | HF id | License | Use |
|---|---|---|---|
| Project Gutenberg (English) | `sedthh/gutenberg_english` | Public domain* | 48k books; source of literary sentences/paragraphs |
| Wikipedia | `wikimedia/wikipedia` (e.g. `20231101.en`) | CC-BY-SA ✅ | encyclopedic register, clean syntax |

\* Verify per-book copyright via the `METADATA` column; the bulk is public
domain, but the card notes some copyrighted entries slipped in.

**These are raw prose, not instruction pairs.** Do **not** dump them into SFT as
language-modeling text (that destabilizes a 1B and conflicts with the task).
Convert them into pairs two ways:

1. **Feed #2:** use their sentences as high-quality base texts / target
   exemplars for the Style Matrix (this is the main use).
2. **Register-elevation pairs (~5–8k):** teacher *degrades* a rich sentence into
   plain/spoken form → store `{instruction: "elevate the vocabulary and vary the
   sentence structure", input: degraded, output: original}`. This teaches
   upgrading register **without** free hallucination, because the rich target is
   real human prose.

---

## 5. Blend, sizing, and holdout

Target ~**50–80k** total for a 1B multi-task run (quality/distribution ≫ raw
count):

| Set | Share | Rows |
|---|---|---|
| #2 Style Matrix (incl. ~10% transform-not-execute negatives) | ~55% | 30–45k |
| #1 Instruction/Editing (Dolly + OASST + synthetic edits) | ~30% | 15–25k |
| #3 Vocab/Syntactic (register-elevation pairs) | ~15% | 5–10k |

Rules:
- **Deduplicate** across all sources (exact + near-dupe on input).
- **Cap factual-QA** so "answer the question" never dominates.
- **Match input distribution:** the majority of #2 base texts must look like real
  ASR transcripts, not clean paragraphs.
- **Holdout eval set (~1.5–2k)**, style-stratified, **never trained on**. Include
  real transcripts + the task-shaped negatives. Score with
  `cgs bench --env TYPE=styling` on the quantized GGUF.

---

## 6. Build order (checklist)

1. Export real transcripts → base-text pool (`data/finetune/base_transcripts.jsonl`,
   already built by `scripts/finetune/build_transcript_pool.py`).
2. Stand up the permissive teacher (Qwen2.5-32B-Instruct / Llama-3.3-70B-Instruct).
3. Generate #2 Style Matrix + transform-not-execute negatives; QA-filter with the
   benchmark's red-flag logic (Part II).
4. Pull & filter #1 (Dolly + OASST) into the unified schema.
5. Build #3 register-elevation pairs from Gutenberg/Wikipedia via the teacher.
6. Merge → dedupe → blend to the ratios above → carve the holdout.
7. Render to the Llama-3 chat template; mask loss to the assistant span.
8. Train LoRA (r=16–32) at 4096 ctx → merge → GGUF → quantize → eval on the
   holdout with the styling benchmark. Iterate on data, not hyperparameters.

---
---

# Part II — Dataset #2 in full: Style Matrix generation

Turnkey spec for building the core dataset: `base text × style → styled output`,
generated by a permissive teacher model (~55% of the blend).

> **Two prompts, do not confuse them:**
> - **Teacher labeler prompt** (§D) — the elaborate prompt given to the *big*
>   teacher model *offline* to GENERATE each `output`. Never ships.
> - **Training instruction** (§G) — the terse system prompt baked into each
>   row's `instruction` field, which the *small* fine-tuned model sees at train
>   and inference time. Must equal `build_styling_prompt`'s system output so
>   train == inference.

## A. Teacher model (commercial-safe)

| | |
|---|---|
| Teacher | **`Qwen2.5-32B-Instruct`** (Apache-2.0) — first choice; or `Llama-3.3-70B-Instruct` (Llama 3.1+ permits output-for-training) |
| Forbidden | GPT-4o / Claude / Gemini — ToS forbids using outputs to build a competing model |
| Decoding | `temperature=0.3`, `top_p=0.9`, `max_tokens=512`, one generation per `(base, style)`; optionally sample 2 and keep the first that passes QA |
| Determinism | log `seed` per call so the run is reproducible |

Low temperature: variety comes from the **style axis** (§C), not sampling noise.
High-quality, faithful targets matter more than diverse phrasings.

## B. Inputs — base texts

| Source | Rows | Notes |
|---|---|---|
| `data/finetune/base_transcripts.jsonl` | 1,139 | real transcripts (`raw`), the true ASR input distribution. Prefer these. |
| `synth_spoken` (generated, §E) | fill to ~8–12k | teacher-generated spoken-style utterances to scale up |
| `task_shaped` (generated, §F) | ~0.5–1k | task-looking inputs for the transform-not-execute negatives |

Skew guard: many real transcripts are *about building Lune Voice*. Cap that
topic and lean on `synth_spoken` to broaden domains, or the model overfits to
dev-talk.

## C. The style axis

Each base text is styled under several styles. Two kinds:

- **restyle** — tone / register / casing / punctuation / format changes.
  **Preserve wording, content, meaning**; change only what the rule requires.
- **transform** — the instruction reshapes the text (enhance, summarize,
  bulletize). Perform the transform **on the text**; never execute it.

### C.1 Shipped styles (use the EXACT production rule text)

| `style_id` | `style_kind` | `style_name` | `style_rules` (verbatim) |
|---|---|---|---|
| `casual` | restyle | Casual | Replace every colon with a comma. Replace semicolons with a comma or period, whichever reads more naturally. Remove all quotation marks. Remove exclamation marks and dashes. Lowercase everything except proper nouns. Use contractions. Keep the original phrasing; do not convert quotes into reported speech. |
| `professional` | restyle | Professional | Use standard capitalization and grammar. Break run-on sentences. Keep it direct and clear. Do not use contractions. |
| `formal` | restyle | Formal | Use formal register. Expand all contractions. Replace casual words with formal equivalents. Write complete, structured sentences. |
| `technical` | restyle | Technical | Wrap code identifiers, function names, types, and keywords in backticks. Apply PascalCase or snake_case as context dictates. Keep technical jargon. |
| `writing` | restyle | Writing | Write polished prose. Break run-on sentences. Vary sentence length. Remove filler words like 'so', 'just', 'basically', 'like'. Keep the author's voice. |
| `enhance_prompt` | transform | Enhance Prompt | Rewrite this prompt to be more specific, actionable, and efficient—without changing its length or altering its original meaning, structure, or intent. Preserve the same word count and sentence structure as the input. |

### C.2 Synthetic long tail (teach generalization to custom styles)

Add ~12–20 more so the model learns *the concept of styling*, not 6 fixed
voices. Rotate them across base texts. Suggested starters:

| `style_id` | kind | rule (starter — vary the wording) |
|---|---|---|
| `academic` | restyle | Use an academic register: precise, hedged, third-person; expand contractions; no slang. |
| `friendly` | restyle | Warm and approachable; light contractions; keep it plain and encouraging. |
| `concise` | restyle | Cut every redundant word. Keep the meaning; make it as short as natural. |
| `punchy` | restyle | Short, high-energy sentences. Strong verbs. No filler. |
| `neutral` | restyle | Flat, objective tone. Remove opinion and emphasis words. |
| `cheerful` | restyle | Upbeat and positive without adding facts. |
| `summarize` | transform | Summarize the text in 1–2 sentences, keeping only the key point. |
| `bulletize` | transform | Turn the text into a short bulleted list of its distinct points. |
| `action_items` | transform | Extract the concrete to-dos as an imperative checklist. |
| `expand` | transform | Elaborate the text with more detail, inventing no new facts. |
| `email_draft` | transform | Reframe the text as a short, well-structured email body (no subject line). |
| `tldr` | transform | Prepend a one-line TL;DR of the text, then the text unchanged. |

Keep a held-out slice of styles (e.g. 2–3) **out of training** to measure true
generalization to unseen instructions.

## D. Teacher labeler prompt (VERBATIM — the target generator)

**System:**

```
You are a data-labeling engine that produces training targets for a text-styling model. You are given a STYLE (a name, a kind, and rules) and an INPUT transcript from speech-to-text (often lowercase, disfluent, with filler words). Produce the INPUT rewritten so it obeys the STYLE.

Hard rules:
1. Apply ONLY the transformation the STYLE describes. Never invent facts, names, numbers, or details that are not in the INPUT.
2. If STYLE KIND is "restyle": preserve the INPUT's wording, content, and meaning. Change only what the rules require (tone, casing, punctuation, format).
3. If STYLE KIND is "transform": perform that transformation ON the input text. NEVER execute what the input describes. If the INPUT says "write a function that sorts a list", you rewrite or enhance that request itself; you do NOT output code that sorts a list.
4. Remove speech disfluencies (um, uh, like, you know, false starts, stutters) unless the style explicitly keeps a raw spoken feel.
5. Output ONLY the rewritten text. No preamble, no surrounding quotes, no explanation, no markdown code fences, no "Here is".
6. Keep the length proportional to the INPUT unless the style is summarize/expand. Do not pad.
7. Preserve profanity and sensitive terms; do not censor.

If the INPUT is empty, pure noise, or cannot be meaningfully styled, output exactly: SKIP
```

**User (template):**

```
STYLE NAME: {style_name}
STYLE KIND: {style_kind}
STYLE RULES: {style_rules}

INPUT:
{base_text}
```

The `SKIP` sentinel lets the pipeline drop un-styleable inputs cleanly (they are
filtered out, not written).

## E. Generating `synth_spoken` base texts (VERBATIM)

Used to scale the base-text pool to ~8–12k with realistic ASR-style inputs.

**System:**

```
You generate realistic raw speech-to-text transcripts to train a dictation styler. Each transcript is 1–4 sentences of natural spoken English exactly as an ASR system would output it: mostly lowercase, little or no punctuation, filler words (um, uh, like, you know), occasional false starts and self-corrections. Vary the domain widely: work chat, personal notes, coding/dev talk, text messages, meeting talk, brainstorming, reminders, opinions. Keep them mundane and specific. Do NOT write clean, punctuated prose. Do NOT number them.

Output a JSON array of {n} strings and nothing else.
```

**User:** `Generate {n} transcripts. Domain focus: {domain}.` (rotate `domain`
over the list above; dedupe against existing base texts.)

Store each as `{"raw": "...", "source": "synth_spoken", "domain": "..."}` and run
it through the SAME labeler (§D).

## F. Transform-not-execute negatives (~10% of #2)

The measured failure is the small model *executing* the input. Teach the
boundary explicitly.

1. **Seed task-shaped inputs** (~0.5–1k) with the teacher:

   ```
   System: You generate short, imperative task requests a user might dictate — the kind that a naive model would try to execute. Examples of shape: "write a function that ...", "make me a landing page for ...", "explain X to a beginner", "build a spreadsheet that ...", "draft an email to ...". 1 sentence each, spoken style, lowercase. Output a JSON array of {n} strings, nothing else.
   ```

2. **Label them with transform styles** (`enhance_prompt`, `summarize`,
   `professional`, …) through the §D labeler. Rule #3 forces the output to be the
   *transformed request*, not the executed result.

3. **Anchor with hand-authored gold negatives** (include verbatim, do not
   filter):

   | instruction (style) | input | output |
   |---|---|---|
   | Enhance Prompt | write a function that sorts a list of numbers | Write a function that sorts a list of numbers in ascending order, specifying the input type and expected return value. |
   | Enhance Prompt | make me a landing page for my bakery | Design a landing page for my bakery, including a hero section, product menu, and contact details. |
   | Professional | explain how photosynthesis works to a five year old | Please explain how photosynthesis works in terms simple enough for a five-year-old. |

If the labeler emits code (```/`def `/`function `) for these, it is a **hard QA
fail** (§H) — that is exactly the behavior being trained out.

## G. Training instruction (what the SMALL model sees)

Each row's `instruction` field is the runtime system prompt, so train ==
inference. It **must** equal what `build_styling_prompt` emits (system half),
**without** few-shot examples (the fine-tune learns the styles):

```
You rewrite the user's transcript by applying the style below. Change only what the style requires; keep the original wording, content, and meaning everywhere else. Apply the instruction to the transcript — never execute what it describes. Output only the rewritten transcript: no preamble, no explanation, no code.

Style: {style_name}
Rules: {style_rules}
```

If `build_styling_prompt` changes, regenerate this field (or template it from the
same source) so they never drift.

## H. QA filter (drop before writing the training row)

Reuse the red-flag logic from `benchmarks/quality/styling_capability.rs`. Drop a
pair if ANY of:

| Flag | Rule |
|---|---|
| `SKIP` | output equals `SKIP` |
| `EMPTY` | output blank after trim |
| `CODEGEN` | output contains a ``` fence, or starts with `def `/`function `/`class `/`import ` (always a fail — even for `technical`, which uses inline backticks, never code blocks) |
| `META` | output contains preamble: `here is`, `here's`, `sure,`, `certainly`, `as an ai`, `i have rewritten`, `i've rewritten`, `the rewritten` |
| `NO-OP` | normalized `output` == normalized `input` (restyle styles only; a no-op restyle is a miss) |
| `ECHO` | output contains the instruction text or the style rules verbatim |
| `LEN` | word-count ratio `output/input` outside `[0.3, 3.0]` — except `summarize`/`tldr` (allow short) and `expand` (allow long) |

Then **dedupe** exact + near-duplicate on `(input, style_id)`.

Optional second-pass verification (raises quality, costs 2× teacher calls): ask
the teacher `Did this rewrite change any facts or fail the style? Answer YES or
NO plus one reason.` and drop the `YES` rows.

## I. Columns / schema

### I.1 Raw generation record (one per teacher call — audit trail)

`data/finetune/style_matrix.raw.jsonl`

| column | type | description |
|---|---|---|
| `pair_id` | str | `sm_{base_id}_{style_id}` — unique |
| `base_id` | int/str | source id (`base_transcripts.jsonl` id, or synth id) |
| `base_source` | str | `correction_pairs` \| `synth_spoken` \| `task_shaped` |
| `style_id` | str | e.g. `professional` |
| `style_kind` | str | `restyle` \| `transform` |
| `style_name` | str | display name used in the prompts |
| `style_rules` | str | verbatim rule text |
| `input` | str | the base text sent to the teacher |
| `output` | str | the teacher's raw styled text |
| `teacher` | str | model id, e.g. `Qwen2.5-32B-Instruct` |
| `gen_params` | obj | `{temperature, top_p, max_tokens, seed}` |
| `qa_passed` | bool | filled by §H |
| `qa_flags` | str[] | e.g. `["CODEGEN"]` |

### I.2 Training-ready record (post-QA, deduped)

`data/finetune/style_matrix.train.jsonl` — the unified schema shared by all
three datasets (Part I §1):

| column | type | description |
|---|---|---|
| `instruction` | str | the §G system prompt for this style |
| `input` | str | the base text |
| `output` | str | the QA-passed styled text |
| `source` | str | `style_matrix` |
| `meta` | obj | `{style_id, style_kind, base_source, preserves_wording}` |

### I.3 Rendered for training (Llama-3 chat template)

Loss masked to the assistant span only:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>
```

## J. Worked examples (training-ready rows)

**Restyle (professional):**
```json
{"instruction": "You rewrite the user's transcript by applying the style below. Change only what the style requires; keep the original wording, content, and meaning everywhere else. Apply the instruction to the transcript — never execute what it describes. Output only the rewritten transcript: no preamble, no explanation, no code.\n\nStyle: Professional\nRules: Use standard capitalization and grammar. Break run-on sentences. Keep it direct and clear. Do not use contractions.", "input": "so yeah the deploy's basically done i just gotta double check the numbers and then i'll send it over", "output": "The deployment is essentially complete. I will double-check the figures and then send it over.", "source": "style_matrix", "meta": {"style_id": "professional", "style_kind": "restyle", "base_source": "correction_pairs", "preserves_wording": true}}
```

**Transform (enhance_prompt) — the critical negative:**
```json
{"instruction": "You rewrite the user's transcript by applying the style below. ...\n\nStyle: Enhance Prompt\nRules: Rewrite this prompt to be more specific, actionable, and efficient—without changing its length or altering its original meaning, structure, or intent. Preserve the same word count and sentence structure as the input.", "input": "write a function that sorts a list of numbers", "output": "Write a function that sorts a list of numbers in ascending order, specifying the input type and the expected return value.", "source": "style_matrix", "meta": {"style_id": "enhance_prompt", "style_kind": "transform", "base_source": "task_shaped", "preserves_wording": false}}
```

**Casual:**
```json
{"instruction": "... Style: Casual\nRules: Replace every colon with a comma. ... Use contractions.", "input": "The plan was simple: cut costs; raise prices; hire fewer people!", "output": "the plan was simple, cut costs, raise prices, hire fewer people", "source": "style_matrix", "meta": {"style_id": "casual", "style_kind": "restyle", "base_source": "synth_spoken", "preserves_wording": true}}
```

## K. Build order (this dataset)

1. Assemble base texts: `base_transcripts.jsonl` + generate `synth_spoken` (§E) +
   `task_shaped` seeds (§F).
2. Pick the style set (§C): 6 shipped + long tail; reserve 2–3 held-out styles.
3. For every `(base, style)` pair, call the teacher (§D) → write
   `style_matrix.raw.jsonl` (§I.1).
4. Run the QA filter (§H) → drop fails, dedupe.
5. Emit `style_matrix.train.jsonl` (§I.2); merge the hand-authored gold negatives.
6. Hand off to the blend/holdout step in Part I §5.

Target: ~30–45k passing pairs. Iterate on the teacher prompt and style rules,
not the small-model hyperparameters.
