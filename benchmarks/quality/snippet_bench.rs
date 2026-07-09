//! Voice-snippet stage benchmark — the cost and safety of the snippet
//! expansion stage added to every dictation (`session.rs::process_and_deliver`).
//!
//! Measures, with the real sqlite store (lune-persist) and the real matcher
//! (lune-dict::snippets), no ASR models required:
//!
//! 1. **Per-utterance overhead** — `list_snippets` (file-backed sqlite) and
//!    matching on the common non-matching dictation path, scaled across
//!    library sizes. This is the price every dictation pays for the feature.
//! 2. **Mishear robustness** — one-edit corruptions of spoken trigger names
//!    must still resolve to the right snippet (or safely to none), never to a
//!    *different* snippet.
//! 3. **False-fire safety** — ordinary dictation must never expand a snippet.
//! 4. **Render throughput** — placeholder substitution cost.
//!
//! Exits non-zero on any correctness violation (wrong expansion / false fire),
//! so it doubles as a regression harness.
//!
//! Run: cgs bench --env TYPE=snippets
//!      (cargo run --release --example snippet_bench -p lune-app)

use std::hint::black_box;
use std::time::Instant;

use lune_core::dict::Snippet;
use lune_dict::snippets::{PlaceholderValues, match_utterance, render_body, resolve_name};
use lune_persist::Database;

/// Realistic user library for the quality sections.
const REALISTIC: &[(&str, &str)] = &[
    ("signature", "Best regards,\nDayton"),
    ("meeting header", "## Meeting — {date}\n\nAttendees:\n- "),
    ("meeting footer", "Adjourned at {time}."),
    (
        "standup note",
        "### Standup {date}\nYesterday:\nToday:\nBlockers:",
    ),
    ("email intro", "Hi,\n\nThanks for reaching out."),
    (
        "email outro",
        "Let me know if anything is unclear.\n\nBest,\nDayton",
    ),
    ("address block", "742 Evergreen Terrace\nSpringfield"),
    (
        "review checklist",
        "- [ ] tests\n- [ ] docs\n- [ ] changelog",
    ),
    ("daily journal", "# {date} — {time}\n\n"),
    ("invoice footer", "Payment due within 30 days of {date}."),
    (
        "out of office",
        "I am out of office and will reply after my return.",
    ),
    (
        "bug report",
        "**Seen:** {datetime}\n**Steps:**\n1. \n**Expected:**\n**Actual:**",
    ),
];

/// Ordinary dictation that must never expand a snippet (several share words
/// with triggers; two start with the `insert` command prefix).
const DICTATION: &[&str] = &[
    "this is an ordinary dictated sentence about nothing in particular",
    "the meeting header should go at the top of the document",
    "please add my signature to the form before sending it",
    "we adjourned the meeting and went home",
    "a coin to continue",
    "the disk into the drive",
    "email intro sections are usually short",
    "out of office replies annoy everyone",
    "standup notes were missing yesterday",
    "review the checklist before merging",
    "I filed a bug report this morning",
    "the invoice footer needs a new date",
    "daily journals help me focus",
    "insert",
    "address the block of flats on the corner",
    "will this work now",
];

fn snips(pairs: &[(&str, &str)]) -> Vec<Snippet> {
    pairs
        .iter()
        .enumerate()
        .map(|(i, (t, b))| Snippet {
            id: format!("s{i}"),
            trigger_phrase: (*t).to_string(),
            body: (*b).to_string(),
        })
        .collect()
}

/// Deterministic xorshift so corruption patterns are stable across runs.
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Apply one edit (substitution / deletion / insertion) to one word.
fn corrupt_word(word: &str, kind: u64, pos_seed: u64) -> String {
    let chars: Vec<char> = word.chars().collect();
    let pos = (pos_seed as usize) % chars.len();
    let letter = char::from(b'a' + (pos_seed % 26) as u8);
    let mut out = chars.clone();
    match kind % 3 {
        0 => out[pos] = if out[pos] == letter { 'z' } else { letter },
        1 => {
            out.remove(pos);
        }
        _ => out.insert(pos, letter),
    }
    out.into_iter().collect()
}

/// Mean µs/op over `iters` calls.
fn time_us<F: FnMut()>(iters: u32, mut f: F) -> f64 {
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters)
}

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       SNIPPET STAGE — overhead, robustness, false-fire, render    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let mut failures = 0u32;

    // ── 1. Per-utterance overhead vs library size ────────────────────────
    println!();
    println!("── Per-utterance overhead (real file-backed sqlite + matcher) ──");
    println!();
    println!(
        "  {:>8}  {:>16}  {:>18}  {:>18}",
        "library", "list_snippets µs", "match (miss) µs", "resolve (hit) µs"
    );

    let db_path =
        std::env::temp_dir().join(format!("lune-snippet-bench-{}.db", std::process::id()));
    let miss_utterance = "this is an ordinary dictated sentence about nothing in particular";

    for &n in &[0usize, 10, 100, 1000] {
        let _ = std::fs::remove_file(&db_path);
        let mut db = Database::open(&db_path).expect("open bench db");
        // Two-word generated triggers, unique per index.
        let lib: Vec<Snippet> = (0..n)
            .map(|i| Snippet {
                id: format!("gen{i}"),
                trigger_phrase: format!("alpha{} omega{}", i / 40, i % 40),
                body: format!("generated body {i} on {{date}}"),
            })
            .collect();
        for s in &lib {
            db.insert_snippet(&s.id, &s.trigger_phrase, &s.body)
                .expect("seed snippet");
        }

        let db_us = time_us(200, || {
            black_box(db.list_snippets().expect("list"));
        });
        let miss_us = time_us(20_000, || {
            black_box(match_utterance(black_box(&lib), black_box(miss_utterance)));
        });
        // Worst case: the spoken name resolves to the *last* trigger.
        let hit_us = if n == 0 {
            0.0
        } else {
            let last = lib[n - 1].trigger_phrase.clone();
            time_us(20_000, || {
                black_box(resolve_name(black_box(&lib), black_box(&last)));
            })
        };
        println!("  {n:>8}  {db_us:>16.1}  {miss_us:>18.2}  {hit_us:>18.2}");
    }
    let _ = std::fs::remove_file(&db_path);
    let _ = std::fs::remove_file(db_path.with_extension("db-wal"));
    let _ = std::fs::remove_file(db_path.with_extension("db-shm"));

    // ── 2. Mishear robustness (one edit per corrupted word) ─────────────
    println!();
    println!("── Mishear robustness — `insert <name>` with one-edit corruptions ──");
    println!();

    let lib = snips(REALISTIC);
    let mut rng = 0x5EED_CAFE_D00D_u64;
    let (mut hits, mut safe_misses, mut wrong) = (0u32, 0u32, 0u32);
    let mut trials = 0u32;

    for snippet in &lib {
        let words: Vec<&str> = snippet.trigger_phrase.split_whitespace().collect();
        for (wi, word) in words.iter().enumerate() {
            if word.len() < 3 {
                continue; // sub-3-letter words: a single edit is a different word
            }
            for kind in 0..3u64 {
                let pos_seed = xorshift(&mut rng);
                let corrupted = corrupt_word(word, kind, pos_seed);
                let spoken: Vec<String> = words
                    .iter()
                    .enumerate()
                    .map(|(i, w)| {
                        if i == wi {
                            corrupted.clone()
                        } else {
                            (*w).to_string()
                        }
                    })
                    .collect();
                let spoken = spoken.join(" ");
                trials += 1;
                match resolve_name(&lib, &spoken) {
                    Some(hit) if hit.id == snippet.id => hits += 1,
                    // Ambiguity (or an edit that collided with another
                    // trigger's fuzzy radius) safely resolving to nothing.
                    None => safe_misses += 1,
                    Some(other) => {
                        wrong += 1;
                        println!(
                            "  [WRONG] “{spoken}” → “{}” (expected “{}”)",
                            other.trigger_phrase, snippet.trigger_phrase
                        );
                    }
                }
            }
        }
    }
    let hit_rate = 100.0 * f64::from(hits) / f64::from(trials);
    println!(
        "  trials: {trials}   recovered: {hits} ({hit_rate:.1}%)   safe-miss: {safe_misses}   wrong: {wrong}"
    );
    if wrong > 0 {
        failures += wrong;
        println!("  [FAIL] a mishear expanded the wrong snippet");
    }

    // ── 3. False-fire safety on ordinary dictation ───────────────────────
    println!();
    println!("── False-fire safety — dictation must never expand ──");
    println!();
    let mut fired = 0u32;
    for utterance in DICTATION {
        if let Some(s) = match_utterance(&lib, utterance) {
            fired += 1;
            println!("  [FIRE] “{utterance}” expanded “{}”", s.trigger_phrase);
        }
    }
    println!(
        "  {} dictation utterances, {} false fires",
        DICTATION.len(),
        fired
    );
    if fired > 0 {
        failures += fired;
        println!("  [FAIL] ordinary dictation expanded a snippet");
    }

    // ── 4. Render throughput ─────────────────────────────────────────────
    println!();
    println!("── Render throughput — placeholder-heavy body ──");
    println!();
    let values = PlaceholderValues {
        date: "July 8, 2026".into(),
        time: "14:30".into(),
        datetime: "July 8, 2026 at 14:30".into(),
        clipboard: Some("clipboard payload text".into()),
    };
    let body = "# {date} — {time}\n\nFrom clipboard: {clipboard}\nStamped {datetime}. \
                Unknown {tokens} pass {through} untouched, and plain text is copied as-is.";
    let render_us = time_us(100_000, || {
        black_box(render_body(black_box(body), black_box(&values)));
    });
    println!(
        "  render_body: {render_us:.3} µs/op ({} chars in)",
        body.len()
    );

    // ── Verdict ──────────────────────────────────────────────────────────
    println!();
    if failures == 0 {
        println!("RESULT: PASS — no wrong expansions, no false fires");
    } else {
        println!("RESULT: FAIL — {failures} correctness violation(s)");
        std::process::exit(1);
    }
}
