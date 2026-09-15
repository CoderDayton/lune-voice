// SPDX-License-Identifier: GPL-3.0-or-later

//! Compiles per-language frequency lists from `corpora/{lang}.csv` into the
//! `data/{lang}_freq.bin` artifacts bundled via `include_bytes!`.
//!
//! Run with `cargo run -p lune-dict --features data-tools --bin build_freq`.
//!
//! Source corpora are the OpenSubtitles-derived word lists from
//! hermitdave/FrequencyWords (MIT License, Copyright (c) 2016 Hermit Dave).
//! Input accepts either CSV (`word,freq`) or the raw whitespace-separated
//! upstream format (`word freq`).
//!
//! `corpora/` is gitignored — the compiled `data/{lang}_freq.bin` are what
//! ship, so the ~9 MB of inputs are not carried in the repo. Re-fetch them
//! only when regenerating, one file per language, from
//! <https://github.com/hermitdave/FrequencyWords>:
//!
//! ```text
//! content/2018/{lang}/{lang}_full.txt  ->  corpora/{lang}.csv
//! ```
//!
//! A missing language is skipped with a warning rather than failing the run,
//! so you can regenerate a single language without fetching the rest.

use std::collections::HashMap;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let corpora_dir = manifest_dir.join("../../corpora");
    let out_dir = manifest_dir.join("data");

    let languages = [
        "en", "de", "es", "fr", "it", "ja", "ko", "nl", "pt", "ru", "zh",
    ];
    let mut total_entries = 0;
    let mut total_bytes = 0;

    for lang in &languages {
        let csv_path = corpora_dir.join(format!("{lang}.csv"));
        if !csv_path.exists() {
            eprintln!("  ⚠ skipping {lang}: {}", csv_path.display());
            continue;
        }

        let content = std::fs::read_to_string(&csv_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", csv_path.display()));

        let mut freq_map: HashMap<String, u32> = HashMap::new();

        for (i, line) in content.lines().enumerate() {
            if i == 0 && line.starts_with("word,") {
                continue;
            }
            // Accept either CSV (`word,freq`) or the raw hermitdave format
            // (`word freq`, whitespace-separated) so source lists can be dropped
            // in without conversion.
            let Some((word, freq_str)) = line
                .split_once(',')
                .or_else(|| line.split_once(char::is_whitespace))
            else {
                continue;
            };
            let word = word.trim().to_lowercase();
            if word.is_empty() {
                continue;
            }
            // Skip words with digits or control chars; allow letters, apostrophes, hyphens, CJK
            if word.chars().any(|c| c.is_ascii_digit() || c.is_control()) {
                continue;
            }
            let Ok(freq) = freq_str.trim().parse::<u64>() else {
                continue;
            };
            let freq = freq.min(u32::MAX as u64) as u32;
            freq_map
                .entry(word)
                .and_modify(|e| *e = (*e).max(freq))
                .or_insert(freq);
        }

        let entries: Vec<(String, u32)> = freq_map.into_iter().collect();
        let bytes =
            bincode::serde::encode_to_vec(&entries, bincode::config::standard()).expect("encode");

        let out_path = out_dir.join(format!("{lang}_freq.bin"));
        std::fs::write(&out_path, &bytes).expect("write bin");

        println!(
            "  ✓ {lang}_freq.bin: {:>6} entries, {:>6.1} KB",
            entries.len(),
            bytes.len() as f64 / 1024.0
        );
        total_entries += entries.len();
        total_bytes += bytes.len();
    }

    println!(
        "\n  Total: {} entries, {:.1} MB across {} languages",
        total_entries,
        total_bytes as f64 / (1024.0 * 1024.0),
        languages.len()
    );
}
