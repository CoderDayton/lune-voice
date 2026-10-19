// SPDX-License-Identifier: GPL-3.0-or-later

use lune_core::config::FormatToggles;
use lune_core::dict::DictionaryEntry;
use lune_dict::UserDictionary;
use lune_dict::corrector::{HWCE_FLAG_THRESHOLD, LexicalGate};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, raws, dict, out] = args.as_slice() else {
        eprintln!("usage: replay_pairs <raws> <dictionary.tsv> <out>");
        std::process::exit(2);
    };
    let raws = std::fs::read_to_string(raws).expect("read raws");
    let entries = std::fs::read_to_string(dict)
        .expect("read dictionary")
        .lines()
        .filter_map(|l| l.split_once('\t'))
        .map(|(phrase, replacement)| DictionaryEntry {
            phrase: phrase.into(),
            replacement: replacement.into(),
            priority: 5,
        })
        .collect();
    let dict = UserDictionary::new(entries);
    let gate = LexicalGate::new();
    let fmt = FormatToggles {
        brand_casing: Some(true),
        ..FormatToggles::default()
    };
    let none: &[(&str, f32, f32)] = &[];
    let shaped: Vec<String> = raws
        .split('\u{1e}')
        .map(|raw| {
            let text = raw.trim();
            let matches = dict.scan(text);
            let corrected = gate
                .correct_with_risk_threshold(
                    text,
                    none,
                    &matches,
                    "English",
                    HWCE_FLAG_THRESHOLD,
                    true,
                )
                .corrected_text;
            lune_dict::formatting::shape(&corrected, "English", &fmt)
        })
        .collect();
    std::fs::write(out, shaped.join("\u{1e}")).expect("write output");
}
