//! Compiles the supplemental vocabulary at build time.
//!
//! Each `data/vocab/*.txt` file is one category of words (one word per line;
//! `#` comments and `a|b` alternate forms are supported). Categories are a
//! source-file grouping only — they are not carried into the compiled set.
//! The compiled bincode blob is
//! written to `$OUT_DIR/supplemental_vocab.bin` and embedded by `vocab.rs` via
//! `include_bytes!`. The `.txt` files are the single source of truth — there is
//! no committed binary artifact to keep in sync.

use std::collections::HashSet;
use std::path::Path;

#[derive(serde::Serialize)]
struct SupplementalVocab {
    words: HashSet<String>,
}

fn main() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = manifest_dir.join("data/vocab");
    let out_path =
        Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR")).join("supplemental_vocab.bin");

    // Rebuild whenever the vocab directory or any category file changes.
    println!("cargo:rerun-if-changed=data/vocab");

    let mut vocab = SupplementalVocab {
        words: HashSet::new(),
    };

    let mut entries: Vec<_> = std::fs::read_dir(&data_dir)
        .expect("read vocab dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "txt"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        println!("cargo:rerun-if-changed={}", path.display());
        let content = std::fs::read_to_string(&path).expect("read vocab file");
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            for word in line.split('|').map(str::trim).filter(|w| !w.is_empty()) {
                let lower = word.to_ascii_lowercase();
                // Preserve the original casing as a separate entry when it differs.
                if word != lower {
                    vocab.words.insert(word.to_string());
                }
                vocab.words.insert(lower);
            }
        }
    }

    let bytes = bincode::serde::encode_to_vec(&vocab, bincode::config::standard())
        .expect("encode supplemental vocab");
    std::fs::write(&out_path, &bytes).expect("write supplemental vocab");
}
