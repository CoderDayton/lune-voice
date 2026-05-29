//! Compiles the supplemental vocabulary at build time.
//!
//! Each `data/vocab/*.txt` file is a category (one word per line; `#` comments
//! and `a|b` alternate forms are supported). The compiled bincode blob is
//! written to `$OUT_DIR/supplemental_vocab.bin` and embedded by `vocab.rs` via
//! `include_bytes!`. The `.txt` files are the single source of truth — there is
//! no committed binary artifact to keep in sync.

use std::collections::HashMap;
use std::path::Path;

#[derive(serde::Serialize)]
struct SupplementalVocab {
    categories: Vec<String>,
    words: HashMap<String, u8>,
}

fn main() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = manifest_dir.join("data/vocab");
    let out_path =
        Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR")).join("supplemental_vocab.bin");

    // Rebuild whenever the vocab directory or any category file changes.
    println!("cargo:rerun-if-changed=data/vocab");

    let mut vocab = SupplementalVocab {
        categories: Vec::new(),
        words: HashMap::new(),
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
        let category = path.file_stem().unwrap().to_string_lossy().to_string();
        let cat_id = u8::try_from(vocab.categories.len()).expect("at most 256 vocab categories");
        vocab.categories.push(category);

        let content = std::fs::read_to_string(&path).expect("read vocab file");
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            for word in line.split('|').map(str::trim).filter(|w| !w.is_empty()) {
                let lower = word.to_ascii_lowercase();
                // First category to claim a word wins (case-insensitive).
                vocab.words.entry(lower.clone()).or_insert(cat_id);
                // Preserve the original casing as a separate key when it differs.
                if word != lower {
                    vocab.words.entry(word.to_string()).or_insert(cat_id);
                }
            }
        }
    }

    let bytes = bincode::serde::encode_to_vec(&vocab, bincode::config::standard())
        .expect("encode supplemental vocab");
    std::fs::write(&out_path, &bytes).expect("write supplemental vocab");
}
