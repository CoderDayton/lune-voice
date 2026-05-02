use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct SupplementalVocab {
    categories: Vec<String>,
    words: HashMap<String, u8>,
}

fn main() {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/vocab");
    let out_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/supplemental_vocab.bin");

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
        let category = path.file_stem().unwrap().to_string_lossy().to_string();
        let cat_id = vocab.categories.len() as u8;
        vocab.categories.push(category.clone());

        let content = std::fs::read_to_string(&path).expect("read vocab file");
        let mut count = 0u32;
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // Support aliases: "Google | Alphabet" adds both forms
            let forms: Vec<&str> = line.split('|').map(|s| s.trim()).collect();
            for word in &forms {
                if word.is_empty() {
                    continue;
                }
                vocab.words.entry(word.to_string()).or_insert(cat_id);
                let lower = word.to_ascii_lowercase();
                if lower != *word {
                    vocab.words.entry(lower).or_insert(cat_id);
                }
            }
            count += 1;
        }
        println!("  {category}: {count} entries");
    }

    let bytes = bincode::serde::encode_to_vec(&vocab, bincode::config::standard()).expect("encode");
    std::fs::write(&out_path, &bytes).expect("write bin");

    println!(
        "\nSaved {} words across {} categories to {} ({} bytes)",
        vocab.words.len(),
        vocab.categories.len(),
        out_path.display(),
        bytes.len(),
    );
}
