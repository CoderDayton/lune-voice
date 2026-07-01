# n-gram model data (`lune-dict::ngram`)

The trained n-gram model binary lives here as `lune_ngram_model.bin`. It is
embedded into the library at compile time via `include_bytes!` in
`ngram::NGramModel::load_bundled()`, so it must exist before building.

> **Status:** the model is embedded but **not currently loaded at runtime** — the
> ASR candidate reranker that consumed it was removed. `train_model` and this
> `.bin` are retained for potential reuse.

Regenerate it from the bundled seed sentences (no network required):

```
cargo run -p lune-dict --bin train_model
```

The optional `training` feature additionally enables fetching a Wordnik corpus
(`ngram::training`) for a larger model; that path needs a Wordnik API key.
