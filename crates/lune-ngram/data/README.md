# lune-ngram model data

Place the trained model binary here as `lune_ngram_model.bin`.

Generate it by running the training binary with the `training` feature enabled:

```
cargo run -p lune-ngram --features training --bin train -- --api-key $WORDNIK_API_KEY
```

The file is embedded into the library at compile time via `include_bytes!` in
`NGramModel::load_bundled()`. It must exist before building without the
`training` feature if you call that method.
