# Parakeet test model

`for-tests-ggml-parakeet-tdt.bin` is a 16 KB Parakeet TDT model with random
weights, copied unchanged from whisper.cpp v1.9.4
(`models/for-tests-ggml-parakeet-tdt.bin`, commit
`927cfce34f31707e17f2bff35c349632fb9e2c3a`, MIT licensed). whisper.cpp
generates it with `models/generate-parakeet-test-model.py`.

It cannot transcribe speech. `crates/lune-asr/tests/parakeet_backend.rs` uses
it to drive the real native loader, encoder, TDT decoder and teardown in CI
without downloading a production model.

SHA-256: `c25f3b2d2bda19161413789a03dc929d712e4a1b466fef1fb173daac4812944e`
