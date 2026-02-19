# 06 – Performance & Resource Targets

← [External Interfaces](./05-external-interfaces.md) | → [Security](./07-security.md)

---

## 8.1 Latency Targets

### End-to-End Latency Budget

```
Speech → displayed text target: < 1000 ms

Breakdown (typical session):
  Audio capture buffering:      ~  50 ms
  ASR streaming chunk delay:    ~ 200–400 ms   (Voxtral configurable delay)
  Stabilization window:         ~ 200–500 ms   (silence detection)
  LLM post-processing:          ~ 100–400 ms   (1–3B model, short segment)
  Text output injection:        ~  10–50 ms
  ─────────────────────────────────────────
  Worst-case sum:               ~ 1000–1400 ms
  Target (optimized):           < 1000 ms
```

### RTF Target

| Metric | Target | Reference |
|--------|--------|-----------|
| Voxtral RTF | ≤ 0.5 | Published ≈ 0.42 on reference device |
| LLM post-processing latency | ≤ 400 ms | Short segment (≤ 50 words), Q4 3B model |
| End-to-end (optimized) | < 1 s | Measured on target hardware class |

### Latency Knobs

| Knob | Default | Effect |
|------|---------|--------|
| Stabilization window | 500 ms | Shorter → more partials, more flicker |
| LLM batch size | Sentence | Per-token → higher latency but same output |
| ASR chunk size | Model-defined | Smaller → lower latency, more requests |
| `stream: true` (LLM) | Off (v0.1) | On → perceived latency reduction in v0.2 |

---

## 8.2 Memory & Model Size Budget

### Model Footprints (Approximate)

| Model | Format | Weights Size | GPU VRAM (inference) |
|-------|--------|-------------|----------------------|
| Voxtral Mini 4B Realtime | Q4 GGUF | ~2–3 GB | ~3–5 GB |
| Local LLM 0.6B (Qwen3) | Q4 GGUF | ~400 MB | ~1 GB |
| Local LLM 3B | Q4 GGUF | ~2 GB | ~3–4 GB |

### Hardware Recommendations

| Tier | GPU VRAM | System RAM | Expected Experience |
|------|---------|-----------|---------------------|
| Minimum (CPU-only) | None | 16 GB | Functional; higher latency |
| Recommended | 8 GB | 16 GB | Good; both models GPU-accelerated |
| Optimal | 12+ GB | 32 GB | Excellent; headroom for larger LLM |

### Constraints
- Both models (ASR + LLM) must fit simultaneously in VRAM on recommended tier.
- CPU fallback required; Q4 quantized targets usable RTF on modern laptop CPU.
- App process overhead (Tauri + audio) must stay below 200 MB RAM.

---

## 8.3 Default Configuration (Conservative)

```toml
[audio]
frame_size_ms = 30          # aligned to Voxtral encoder
vad_enabled   = true

[asr]
stabilization_window_ms = 500
sliding_context_secs    = 30

[llm]
batch_mode          = "sentence"  # sentence | paragraph | on_stop
max_segment_words   = 50          # larger → batch before sending to LLM
temperature         = 0.2
max_tokens          = 512

[app]
output_mode = "LiveTyping"  # LiveTyping | TypeOnStop
```

---

## Section Checklist

### Latency
- [ ] End-to-end latency measured on macOS reference device (document specs)
- [ ] End-to-end latency measured on Windows reference device (document specs)
- [ ] RTF measured for `VoxtralLocalBackend` on CPU-only path
- [ ] RTF measured for `VoxtralLocalBackend` on GPU path (CUDA / Metal)
- [ ] LLM post-processing latency measured for each default model

### Memory
- [ ] Peak VRAM usage profiled with both models loaded simultaneously
- [ ] System RAM usage profiled during active dictation session
- [ ] CPU-only path: peak RAM measured (no VRAM)
- [ ] App idle RAM measured (Tauri window + background threads, no models loaded)

### Optimization
- [ ] VAD reduces unnecessary ASR calls by ≥ 30% in silence-heavy recordings (measured)
- [ ] LLM batching: single request per sentence vs. per-word compared; sentence chosen
- [ ] Stabilization window tuning: trade-off between flicker and latency documented
- [ ] `stream: true` LLM mode tracked for v0.2 (perceived latency improvement)

### Regression Guards
- [ ] Benchmark suite created: RTF, LLM latency, end-to-end latency
- [ ] CI runs benchmarks on each PR targeting `main` (or nightly)
- [ ] Latency regression threshold: alert if end-to-end p95 > 1500 ms
