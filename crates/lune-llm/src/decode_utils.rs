//! Shared decode-loop utilities for LLM backends.
//!
//! Contains stop-sequence detection, partial-stop trimming, streaming chunk
//! emission, and thinking-gate logic.  Used by `OrtLlmBackend`.

/// Length of the longest stop sequence string (`<|im_end|>` = 10 bytes).
/// Used to bound the suffix window scanned for stop sequences each decode step.
pub const MAX_STOP_SEQ_LEN: usize = 10;

/// All stop sequences checked during decoding.  Order is irrelevant for
/// correctness; we scan a small window (~20 bytes) so iteration cost is
/// dominated by `memchr`, not sequence count.
///
/// NOTE: `<think>` / `</think>` are intentionally NOT stop sequences.
/// When a small model (e.g. Qwen3-0.6B) ignores `/no_think` and reasons
/// anyway, halting at `</think>` would capture only the reasoning text and
/// discard the actual answer that follows.  Instead we let generation
/// continue past the thinking block and rely on `strip_thinking` + the
/// streaming thinking-gate to suppress reasoning from output.
pub const STOP_SEQUENCES: &[&str] = &["<|im_end|>"];

/// Closing tag for model thinking blocks.  Used by the streaming
/// thinking-gate to detect when reasoning is complete.
pub const THINK_CLOSE: &str = "</think>";

/// Minimum bytes past `emitted_up_to` before emitting a chunk.
/// Set to 1 so chunks fire at every word boundary — one word at a time.
pub const MIN_CHUNK_BYTES: usize = 1;

// Compile-time guarantee that MAX_STOP_SEQ_LEN stays in sync with STOP_SEQUENCES
// AND that every stop sequence is pure ASCII (required by byte-level scanning).
const _: () = {
    let mut i = 0;
    let mut max = 0;
    while i < STOP_SEQUENCES.len() {
        let bytes = STOP_SEQUENCES[i].as_bytes();
        let len = bytes.len();
        if len > max {
            max = len;
        }
        // Assert all bytes are ASCII so byte-level memchr + starts_with is safe.
        let mut j = 0;
        while j < len {
            assert!(bytes[j] < 0x80, "stop sequence contains non-ASCII byte");
            j += 1;
        }
        i += 1;
    }
    assert!(
        max == MAX_STOP_SEQ_LEN,
        "MAX_STOP_SEQ_LEN does not match longest STOP_SEQUENCES entry"
    );
};

/// Scan the tail of `output` for any stop sequence that may straddle the
/// old/new boundary. Returns `true` (and truncates `output`) if found.
///
/// Operates entirely on bytes — all stop sequences are compile-time-verified
/// ASCII, so byte comparisons are equivalent to string comparisons. This
/// avoids `floor_char_boundary` and UTF-8 validation overhead per decode step.
#[inline]
pub fn check_stop_sequences(output: &mut String, piece_len: usize) -> bool {
    let bytes = output.as_bytes();
    let win_end = bytes.len();
    let win_start = win_end.saturating_sub(piece_len + MAX_STOP_SEQ_LEN - 1);
    let window = &bytes[win_start..];

    // All stop sequences start with '<'. Find each '<' and test all sequences
    // at that position. First match = earliest occurrence, always correct for
    // straddling detection.
    for lt_off in memchr::memchr_iter(b'<', window) {
        let remaining = &window[lt_off..];
        for seq in STOP_SEQUENCES {
            if remaining.len() >= seq.len() && remaining[..seq.len()] == *seq.as_bytes() {
                output.truncate(win_start + lt_off);
                return true;
            }
        }
    }
    false
}

/// Trim any partial stop sequence that straddles the output boundary when
/// generation hits max tokens. Without this, output can end with a prefix
/// like `<|im_en` which leaks control markup.
///
/// Uses a single `memrchr` for the last `<` instead of O(sequences * prefix_len)
/// nested iteration — all stop sequences are ASCII and start with `<`, so any
/// partial prefix must begin at the rightmost `<` in the tail.
pub fn trim_partial_stop_sequences(output: &mut String) {
    let bytes = output.as_bytes();
    let tail_start = bytes.len().saturating_sub(MAX_STOP_SEQ_LEN);
    let tail = &bytes[tail_start..];

    // All stop sequences start with '<'; a partial prefix must begin at one.
    let Some(lt_off) = memchr::memrchr(b'<', tail) else {
        return;
    };

    let candidate = &bytes[tail_start + lt_off..];

    // If the substring from the last '<' to the end is a strict prefix of any
    // stop sequence, it's a dangling partial — truncate it.
    for seq in STOP_SEQUENCES {
        let seq_bytes = seq.as_bytes();
        if candidate.len() < seq_bytes.len() && seq_bytes[..candidate.len()] == *candidate {
            output.truncate(tail_start + lt_off);
            return;
        }
    }
}

/// Streaming state tracker for the decode loop.
///
/// Encapsulates `emitted_up_to` and the thinking gate so both backends
/// share identical streaming semantics.
pub struct StreamingState {
    /// Bytes of the output buffer already sent to the channel.
    pub emitted_up_to: usize,
    /// Whether the model's thinking phase is complete (or was never started).
    pub thinking_done: bool,
}

impl StreamingState {
    /// Create a new streaming state.
    ///
    /// `is_streaming` should be `true` when a `chunk_tx` channel is provided.
    /// When not streaming, the thinking gate is bypassed.
    #[inline]
    pub fn new(is_streaming: bool) -> Self {
        Self {
            emitted_up_to: 0,
            thinking_done: !is_streaming,
        }
    }

    /// Check if the thinking block just ended; if so, advance `emitted_up_to`
    /// past `</think>` and any trailing whitespace.
    ///
    /// Call this after each token is appended to `output`.
    #[inline]
    pub fn check_thinking_done(&mut self, output: &str, piece_len: usize) {
        if self.thinking_done {
            return;
        }
        let search_start = output
            .len()
            .saturating_sub(piece_len + THINK_CLOSE.len() - 1);
        if let Some(pos) = output[search_start..].find(THINK_CLOSE) {
            self.thinking_done = true;
            let content_start = search_start + pos + THINK_CLOSE.len();
            // Skip whitespace immediately after </think>.
            self.emitted_up_to = output[content_start..]
                .find(|c: char| !c.is_ascii_whitespace())
                .map_or(output.len(), |off| content_start + off);
        }
    }

    /// Try to emit a word-boundary chunk from `output` via `tx`.
    ///
    /// Returns `true` if a chunk was emitted, `false` otherwise.
    /// Only emits after the thinking gate opens.
    #[inline]
    pub fn try_emit_chunk(&mut self, output: &str, tx: &tokio::sync::mpsc::Sender<String>) -> bool {
        if !self.thinking_done {
            return false;
        }
        let safe_end = output.len().saturating_sub(MAX_STOP_SEQ_LEN);
        if safe_end > self.emitted_up_to + MIN_CHUNK_BYTES {
            let safe_slice = &output.as_bytes()[self.emitted_up_to..safe_end];
            if let Some(ws_off) = memchr::memrchr(b' ', safe_slice) {
                let boundary = self.emitted_up_to + ws_off + 1;
                let chunk = output[self.emitted_up_to..boundary].to_string();
                self.emitted_up_to = boundary;
                let _ = tx.blocking_send(chunk);
                return true;
            }
        }
        false
    }

    /// Flush remaining text after generation completes.
    ///
    /// Applies `strip_thinking` and sends any unsent tail through the channel.
    /// Returns the cleaned output string.
    pub fn flush(
        &self,
        output: &str,
        tx: &tokio::sync::mpsc::Sender<String>,
        strip_thinking: impl FnOnce(&str) -> String,
    ) -> String {
        let cleaned = strip_thinking(output);
        if self.thinking_done && self.emitted_up_to < output.len() {
            // Thinking gate passed — emit remaining clean text.
            let remaining = if cleaned.len() < output.len() {
                let already_sent = self
                    .emitted_up_to
                    .saturating_sub(output.len() - cleaned.len());
                &cleaned[already_sent.min(cleaned.len())..]
            } else {
                &output[self.emitted_up_to..]
            };
            if !remaining.is_empty() {
                let _ = tx.blocking_send(remaining.to_string());
            }
        } else if !self.thinking_done && !cleaned.is_empty() {
            // Model never produced </think> — emit the strip_thinking result.
            let _ = tx.blocking_send(cleaned.clone());
        }
        cleaned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_sequence_detection() {
        let mut s = String::from("Hello world<|im_end|>");
        assert!(check_stop_sequences(&mut s, 10));
        assert_eq!(s, "Hello world");
    }

    #[test]
    fn think_tags_not_stop_sequences() {
        let mut s = String::from("result<think>");
        assert!(!check_stop_sequences(&mut s, 7));
        assert_eq!(s, "result<think>");

        let mut s = String::from("leaked</think>");
        assert!(!check_stop_sequences(&mut s, 8));
        assert_eq!(s, "leaked</think>");
    }

    #[test]
    fn partial_stop_sequence_trimming() {
        let mut s = String::from("output text<|im_en");
        trim_partial_stop_sequences(&mut s);
        assert_eq!(s, "output text");
    }

    #[test]
    fn partial_stop_sequence_trimming_longest_wins() {
        let mut s = String::from("output text<|im_end");
        trim_partial_stop_sequences(&mut s);
        assert_eq!(s, "output text");
    }

    #[test]
    fn no_partial_trim_when_clean() {
        let mut s = String::from("clean output");
        trim_partial_stop_sequences(&mut s);
        assert_eq!(s, "clean output");
    }

    #[test]
    fn stop_sequence_early_exit_no_angle_bracket() {
        let mut s = String::from("hello world");
        assert!(!check_stop_sequences(&mut s, 5));
        assert_eq!(s, "hello world");
    }

    #[test]
    fn partial_trim_single_angle_bracket() {
        let mut s = String::from("output<");
        trim_partial_stop_sequences(&mut s);
        assert_eq!(s, "output");
    }

    #[test]
    fn streaming_state_thinking_gate() {
        let mut state = StreamingState::new(true);
        assert!(!state.thinking_done);

        // Simulate incremental decode: the </think> tag is the new piece.
        let partial = "<think>reasoning here</think>";
        state.check_thinking_done(partial, THINK_CLOSE.len());
        assert!(state.thinking_done);

        // Now "answer" is appended — emitted_up_to should point to it.
        let full = "<think>reasoning here</think>answer";
        assert_eq!(&full[state.emitted_up_to..], "answer");
    }

    #[test]
    fn streaming_state_no_thinking() {
        let state = StreamingState::new(false);
        assert!(state.thinking_done);
    }
}
