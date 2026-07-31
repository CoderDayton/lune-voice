#!/usr/bin/env bash
# Sweep tunable knobs against the librispeech_40744 fixture only.
#
# Builds pipeline_real once, then re-runs it for every (mark × skip × margin)
# combo with LUNE_BENCH_FIXTURE=librispeech_40744 so each iteration takes ~1–2 s.
# Prints a sorted table of (WER, mark, skip, margin) at the end.
#
# Usage:
#   ./benchmarks/sweep_40744.sh
#   ASR=qwen ./benchmarks/sweep_40744.sh   # sweep against the Qwen backend instead
set -euo pipefail

cd "$(dirname "$0")/.."

ASR="${ASR:-parakeet}"
FIXTURE="${FIXTURE:-librispeech_40744}"
RESULTS="$(mktemp)"
trap 'rm -f "$RESULTS"' EXIT

# Sweep grid. Tightened to keep total wall time under ~2 minutes.
MARK_VALUES=(0.50 0.60 0.65 0.70 0.80)
SKIP_VALUES=(0.85 0.95 off)
MARGIN_VALUES=(0.0 2.5 5.0 10.0 999.0)

# Pre-build once so iterations don't pay compile cost.
echo ">>> Building pipeline_real (release, cuda) ..."
cargo build --release --features whisper-cuda --example pipeline_real -p lune-voice >/dev/null 2>&1

total=$(( ${#MARK_VALUES[@]} * ${#SKIP_VALUES[@]} * ${#MARGIN_VALUES[@]} ))
echo ">>> Sweeping ${total} combinations against ${FIXTURE} on ASR=${ASR} ..."
echo

i=0
for mark in "${MARK_VALUES[@]}"; do
  for skip in "${SKIP_VALUES[@]}"; do
    for margin in "${MARGIN_VALUES[@]}"; do
      i=$((i + 1))
      printf "[%3d/%d] mark=%-4s skip=%-4s margin=%-6s ... " \
        "$i" "$total" "$mark" "$skip" "$margin"

      out=$(LUNE_BENCH_FIXTURE="$FIXTURE" \
            LUNE_MARK_THRESHOLD="$mark" \
            LUNE_SKIP_THRESHOLD="$skip" \
            LUNE_RERANKER_MARGIN="$margin" \
            ASR="$ASR" \
            cargo run --release --features whisper-cuda --example pipeline_real -p lune-voice 2>&1)

      wer=$(printf '%s\n' "$out" | grep -m1 'WER:' | awk '{print $2}' | tr -d '%')
      if [[ -z "$wer" ]]; then
        wer="ERR"
      fi
      printf "WER=%s\n" "$wer"
      printf "%s\t%s\t%s\t%s\n" "$wer" "$mark" "$skip" "$margin" >> "$RESULTS"
    done
  done
done

echo
echo ">>> Sorted results (best WER first):"
echo
printf "  %-7s  %-6s  %-6s  %-7s\n" "WER%" "mark" "skip" "margin"
printf "  %-7s  %-6s  %-6s  %-7s\n" "-----" "----" "----" "------"
sort -k1,1g "$RESULTS" | awk -F'\t' '{ printf "  %-7s  %-6s  %-6s  %-7s\n", $1, $2, $3, $4 }'
