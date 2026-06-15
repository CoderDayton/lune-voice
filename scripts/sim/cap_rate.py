"""Capitalization-rate simulation: build P(cap | word) from cased corpora.

The LLM merge over-capitalizes discourse words mid-sentence ("so make" ->
"So Make") yet must keep genuine proper nouns ("John"). Frequency alone cannot
separate them — `let's`=144K < `john`=240K in the case-folded table. The clean
signal is the per-word capitalization rate: P(word is capitalized | the word is
NOT at a sentence start). Proper nouns are ~always capitalized mid-sentence;
common words almost never are.

This script:
  1. tallies (word -> capitalized?) over cased, sentence-segmented corpora,
     EXCLUDING sentence-initial tokens (where everything is capitalized),
  2. validates the signal separates low (common) from high (proper) words,
  3. picks a keep-threshold tau by ROC over a labelled gold set,
  4. reports residual error on the homograph set (will/may/mark/rose),
  5. writes the full word -> p_cap table to scripts/sim/en_caprate.tsv.

Run: uv run --with nltk python3 scripts/sim/cap_rate.py
"""

import collections
import re
import sys
from pathlib import Path

import nltk

CORPORA = ["brown", "gutenberg", "reuters", "webtext", "inaugural"]


def ensure_corpora():
    for c in CORPORA + ["punkt", "punkt_tab"]:
        try:
            nltk.download(c, quiet=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] download {c}: {e}", file=sys.stderr)


# Alphabetic tokens only (allow internal apostrophe/hyphen): "let's", "well-known".
WORD = re.compile(r"^[A-Za-z][A-Za-z'’-]*$")


def tally():
    """Return (cap, tot) Counters over non-sentence-initial alphabetic tokens."""
    from nltk.corpus import brown, gutenberg, inaugural, reuters, webtext

    cap = collections.Counter()
    tot = collections.Counter()
    corpora = [
        ("brown", brown),
        ("gutenberg", gutenberg),
        ("reuters", reuters),
        ("webtext", webtext),
        ("inaugural", inaugural),
    ]
    for name, corpus in corpora:
        try:
            n = 0
            for sent in corpus.sents():
                # Skip index 0: the sentence-initial token is capitalized for
                # position, not lexical identity — it would poison every word.
                for i, tok in enumerate(sent):
                    if i == 0 or not WORD.match(tok):
                        continue
                    low = tok.lower()
                    tot[low] += 1
                    if tok[0].isupper():
                        cap[low] += 1
                    n += 1
            print(f"  {name:10} {n:>9,} mid-sentence tokens", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {name}: {e}", file=sys.stderr)
    return cap, tot


def build_pcap(cap, tot, min_count):
    return {w: cap[w] / tot[w] for w in tot if tot[w] >= min_count}


def show(pcap, cap, tot, label, words):
    print(f"\n{label}")
    for w in words:
        if w in pcap:
            print(f"  {w:12} p_cap={pcap[w]:.3f}   (cap {cap[w]:>6}/{tot[w]:<6})")
        else:
            seen = tot.get(w, 0)
            print(f"  {w:12} —          (only {seen} occurrence(s), below min)")


def roc(pcap, positives, negatives):
    """Sweep tau over candidate thresholds; report the one maximizing Youden's J
    (TPR - FPR). Positives = proper nouns (KEEP capital), negatives = common."""
    pos = [pcap[w] for w in positives if w in pcap]
    neg = [pcap[w] for w in negatives if w in pcap]
    if not pos or not neg:
        print("  [warn] ROC: insufficient labelled coverage", file=sys.stderr)
        return None
    best = None
    for tau in [i / 100 for i in range(5, 100, 5)]:
        tpr = sum(p >= tau for p in pos) / len(pos)
        fpr = sum(n >= tau for n in neg) / len(neg)
        j = tpr - fpr
        if best is None or j > best[1]:
            best = (tau, j, tpr, fpr)
    return best, pos, neg


GOLD_POSITIVE = [  # proper nouns / always-capitalized — KEEP the capital
    "john", "paris", "london", "america", "england", "monday", "tuesday",
    "english", "french", "god", "christmas", "europe", "africa", "james",
    "mary", "robert", "boston", "chicago", "rome", "berlin",
]
GOLD_NEGATIVE = [  # common/discourse — REVERT a mid-sentence capital
    "the", "and", "so", "make", "sure", "now", "just", "really", "very",
    "that", "this", "well", "good", "thing", "people", "let's",
    "okay", "yes", "because", "should",
]
HOMOGRAPHS = ["will", "may", "mark", "rose", "bill", "april", "frank",
              "faith", "grace", "hope", "art", "page", "brown", "white"]


def main():
    ensure_corpora()
    print("Tallying cased corpora (excluding sentence-initial tokens)...",
          file=sys.stderr)
    cap, tot = tally()
    min_count = 5
    pcap = build_pcap(cap, tot, min_count)
    total_tokens = sum(tot.values())
    print(f"\nvocab (count>={min_count}): {len(pcap):,}   "
          f"mid-sentence tokens: {total_tokens:,}")

    show(pcap, cap, tot, "LOW — common/discourse (want p_cap LOW -> revert):",
         GOLD_NEGATIVE)
    show(pcap, cap, tot, "HIGH — proper nouns (want p_cap HIGH -> keep):",
         GOLD_POSITIVE)
    show(pcap, cap, tot, "HOMOGRAPHS — ambiguous (the hard cases):", HOMOGRAPHS)

    result = roc(pcap, GOLD_POSITIVE, GOLD_NEGATIVE)
    if result:
        (tau, j, tpr, fpr), pos, neg = result
        print(f"\nROC over gold set ({len(pos)} proper, {len(neg)} common):")
        print(f"  best tau = {tau:.2f}   Youden J = {j:.3f}   "
              f"TPR(keep proper)={tpr:.2f}   FPR(keep common)={fpr:.2f}")
        # Confusion at tau on the homograph set.
        print(f"\n  homographs at tau={tau:.2f}:")
        for w in HOMOGRAPHS:
            if w in pcap:
                verdict = "KEEP cap" if pcap[w] >= tau else "revert"
                print(f"    {w:10} p_cap={pcap[w]:.3f} -> {verdict}")

    # Emit the full table for the Rust loader build step.
    out = Path(__file__).parent / "en_caprate.tsv"
    with out.open("w") as f:
        for w in sorted(pcap, key=lambda x: -tot[x]):
            f.write(f"{w}\t{pcap[w]:.4f}\t{tot[w]}\n")
    print(f"\nwrote {len(pcap):,} entries -> {out}")


if __name__ == "__main__":
    main()
