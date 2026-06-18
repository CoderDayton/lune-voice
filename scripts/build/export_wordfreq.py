#!/usr/bin/env python3
"""
Export multilingual word-frequency CSVs for GMS table generation.

Creates:
  corpora/en.csv
  corpora/fr.csv
  corpora/de.csv
  corpora/es.csv
  corpora/it.csv
  corpora/zh.csv
  corpora/cn.csv
  corpora/ja.csv
  corpora/ko.csv
  corpora/pt.csv
  corpora/ru.csv

Each CSV has:
  word,frequency

Dependency:
  uv pip install wordfreq

Notes:
  - wordfreq frequencies are Zipf-scale estimates.
  - This script converts Zipf frequency to an integer-ish relative frequency.
  - The absolute scale does not matter for your transition-table builder;
    only relative weights matter.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

from wordfreq import available_languages, iter_wordlist, zipf_frequency


LANGUAGES: list[tuple[str, str, int]] = [
    ("en", "English", 50_000),
    ("fr", "French", 50_000),
    ("de", "German", 50_000),
    ("es", "Spanish", 50_000),
    ("it", "Italian", 50_000),
    ("zh", "Chinese", 100_000),
    ("ja", "Japanese", 100_000),
    ("ko", "Korean", 100_000),
    ("pt", "Portuguese", 50_000),
    ("ru", "Russian", 50_000),
]

TOKEN_RE = re.compile(r"^[\w.''\-]+$", re.UNICODE)


def main() -> None:
    args = parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    supported = set(available_languages())

    for output_lang, display_name, lang_top_n in LANGUAGES:
        source_lang = output_lang

        if source_lang not in supported:
            print(
                f"[skip] {output_lang} ({display_name}): wordfreq does not support {source_lang}",
                file=sys.stderr,
            )
            continue

        rows = collect_rows(
            lang=source_lang,
            top_n=lang_top_n if args.top_n == 50_000 else args.top_n,
            min_zipf=args.min_zipf,
            min_len=args.min_len,
            max_len=args.max_len,
            include_numbers=args.include_numbers,
        )

        if len(rows) < args.min_rows:
            print(
                f"[warn] {output_lang}: only {len(rows)} rows after filtering",
                file=sys.stderr,
            )

        path = out_dir / f"{output_lang}.csv"
        write_csv(path, rows)

        print(
            f"[ok] {output_lang:<2} <- {source_lang:<2} "
            f"{display_name:<10} rows={len(rows):>6} path={path}",
            file=sys.stderr,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export wordfreq word,frequency CSVs for GMS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("corpora"),
        help="Output directory.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50_000,
        help="Max rows per language after filtering. Per-language defaults apply if unchanged.",
    )
    parser.add_argument(
        "--min-zipf",
        type=float,
        default=1.0,
        help="Minimum Zipf frequency to keep.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=1,
        help="Minimum Unicode character length.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=64,
        help="Maximum Unicode character length.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1_000,
        help="Warn if a generated CSV has fewer rows than this.",
    )
    parser.add_argument(
        "--include-numbers",
        action="store_true",
        help="Keep pure numeric tokens. Default filters them out.",
    )

    args = parser.parse_args()

    if args.top_n <= 0:
        raise SystemExit("--top-n must be positive")
    if args.min_len <= 0:
        raise SystemExit("--min-len must be positive")
    if args.max_len < args.min_len:
        raise SystemExit("--max-len must be >= --min-len")

    return args


def collect_rows(
    lang: str,
    top_n: int,
    min_zipf: float,
    min_len: int,
    max_len: int,
    include_numbers: bool,
) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    seen: set[str] = set()

    for raw_word in iter_wordlist(lang):
        word = normalize_token(raw_word)

        if word is None:
            continue

        if word in seen:
            continue

        if not include_numbers and word.isnumeric():
            continue

        length = len(word)
        if length < min_len or length > max_len:
            continue

        if not token_is_useful(word):
            continue

        zipf = zipf_frequency(word, lang)
        if zipf < min_zipf:
            continue

        frequency = zipf_to_relative_frequency(zipf)

        rows.append((word, frequency))
        seen.add(word)

        if len(rows) >= top_n:
            break

    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def normalize_token(token: str) -> str | None:
    token = unicodedata.normalize("NFKC", token.strip().lower())
    token = token.strip("\"“”‘’`.,;:!?()[]{}<>«»、。！？；：（）［］【】")

    if not token:
        return None

    token = "".join(ch for ch in token if not unicodedata.category(ch).startswith("C"))

    if not token:
        return None

    return token


def token_is_useful(token: str) -> bool:
    if not TOKEN_RE.match(token):
        return False

    if not any(ch.isalpha() for ch in token):
        return False

    if any(ch.isspace() for ch in token):
        return False

    return True


def zipf_to_relative_frequency(zipf: float) -> int:
    value = 10 ** max(zipf, 0.0)
    return max(1, int(round(value)))


def write_csv(path: Path, rows: list[tuple[str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
