#!/usr/bin/env python3
"""
Corpus-derived GMS table generator.

Reads word-frequency lists and emits Rust constants:

    START: [f32; CLASS_COUNT]
    END: [f32; CLASS_COUNT]
    TRANS: [[f32; CLASS_COUNT]; CLASS_COUNT]

Supports one or more language tables.

Examples:

    python3 tools/build_gms_tables.py \\
      --input corpora/en.csv \\
      --lang en \\
      --name English \\
      --output src/gms_tables.rs

    python3 tools/build_gms_tables.py \\
      --input corpora/en.csv:en:English \\
      --input corpora/es.csv:es:Spanish \\
      --input corpora/fr.csv:fr:French \\
      --output src/gms_tables.rs

Input line formats:

    word
    word,frequency
    word<TAB>frequency
    frequency word

Recommended:
    --weight log
    --alpha 0.25
    --temperature 0.85
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


CLASS_NAMES = [
    "Vowel",
    "Stop",
    "Fricative",
    "Nasal",
    "Liquid",
    "Glide",
    "Rare",
    "Digit",
    "Separator",
    "Neutral",
]

CLASS_COUNT = len(CLASS_NAMES)

VOWEL = 0
STOP = 1
FRICATIVE = 2
NASAL = 3
LIQUID = 4
GLIDE = 5
RARE = 6
DIGIT = 7
SEPARATOR = 8
NEUTRAL = 9


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    name: str
    script: str
    vowels: frozenset[str]
    stops: frozenset[str]
    fricatives: frozenset[str]
    nasals: frozenset[str]
    liquids: frozenset[str]
    glides: frozenset[str]
    rare: frozenset[str]


@dataclass(frozen=True)
class InputSpec:
    path: Path
    lang: str
    name: str


@dataclass
class Counts:
    start: list[float]
    end: list[float]
    trans: list[list[float]]
    class_prior: list[float]
    usable_words: int = 0
    skipped_words: int = 0
    weighted_words: float = 0.0
    transitions: int = 0


@dataclass(frozen=True)
class Tables:
    start: list[float]
    end: list[float]
    trans: list[list[float]]
    class_prior: list[float]
    usable_words: int
    skipped_words: int
    weighted_words: float
    transitions: int


LANGUAGE_PROFILES: dict[str, LanguageProfile] = {
    "en": LanguageProfile(
        code="en",
        name="English",
        script="latin",
        vowels=frozenset("aeiouy"),
        stops=frozenset("bcdgkpt"),
        fricatives=frozenset("fhsvz"),
        nasals=frozenset("mn"),
        liquids=frozenset("lr"),
        glides=frozenset("wj"),
        rare=frozenset("qx"),
    ),
    "es": LanguageProfile(
        code="es",
        name="Spanish",
        script="latin",
        vowels=frozenset("aeiouáéíóúü"),
        stops=frozenset("bcdgkpt"),
        fricatives=frozenset("fhsvzjx"),
        nasals=frozenset("mnñ"),
        liquids=frozenset("lr"),
        glides=frozenset("yw"),
        rare=frozenset("q"),
    ),
    "fr": LanguageProfile(
        code="fr",
        name="French",
        script="latin",
        vowels=frozenset("aeiouyàâæéèêëîïôœùûü"),
        stops=frozenset("bcdgkpt"),
        fricatives=frozenset("fhsvzjç"),
        nasals=frozenset("mn"),
        liquids=frozenset("lr"),
        glides=frozenset("wy"),
        rare=frozenset("qx"),
    ),
    "de": LanguageProfile(
        code="de",
        name="German",
        script="latin",
        vowels=frozenset("aeiouyäöü"),
        stops=frozenset("bcdgkpt"),
        fricatives=frozenset("fhsvzß"),
        nasals=frozenset("mn"),
        liquids=frozenset("lr"),
        glides=frozenset("wj"),
        rare=frozenset("qx"),
    ),
    "it": LanguageProfile(
        code="it",
        name="Italian",
        script="latin",
        vowels=frozenset("aeiouàèéìíòóù"),
        stops=frozenset("bcdgkpt"),
        fricatives=frozenset("fhsvz"),
        nasals=frozenset("mn"),
        liquids=frozenset("lr"),
        glides=frozenset("yjw"),
        rare=frozenset("qx"),
    ),
    "pt": LanguageProfile(
        code="pt",
        name="Portuguese",
        script="latin",
        vowels=frozenset("aeiouáâãàéêíóôõúü"),
        stops=frozenset("bcdgkpt"),
        fricatives=frozenset("fhsvzjxç"),
        nasals=frozenset("mn"),
        liquids=frozenset("lr"),
        glides=frozenset("yw"),
        rare=frozenset("q"),
    ),
    "nl": LanguageProfile(
        code="nl",
        name="Dutch",
        script="latin",
        vowels=frozenset("aeiouyáéíóúëï"),
        stops=frozenset("bcdgkpt"),
        fricatives=frozenset("fhsvzj"),
        nasals=frozenset("mn"),
        liquids=frozenset("lr"),
        glides=frozenset("w"),
        rare=frozenset("qx"),
    ),
    "sv": LanguageProfile(
        code="sv",
        name="Swedish",
        script="latin",
        vowels=frozenset("aeiouyåäö"),
        stops=frozenset("bcdgkpt"),
        fricatives=frozenset("fhsvzj"),
        nasals=frozenset("mn"),
        liquids=frozenset("lr"),
        glides=frozenset("w"),
        rare=frozenset("qx"),
    ),
    "no": LanguageProfile(
        code="no",
        name="Norwegian",
        script="latin",
        vowels=frozenset("aeiouyåæø"),
        stops=frozenset("bcdgkpt"),
        fricatives=frozenset("fhsvzj"),
        nasals=frozenset("mn"),
        liquids=frozenset("lr"),
        glides=frozenset("w"),
        rare=frozenset("qx"),
    ),
    "da": LanguageProfile(
        code="da",
        name="Danish",
        script="latin",
        vowels=frozenset("aeiouyåæø"),
        stops=frozenset("bcdgkpt"),
        fricatives=frozenset("fhsvzj"),
        nasals=frozenset("mn"),
        liquids=frozenset("lr"),
        glides=frozenset("w"),
        rare=frozenset("qx"),
    ),
    "pl": LanguageProfile(
        code="pl",
        name="Polish",
        script="latin",
        vowels=frozenset("aeiouyąęó"),
        stops=frozenset("bcdgkpt"),
        fricatives=frozenset("fhsvzjśźżłćń"),
        nasals=frozenset("mn"),
        liquids=frozenset("lr"),
        glides=frozenset("w"),
        rare=frozenset("qx"),
    ),
    "ru": LanguageProfile(
        code="ru",
        name="Russian",
        script="cyrillic",
        vowels=frozenset(),
        stops=frozenset(),
        fricatives=frozenset(),
        nasals=frozenset(),
        liquids=frozenset(),
        glides=frozenset(),
        rare=frozenset(),
    ),
    "uk": LanguageProfile(
        code="uk",
        name="Ukrainian",
        script="cyrillic",
        vowels=frozenset(),
        stops=frozenset(),
        fricatives=frozenset(),
        nasals=frozenset(),
        liquids=frozenset(),
        glides=frozenset(),
        rare=frozenset(),
    ),
}



def main() -> None:
    args = parse_args()

    specs = parse_input_specs(args)
    generated: list[tuple[InputSpec, Tables]] = []

    for spec in specs:
        profile = profile_for(spec.lang, spec.name)
        counts = count_file(
            path=spec.path,
            profile=profile,
            weight_mode=args.weight,
            min_len=args.min_len,
            max_len=args.max_len,
            max_words=args.max_words,
        )

        if counts.usable_words < args.min_words:
            print(
                f"[skip] {spec.lang}: only {counts.usable_words} usable words "
                f"(< --min-words {args.min_words})",
                file=sys.stderr,
            )
            continue

        tables = counts_to_tables(
            counts=counts,
            alpha=args.alpha,
            temperature=args.temperature,
            floor=args.floor,
        )
        generated.append((spec, tables))

        print(
            f"[ok] {spec.lang:<3} words={tables.usable_words} "
            f"skipped={tables.skipped_words} weighted={tables.weighted_words:.2f} "
            f"transitions={tables.transitions}",
            file=sys.stderr,
        )

    if not generated:
        raise SystemExit("no tables generated")

    output = emit_rust(
        generated=generated,
        alpha=args.alpha,
        temperature=args.temperature,
        floor=args.floor,
        weight=args.weight,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build corpus-derived Rust GMS transition tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "Input file. Either PATH with --lang/--name, or PATH:LANG:NAME. "
            "May be repeated."
        ),
    )
    parser.add_argument("--lang", help="Language code for single --input PATH mode.")
    parser.add_argument("--name", help="Display name for single --input PATH mode.")
    parser.add_argument("--output", type=Path, help="Output Rust file. Defaults to stdout.")

    parser.add_argument("--alpha", type=float, default=0.25, help="Dirichlet smoothing alpha.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.85,
        help="Probability temperature. <1 sharpens, >1 flattens.",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=1e-5,
        help="Minimum emitted probability before final renormalization.",
    )
    parser.add_argument(
        "--weight",
        choices=("log", "sqrt", "raw", "one"),
        default="log",
        help="How to convert corpus frequency into count weight.",
    )
    parser.add_argument("--min-words", type=int, default=100)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument(
        "--max-words",
        type=int,
        default=0,
        help="Optional cap on usable words per file. 0 means unlimited.",
    )

    args = parser.parse_args()

    if args.alpha <= 0.0 or not math.isfinite(args.alpha):
        raise SystemExit("--alpha must be positive and finite")
    if args.temperature <= 0.0 or not math.isfinite(args.temperature):
        raise SystemExit("--temperature must be positive and finite")
    if not (0.0 <= args.floor < 1.0):
        raise SystemExit("--floor must be in [0, 1)")
    if args.min_words < 1:
        raise SystemExit("--min-words must be >= 1")
    if args.min_len < 1:
        raise SystemExit("--min-len must be >= 1")
    if args.max_len < args.min_len:
        raise SystemExit("--max-len must be >= --min-len")

    return args



def parse_input_specs(args: argparse.Namespace) -> list[InputSpec]:
    specs: list[InputSpec] = []

    if len(args.input) == 1 and ":" not in args.input[0]:
        if not args.lang:
            raise SystemExit("--lang is required when --input is a plain path")
        lang = args.lang.lower()
        name = args.name or LANGUAGE_PROFILES.get(lang, default_profile(lang, lang)).name
        specs.append(InputSpec(path=Path(args.input[0]), lang=lang, name=name))
        return specs

    for raw in args.input:
        parts = raw.split(":", 2)
        if len(parts) == 1:
            if not args.lang:
                raise SystemExit(
                    "repeated or multi-language --input requires PATH:LANG:NAME format"
                )
            lang = args.lang.lower()
            name = args.name or LANGUAGE_PROFILES.get(lang, default_profile(lang, lang)).name
            specs.append(InputSpec(path=Path(parts[0]), lang=lang, name=name))
        elif len(parts) == 2:
            path, lang = parts
            lang = lang.lower()
            name = LANGUAGE_PROFILES.get(lang, default_profile(lang, lang)).name
            specs.append(InputSpec(path=Path(path), lang=lang, name=name))
        else:
            path, lang, name = parts
            specs.append(InputSpec(path=Path(path), lang=lang.lower(), name=name))

    return specs


def profile_for(lang: str, name: str) -> LanguageProfile:
    if lang in LANGUAGE_PROFILES:
        base = LANGUAGE_PROFILES[lang]
        return LanguageProfile(
            code=base.code,
            name=name or base.name,
            script=base.script,
            vowels=base.vowels,
            stops=base.stops,
            fricatives=base.fricatives,
            nasals=base.nasals,
            liquids=base.liquids,
            glides=base.glides,
            rare=base.rare,
        )

    return default_profile(lang, name)


def default_profile(lang: str, name: str) -> LanguageProfile:
    return LanguageProfile(
        code=lang,
        name=name,
        script="latin",
        vowels=frozenset("aeiouy"),
        stops=frozenset("bcdgkpt"),
        fricatives=frozenset("fhsvz"),
        nasals=frozenset("mn"),
        liquids=frozenset("lr"),
        glides=frozenset("wj"),
        rare=frozenset("qx"),
    )


def count_file(
    path: Path,
    profile: LanguageProfile,
    weight_mode: str,
    min_len: int,
    max_len: int,
    max_words: int,
) -> Counts:
    if not path.is_file():
        raise SystemExit(f"input file does not exist: {path}")

    counts = Counts(
        start=[0.0] * CLASS_COUNT,
        end=[0.0] * CLASS_COUNT,
        trans=[[0.0] * CLASS_COUNT for _ in range(CLASS_COUNT)],
        class_prior=[0.0] * CLASS_COUNT,
    )

    for word, freq in read_word_frequency_lines(path):
        normalized = normalize_word(word)
        if normalized is None:
            counts.skipped_words += 1
            continue

        if not (min_len <= char_len(normalized) <= max_len):
            counts.skipped_words += 1
            continue

        if not compatible_with_profile(normalized, profile):
            counts.skipped_words += 1
            continue

        weight = frequency_weight(freq, weight_mode)
        observe_word(normalized, weight, profile, counts)

        if max_words > 0 and counts.usable_words >= max_words:
            break

    return counts


def read_word_frequency_lines(path: Path) -> Iterator[tuple[str, float]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        if "\t" in sample:
            delimiter = "\t"
        elif "," in sample:
            delimiter = ","
        else:
            delimiter = None

        if delimiter:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                parsed = parse_row(row)
                if parsed is not None:
                    yield parsed
        else:
            for line in f:
                parsed = parse_plain_line(line)
                if parsed is not None:
                    yield parsed


def parse_row(row: Sequence[str]) -> tuple[str, float] | None:
    if not row:
        return None

    row = [cell.strip() for cell in row]
    if not row[0] or row[0].startswith("#"):
        return None

    if len(row) >= 2 and looks_numeric(row[0]) and not looks_numeric(row[1]):
        return row[1], parse_freq(row[0])

    word = row[0]
    freq = parse_freq(row[1]) if len(row) >= 2 and looks_numeric(row[1]) else 1.0
    return word, freq


def parse_plain_line(line: str) -> tuple[str, float] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split()
    if len(parts) >= 2 and looks_numeric(parts[0]):
        return parts[1], parse_freq(parts[0])
    if len(parts) >= 2 and looks_numeric(parts[1]):
        return parts[0], parse_freq(parts[1])
    return parts[0], 1.0


def looks_numeric(value: str) -> bool:
    value = value.strip().replace("_", "").replace(",", "")
    try:
        float(value)
        return True
    except ValueError:
        return False


def parse_freq(value: str) -> float:
    value = value.strip().replace("_", "").replace(",", "")
    try:
        freq = float(value)
    except ValueError:
        return 1.0

    if not math.isfinite(freq) or freq <= 0.0:
        return 1.0
    return freq



def normalize_word(word: str) -> str | None:
    word = unicodedata.normalize("NFKC", word.strip().lower())
    word = word.strip("\"\"''`.,;:!?()[]{}<>«»")

    if not word:
        return None

    out: list[str] = []
    for ch in word:
        category = unicodedata.category(ch)
        if category.startswith("C"):
            continue
        if ch.isspace():
            continue
        if ch.isalpha() or ch.isdigit() or ch in {"'", "’", "-", "_", "."}:
            out.append(ch)

    if not out:
        return None

    return "".join(out)


def char_len(word: str) -> int:
    return sum(1 for _ in word)


def compatible_with_profile(word: str, profile: LanguageProfile) -> bool:
    alpha = 0
    compatible = 0

    for ch in word:
        if not ch.isalpha():
            continue

        alpha += 1

        if profile.script == "latin":
            ok = is_latin(ch)
        elif profile.script == "cyrillic":
            ok = is_cyrillic(ch)
        else:
            ok = True

        if ok:
            compatible += 1

    return alpha > 0 and compatible / alpha >= 0.80


def observe_word(
    word: str,
    weight: float,
    profile: LanguageProfile,
    counts: Counts,
) -> None:
    classes = [classify_char(ch, profile) for ch in word]
    if not classes:
        counts.skipped_words += 1
        return

    counts.usable_words += 1
    counts.weighted_words += weight

    counts.start[classes[0]] += weight
    counts.end[classes[-1]] += weight

    for cls in classes:
        counts.class_prior[cls] += weight

    for a, b in zip(classes, classes[1:]):
        counts.trans[a][b] += weight
        counts.transitions += 1


def classify_char(ch: str, profile: LanguageProfile) -> int:
    ch = ch.lower()

    if ch.isdigit():
        return DIGIT
    if ch in {"'", "’", "`", "-", "_", "."}:
        return SEPARATOR

    if profile.script == "cyrillic":
        return classify_cyrillic(ch)

    if ch in profile.vowels:
        return VOWEL
    if ch in profile.stops:
        return STOP
    if ch in profile.fricatives:
        return FRICATIVE
    if ch in profile.nasals:
        return NASAL
    if ch in profile.liquids:
        return LIQUID
    if ch in profile.glides:
        return GLIDE
    if ch in profile.rare:
        return RARE

    if ch.isalpha():
        return NEUTRAL
    return SEPARATOR


def classify_cyrillic(ch: str) -> int:
    if ch in "аеёиіїоөүуыэюяє":
        return VOWEL
    if ch in "бгґдкптц":
        return STOP
    if ch in "вжзсфхшщч":
        return FRICATIVE
    if ch in "мн":
        return NASAL
    if ch in "лр":
        return LIQUID
    if ch == "й":
        return GLIDE
    if ch in "ъь":
        return VOWEL
    if ch.isalpha():
        return NEUTRAL
    return SEPARATOR


def frequency_weight(freq: float, mode: str) -> float:
    if mode == "raw":
        return max(freq, 1.0)
    if mode == "sqrt":
        return max(math.sqrt(freq), 1.0)
    if mode == "one":
        return 1.0
    if mode == "log":
        return max(math.log1p(freq), 1.0)
    raise ValueError(f"unknown weight mode: {mode}")


def counts_to_tables(
    counts: Counts,
    alpha: float,
    temperature: float,
    floor: float,
) -> Tables:
    return Tables(
        start=normalize_distribution(counts.start, alpha, temperature, floor),
        end=normalize_distribution(counts.end, alpha, temperature, floor),
        trans=[
            normalize_distribution(row, alpha, temperature, floor)
            for row in counts.trans
        ],
        class_prior=normalize_distribution(counts.class_prior, alpha, temperature, floor),
        usable_words=counts.usable_words,
        skipped_words=counts.skipped_words,
        weighted_words=counts.weighted_words,
        transitions=counts.transitions,
    )


def normalize_distribution(
    raw_counts: Sequence[float],
    alpha: float,
    temperature: float,
    floor: float,
) -> list[float]:
    smoothed = [max(c, 0.0) + alpha for c in raw_counts]
    total = sum(smoothed)

    probs = [c / total for c in smoothed]

    if temperature != 1.0:
        probs = [p ** (1.0 / temperature) for p in probs]
        total = sum(probs)
        probs = [p / total for p in probs]

    if floor > 0.0:
        probs = [max(p, floor) for p in probs]
        total = sum(probs)
        probs = [p / total for p in probs]

    return probs



def emit_rust(
    generated: list[tuple[InputSpec, Tables]],
    alpha: float,
    temperature: float,
    floor: float,
    weight: str,
) -> str:
    lines: list[str] = []

    lines.append("//! @generated by scripts/build_gms_tables.py")
    lines.append("//!")
    lines.append("//! Do not edit by hand.")
    lines.append("//!")
    lines.append(f"//! alpha: {alpha}")
    lines.append(f"//! temperature: {temperature}")
    lines.append(f"//! floor: {floor}")
    lines.append(f"//! weight: {weight}")
    lines.append("")
    lines.append(f"pub const CLASS_COUNT: usize = {CLASS_COUNT};")
    lines.append("")
    lines.append("#[derive(Clone, Copy, Debug, PartialEq, Eq)]")
    lines.append("pub enum GmsLanguage {")
    for spec, _ in generated:
        lines.append(f"    {rust_variant(spec.lang, spec.name)},")
    lines.append("}")
    lines.append("")
    lines.append("#[derive(Clone, Copy, Debug)]")
    lines.append("pub struct GmsTables {")
    lines.append("    pub code: &'static str,")
    lines.append("    pub name: &'static str,")
    lines.append("    pub start: &'static [f32; CLASS_COUNT],")
    lines.append("    pub end: &'static [f32; CLASS_COUNT],")
    lines.append("    pub trans: &'static [[f32; CLASS_COUNT]; CLASS_COUNT],")
    lines.append("    pub class_prior: &'static [f32; CLASS_COUNT],")
    lines.append("    pub usable_words: usize,")
    lines.append("    pub skipped_words: usize,")
    lines.append("    pub weighted_words: f64,")
    lines.append("    pub transitions: usize,")
    lines.append("}")
    lines.append("")

    for spec, tables in generated:
        prefix = rust_const_prefix(spec.lang)
        lines.extend(emit_table_constants(prefix, spec, tables))
        lines.append("")

    lines.append("pub static GMS_TABLES: &[GmsTables] = &[")
    for spec, tables in generated:
        prefix = rust_const_prefix(spec.lang)
        lines.append("    GmsTables {")
        lines.append(f'        code: "{escape_rust_string(spec.lang)}",')
        lines.append(f'        name: "{escape_rust_string(spec.name)}",')
        lines.append(f"        start: &{prefix}_START,")
        lines.append(f"        end: &{prefix}_END,")
        lines.append(f"        trans: &{prefix}_TRANS,")
        lines.append(f"        class_prior: &{prefix}_CLASS_PRIOR,")
        lines.append(f"        usable_words: {tables.usable_words},")
        lines.append(f"        skipped_words: {tables.skipped_words},")
        lines.append(f"        weighted_words: {tables.weighted_words:.6}_f64,")
        lines.append(f"        transitions: {tables.transitions},")
        lines.append("    },")
    lines.append("];")
    lines.append("")
    lines.append("pub fn tables_for_language(language: GmsLanguage) -> &'static GmsTables {")
    lines.append("    match language {")
    for index, (spec, _) in enumerate(generated):
        lines.append(
            f"        GmsLanguage::{rust_variant(spec.lang, spec.name)} => &GMS_TABLES[{index}],"
        )
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append("pub fn tables_for_code(code: &str) -> Option<&'static GmsTables> {")
    lines.append("    GMS_TABLES")
    lines.append("        .iter()")
    lines.append("        .find(|tables| tables.code.eq_ignore_ascii_case(code))")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def emit_table_constants(prefix: str, spec: InputSpec, tables: Tables) -> list[str]:
    return [
        f"// {spec.name} ({spec.lang})",
        f"pub const {prefix}_START: [f32; CLASS_COUNT] = {format_array(tables.start)};",
        f"pub const {prefix}_END: [f32; CLASS_COUNT] = {format_array(tables.end)};",
        f"pub const {prefix}_CLASS_PRIOR: [f32; CLASS_COUNT] = {format_array(tables.class_prior)};",
        f"pub const {prefix}_TRANS: [[f32; CLASS_COUNT]; CLASS_COUNT] = [",
        *[f"    {format_array(row)}," for row in tables.trans],
        "];",
    ]


def format_array(values: Sequence[float]) -> str:
    body = ", ".join(f"{v:.8}_f32" for v in values)
    return f"[{body}]"


def rust_const_prefix(lang: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", lang.upper()).strip("_")
    if not cleaned:
        return "LANG"
    if cleaned[0].isdigit():
        return f"LANG_{cleaned}"
    return cleaned


def rust_variant(lang: str, name: str) -> str:
    base = name.strip() or lang
    pieces = re.split(r"[^A-Za-z0-9]+", base)
    variant = "".join(piece[:1].upper() + piece[1:] for piece in pieces if piece)
    if not variant:
        variant = rust_const_prefix(lang).title().replace("_", "")
    if variant[0].isdigit():
        variant = f"Lang{variant}"
    return variant


def escape_rust_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def is_latin(ch: str) -> bool:
    code = ord(ch)
    return 0x0041 <= code <= 0x024F or ch in {
        "æ", "œ", "ç", "ñ", "ß", "þ", "ð", "ł", "ś", "ź", "ż", "ć", "ń", "ą", "ę",
    }


def is_cyrillic(ch: str) -> bool:
    code = ord(ch)
    return 0x0400 <= code <= 0x052F


if __name__ == "__main__":
    main()
