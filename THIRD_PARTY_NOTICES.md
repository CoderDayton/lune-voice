# Third-Party Notices

Lune Voice incorporates the following third-party components. Their licenses are
reproduced below as required for redistribution. This file ships with the
application binary.

The Rust crates compiled into the binary (Tauri and its plugins among them) and
the JavaScript packages in the interface bundle are listed, with their full
licence texts, in **THIRD_PARTY_LICENSES.md**, which ships alongside this file.
This file covers what that generated list cannot: native libraries, fonts, data
files, models and modified sources.

---

## whisper.cpp (and ggml)

Compiled into the application when the local Whisper speech-to-text backend is
enabled. Source: https://github.com/ggerganov/whisper.cpp

```
MIT License

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## whisper-rs

Rust bindings to whisper.cpp. Source: https://codeberg.org/tazz4843/whisper-rs

Released into the public domain under the Unlicense. No attribution required;
listed here for completeness.

**Modified.** `whisper-rs-sys` is vendored at `vendor/whisper-rs-sys` with two
changed files, both marked `LUNE FORK`: `build.rs` adds `GGML_BACKEND_DL`
support, and `src/lib.rs` adds a lint allow for the generated bindings. The
changes are distributed under Lune Voice's GPL-3.0-or-later licence; the
vendored whisper.cpp source and its MIT licence are unchanged.

## Whisper GGML models

The `ggml-*.bin` models downloaded on demand from
https://huggingface.co/ggerganov/whisper.cpp are conversions of OpenAI's Whisper
models, released by OpenAI under the MIT License (Copyright (c) 2022 OpenAI). The
models are downloaded by the user at runtime and are not bundled with the binary.

## Whisper VAD model (silero)

The `ggml-silero-*.bin` voice-activity-detection model downloaded on demand from
https://huggingface.co/ggml-org/whisper-vad is a conversion of the Silero VAD
model (MIT License, Copyright (c) Silero Team). Downloaded by the user at
runtime; not bundled with the binary.

## Word frequency lists (hermitdave/FrequencyWords)

The bundled `{lang}_freq.bin` artifacts are compiled from the
OpenSubtitles-derived word frequency lists at
https://github.com/hermitdave/FrequencyWords. Unlike the models above, these
are **compiled into the binary**, so the notice below is reproduced in full.

```
MIT License

Copyright (c) 2016 Hermit Dave

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Fonts: IBM Plex Sans and Manrope

The interface bundles webfont files for **IBM Plex Sans** (Copyright 2019 IBM
Corp.) and **Manrope** (Copyright 2019 The Manrope Project Authors,
https://github.com/sharanda/manrope), packaged by Fontsource as
`@fontsource/ibm-plex-sans` and `@fontsource-variable/manrope`.

Both are licensed under the **SIL Open Font License, Version 1.1**. The full
licence text travels with the fonts in THIRD_PARTY_LICENSES.md, under those two
packages, and is also available at https://openfontlicense.org. The fonts are
not sold on their own and are used under their original names.

## Supplemental vocabulary (`crates/lune-dict/data/vocab/*.txt`)

Compiled into the binary. Lists of company, product, platform, AI model and
library names and technical terms, written for Lune Voice and part of this
project under GPL-3.0-or-later. The names themselves belong to their owners and
are used only so the recognizer spells them correctly.

## Capitalization rates (`crates/lune-dict/data/en_caprate.bin`)

Compiled into the binary. A table of per-word capitalization rates (how often a
word is capitalized when it is not at the start of a sentence), computed by the
Lune Voice project from the Brown, Project Gutenberg, Reuters-21578, Web Text
and Inaugural Address corpora as distributed with NLTK
(https://www.nltk.org/nltk_data/). It contains derived counts only, no text
from those corpora.

## Wordnik word list (via the `wordnik_list` crate)

Compiled into the binary through the `wordnik_list` crate, whose code and word
list carry separate MIT licences.

```
MIT License

Copyright (c) 2024 Benjamin Cates

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```
MIT License

Copyright (c) 2020 Wordnik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## NVIDIA CUDA runtime libraries

The optional GPU plugin archive
(`lune-voice-gpu-cuda<major>-linux-x86_64.tar.gz`, downloaded by the user at
runtime and not bundled with the binary) contains `libcudart.so`,
`libcublas.so`, and `libcublasLt.so` from the NVIDIA CUDA Toolkit. These are
redistributed under the redistributable-software terms of the NVIDIA CUDA
Toolkit End User License Agreement:
https://docs.nvidia.com/cuda/eula/

The NVIDIA display driver, including `libcuda.so.1`, is **not** redistributed
and is not part of any Lune Voice artifact. It must be installed separately by
the user through their distribution or NVIDIA.

## COSMIC Wayland protocol definitions

`crates/lune-voice/protocols/cosmic-toplevel-info-unstable-v1.xml` and
`cosmic-workspace-unstable-v1.xml` are vendored from
https://github.com/pop-os/cosmic-protocols and compiled into the binary as
generated client bindings. They are used to read which window has focus on
cosmic-comp, which implements no wlr-foreign-toplevel equivalent.

Copyright © 2018, 2020 Ilia Bozhinov
Copyright © 2019 Christopher Billington
Copyright © 2020 Isaac Freund
Copyright © 2022, 2024 Victoria Brekenfeld

Permission to use, copy, modify, distribute, and sell this software and its
documentation for any purpose is hereby granted without fee, provided that the
above copyright notice appear in all copies and that both that copyright notice
and this permission notice appear in supporting documentation, and that the name
of the copyright holders not be used in advertising or publicity pertaining to
distribution of the software without specific, written prior permission. The
copyright holders make no representations about the suitability of this software
for any purpose. It is provided "as is" without express or implied warranty.

THE COPYRIGHT HOLDERS DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
