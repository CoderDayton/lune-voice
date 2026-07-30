# Third-Party Notices

Lune Voice incorporates the following third-party components. Their licenses are
reproduced below as required for redistribution. This file ships with the
application binary.

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
