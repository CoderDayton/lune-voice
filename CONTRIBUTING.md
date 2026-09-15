# Contributing

Thanks for helping. Found a security problem? Don't open an issue for it; see
[SECURITY.md](SECURITY.md).

## Set up

Install the prerequisites listed under
[Building from source](README.md#building-from-source), then:

```bash
git clone https://github.com/CoderDayton/lune-voice
cd lune-voice
lefthook install   # runs the same gate as CI before each commit
cgs dev            # run the app with hot reload
```

`cgs show` lists every task. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains
the pipeline and the crate layout.

## Before you open a pull request

`cgs ci` runs fmt, clippy with `-D warnings`, `cargo check` and the Rust tests.
The pre-commit hook runs those plus the DTO drift check, biome and the frontend
tests.

- **Tests first.** A bug fix starts with a test that reproduces it.
- **DTOs.** After changing a Rust DTO, run `cargo test -p lune-ui` and commit
  the regenerated files in `desktop-ui/src/lib/backend/generated/`.
- **Strings.** Every user-facing frontend string goes through lingui. Run
  `bun run extract` in `desktop-ui/` after adding one.
- **Platforms.** Each OS-specific subsystem (hotkey, text delivery, focused
  window, pill overlay) sits behind one interface with a Linux and a Windows
  half. Run `cgs check-windows` when you touch Windows code from Linux.
- **New files** start with `SPDX-License-Identifier: GPL-3.0-or-later` in a
  comment. `scripts/dev/check-spdx.sh` enforces it in the pre-commit hook and CI.
- **Commits** use `type(scope): summary`, where type is one of feat, fix,
  update, refactor, perf, docs, test or chore.

## License

Lune Voice is GPL-3.0-or-later. By contributing, you agree that your
contribution is licensed under the same terms.
