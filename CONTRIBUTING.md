# Contributing to YantrikDB

YantrikDB is the Rust cognitive-memory engine behind
[yantrikdb-server](https://github.com/yantrikos/yantrikdb-server) and
[yantrikdb-mcp](https://github.com/yantrikos/yantrikdb-mcp). Bug reports,
failing-test repros and PRs are all welcome.

## Licensing and copyright

The project is **Apache-2.0** (relicensed from AGPL-3.0 on 2026-08-18; every
version published under Apache-2.0 stays available under it permanently).

There is **no CLA and no copyright assignment**. Apache-2.0 section 5 applies:
unless you state otherwise in the PR, your contribution is licensed under
Apache-2.0 and you keep your copyright. A patent has been filed on part of this
work; Apache-2.0 section 3 grants patent rights for the licensed material
explicitly, which is the reason for Apache over MIT.

## Getting the code building

```sh
git clone https://github.com/yantrikos/yantrikdb
cd yantrikdb
cargo test --workspace
```

A default `cargo build` has **no network code path**: the `embedder-download`
feature is off for Rust consumers, so `with_default()` uses the bundled 64-dim
`potion-base-2M`. The Python wheel turns it on, which is why pip users get the
256-dim `potion-base-8M` default. If you are touching embedder code, build both:

```sh
cargo build --features embedder-download
cargo build --no-default-features        # slim: no bundled embedder at all
```

Python bindings live in `crates/yantrikdb-python` (pyo3 + maturin). The wasm
build used by the in-browser demo is `crates/yantrikdb-wasm`.

## What CI will check

Both workflows must pass before a PR can merge:

- `.github/workflows/lint.yml` — `cargo fmt --all -- --check`,
  `cargo clippy --all-targets --all-features --workspace --exclude
  yantrikdb-python -- --cap-lints warn`, plus flake8 and pylint on the Python
  side.
- `.github/workflows/test.yml` — `cargo test --workspace` across five OS/arch
  combinations, a `--no-default-features` slim job, and pytest.

Run `cargo fmt --all` and the clippy line above locally first; they are the two
that fail most often.

## Workflow

`main` is branch-protected, so every change lands through a pull request and is
squash-merged. Please:

- keep one logical change per PR;
- add or extend a test that fails without your change — the engine has a lot of
  behaviour that only shows up under concurrency or after temporal decay, and
  `CONCURRENCY.md` documents the invariants that are easy to break by accident;
- describe what you observed, not only what you changed.

If you are picking up an existing issue, say so on the issue first so two people
don't write the same patch.

## Issue difficulty labels

Issues tagged for outside contributors carry a difficulty label —
`difficulty: easy`, `difficulty: medium`, `difficulty: hard` or
`difficulty: tedious` — describing how much engine context you need before the
change makes sense, not how much code it takes.

## Attribution

External contributions are recorded in
[CONTRIBUTORS.md](CONTRIBUTORS.md). Squash-merge writes the `Co-Authored-By`
trailer correctly; if your contribution should be listed there and isn't, open a
PR adding it or say so on the issue.
