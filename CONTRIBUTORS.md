# Contributors

YantrikDB is primarily developed by [Pranab Sarkar](https://github.com/spranab) with substantive contributions from the wider community. This file records contributions that didn't render correctly in commit metadata or warrant standalone acknowledgment.

## External contributors

### Don Bowman ([@donbowman](https://github.com/donbowman), agilicus.com)

- **[PR #12](https://github.com/yantrikos/yantrikdb/pull/12) — CI workflows + SLSA build provenance** (2026-05-13). Authored `.github/workflows/lint.yml` (flake8 + pylint + rustfmt + clippy), `.github/workflows/test.yml` (`cargo test --workspace` across 5 OS/arch combos + slim `--no-default-features` job + pytest), and modifications to `.github/workflows/pypi.yml` (Sigstore/SLSA build provenance attestation via `actions/attest-build-provenance@v2`, plus switch from `PyO3/maturin-action@v1 upload` to `pypa/gh-action-pypi-publish@release/v1`). Substance landed as commit [0ffd361](https://github.com/yantrikos/yantrikdb/commit/0ffd361); the `Co-Authored-By` trailer in that commit was placed incorrectly (right after subject instead of as the final paragraph) so GitHub's UI didn't render Don as co-author. This file records the attribution explicitly.

- **[PR #11](https://github.com/yantrikos/yantrikdb/pull/11) — pyo3 0.23 → 0.28.3 driver** (2026-05-13). Filed the upstream driver that surfaced the Python 3.14 build gap on Ubuntu 26.04. The actual migration (closed PR + reopened internally as commit [702926b](https://github.com/yantrikos/yantrikdb/commit/702926b), released as v0.7.11) was authored by the maintainer, but Don's report + workaround steps + PR attempt established the priority and surface area.

- **Governance suggestion (2026-05-13).** Recommended branch protection + PR-required workflow on main, citing workflow pass/fail history, force-push prevention, audit trail, and supply-chain security framing. Adopted same session: branch protection now active on all yantrikos repos, `gh pr merge --auto --squash` is the standard workflow going forward.

### Reporters

- **[alienos](https://github.com/alienos)** — filed [yantrikdb-hermes-plugin#1](https://github.com/yantrikos/yantrikdb-hermes-plugin/issues/1) (multilingual embedding support → engine v0.7.9 `potion-multilingual-128M` bundling) and [yantrikdb-hermes-plugin#4](https://github.com/yantrikos/yantrikdb-hermes-plugin/issues/4) (`has_embedder()` false-negative bug → engine v0.7.10 fix). Clear repros + evidence sections made diagnosis straightforward in both cases.

## How attribution works going forward

The PR-required + auto-merge workflow set up 2026-05-13 means every change goes through a PR, and the squash-merge commit auto-formats `Co-Authored-By` trailers correctly at the end of the commit message. Hand-authored commits to main are no longer the path — branch protection blocks them. So the trailer-placement bug that necessitated this file shouldn't recur.

If your contribution should be recognized here and isn't, please open a PR adding the entry, or comment on the relevant issue/PR and the maintainer will add it.
