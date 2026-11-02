# Contributors

YantrikDB is primarily developed by [Pranab Sarkar](https://github.com/spranab) with substantive contributions from the wider community. This file records contributions that didn't render correctly in commit metadata or warrant standalone acknowledgment.

## External contributors

### Don Bowman ([@donbowman](https://github.com/donbowman), agilicus.com)

- **[PR #12](https://github.com/yantrikos/yantrikdb/pull/12) — CI workflows + SLSA build provenance** (2026-05-13). Authored `.github/workflows/lint.yml` (flake8 + pylint + rustfmt + clippy), `.github/workflows/test.yml` (`cargo test --workspace` across 5 OS/arch combos + slim `--no-default-features` job + pytest), and modifications to `.github/workflows/pypi.yml` (Sigstore/SLSA build provenance attestation via `actions/attest-build-provenance@v2`, plus switch from `PyO3/maturin-action@v1 upload` to `pypa/gh-action-pypi-publish@release/v1`). Substance landed as commit [0ffd361](https://github.com/yantrikos/yantrikdb/commit/0ffd361); the `Co-Authored-By` trailer in that commit was placed incorrectly (right after subject instead of as the final paragraph) so GitHub's UI didn't render Don as co-author. This file records the attribution explicitly.

- **[PR #11](https://github.com/yantrikos/yantrikdb/pull/11) — pyo3 0.23 → 0.28.3 driver** (2026-05-13). Filed the upstream driver that surfaced the Python 3.14 build gap on Ubuntu 26.04. The actual migration (closed PR + reopened internally as commit [702926b](https://github.com/yantrikos/yantrikdb/commit/702926b), released as v0.7.11) was authored by the maintainer, but Don's report + workaround steps + PR attempt established the priority and surface area.

- **Governance suggestion (2026-05-13).** Recommended branch protection + PR-required workflow on main, citing workflow pass/fail history, force-push prevention, audit trail, and supply-chain security framing. Adopted same session: branch protection now active on all yantrikos repos, `gh pr merge --auto --squash` is the standard workflow going forward.

### Hugo Gundlach ([@Huy3ko](https://github.com/Huy3ko))

- **[PR #256](https://github.com/yantrikos/yantrikdb/pull/256): `think()` triggers stay inside one namespace** (2026-10-27, released in v0.23.4). Found and fixed a cross-namespace disclosure. `check_redundancy` paired the most recent active memories across the whole store, so a store holding several namespaces could persist a `redundancy` or `potential_conflict` trigger in `trigger_log` carrying verbatim text from two of them. The fix scans per namespace and adds a final check before persistence that drops any trigger whose source records span namespaces. The PR includes a control test proving the same-namespace check still fires, and it names the out-of-scope gaps it measured. The commit landed as [27b11d7](https://github.com/yantrikos/yantrikdb/commit/27b11d7) under an author email not linked to a GitHub account, so GitHub doesn't credit it; this entry records the attribution.

### [@wolf-crazy](https://github.com/wolf-crazy)

- **[PR #254](https://github.com/yantrikos/yantrikdb/pull/254): a compression gist is never a cluster seed** (2026-10-23, released in v0.23.3). Scoped the fix for [#253](https://github.com/yantrikos/yantrikdb/issues/253) to seed selection only: a gist stays eligible as a member but can no longer seed a cluster. Applied it at both the consolidation and topic-mining call sites, with tests for each side of that rule.
- **[PR #252](https://github.com/yantrikos/yantrikdb/pull/252): bundled embedder cache location** (released in v0.23.3). The bundled model is cached under the platform cache dir, falling back to the temp dir when that can't be written.

### [@TianHengZhuang](https://github.com/TianHengZhuang)

Picked up all three good-first issues within hours of their being filed (2026-11-01):

- **[PR #263](https://github.com/yantrikos/yantrikdb/pull/263): `check_redundancy` honors its threshold** (fixes [#259](https://github.com/yantrikos/yantrikdb/issues/259)). The check ignored its argument and hardcoded 0.85, while `think()` passed the consolidation threshold. Adds `ThinkConfig.redundancy_sim_threshold`, defaulting to 0.85 so behavior is unchanged, with a test at the similarity boundary.
- **[PR #264](https://github.com/yantrikos/yantrikdb/pull/264): `reembed(name)` on the Python binding** (fixes [#260](https://github.com/yantrikos/yantrikdb/issues/260)). A populated store can now move to another named embedder from Python, for example `potion-multilingual-128M` for a non-English corpus, instead of starting fresh. Includes the build stub for builds without the `embedder-download` feature, and fixes the stale doc comment on `reembed()`.
- **[yantrikdb-hermes-plugin#103](https://github.com/yantrikos/yantrikdb-hermes-plugin/pull/103): `YANTRIKDB_COMPRESSION_GIST` switch** (fixes [plugin#102](https://github.com/yantrikos/yantrikdb-hermes-plugin/issues/102)). Lets users turn off the pre-compression gist write.

### Reporters

- **[antonius888](https://github.com/antonius888)**: filed [#253](https://github.com/yantrikos/yantrikdb/issues/253) (consolidation merging unrelated records on a Hermes store). The live repro, cluster-size histogram, trigger dump and SQL for inspecting `consolidated_into` pointed to both the gist seed problem fixed in v0.23.3 and the default embedder's behaviour on non-English text.
- **[alienos](https://github.com/alienos)** — filed [yantrikdb-hermes-plugin#1](https://github.com/yantrikos/yantrikdb-hermes-plugin/issues/1) (multilingual embedding support → engine v0.7.9 `potion-multilingual-128M` bundling) and [yantrikdb-hermes-plugin#4](https://github.com/yantrikos/yantrikdb-hermes-plugin/issues/4) (`has_embedder()` false-negative bug → engine v0.7.10 fix). Clear repros + evidence sections made diagnosis straightforward in both cases.

### Reviewers

- **[hakarrr](https://github.com/hakarrr)**: reviewed [PR #263](https://github.com/yantrikos/yantrikdb/pull/263) and caught two real problems before merge: a `cargo fmt` failure, and a new `ThinkConfig` field without a serde default, which would have broken deserializing older configs. Included a minimal serde reproduction.

## How attribution works going forward

The PR-required + auto-merge workflow set up 2026-05-13 means every change goes through a PR, and the squash-merge commit auto-formats `Co-Authored-By` trailers correctly at the end of the commit message. Hand-authored commits to main are no longer the path — branch protection blocks them. So the trailer-placement bug that necessitated this file shouldn't recur.

If your contribution should be recognized here and isn't, please open a PR adding the entry, or comment on the relevant issue/PR and the maintainer will add it.
