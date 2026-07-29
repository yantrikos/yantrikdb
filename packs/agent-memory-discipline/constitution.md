# agent-memory-discipline constitution

Tier 1. These inject on **every turn** while the pack is mounted, so this
file holds only rules that fail if they are ever absent. Anything that is
merely useful when the topic comes up belongs in `corpus.md`, where
retrieval serves it on demand.

Budget: ~1500 tokens, enforced at seal time.

## Name the subject

Every memory you write must name its subject explicitly. Never store a
memory whose subject is a pronoun or a deictic — "he", "they", "this",
"it", "here", "that one". If the subject is not identifiable from the
request, say so and ask rather than storing an unretrievable record.

## Absolute dates only

Convert every relative date to an absolute one before storing. "Next
Tuesday", "last month", "in two weeks" and "yesterday" are meaningless
when retrieved later and, worse, read as current. Resolve against today's
date and store the resolved date.

## Correct, never duplicate

When a fact you already stored has changed, issue a correction against
the existing record. Never store the new value as a second memory: both
versions stay live and retrieval arbitrates between them by similarity,
which it does badly.

## Do not store what the repository records

Never store commit hashes, diffs, file layouts, or anything else
recoverable from the repository at higher fidelity. Say that it is
repo-derivable instead. The memory copy goes stale silently; the
repository does not.

## Bulk content ingests at 0.6

Bulk or reference content is stored at importance 0.6, never 1.0.
Per-namespace calibration compresses later high marks once the mean
saturates, so a bulk load at 1.0 ranks its own later records below its
earlier ones.
