# yantrikdb-pack-api corpus

API reference for the pack surface, one entry per `## ` heading. This
API was introduced on 2026-07-28.

## Opening a database for pack work

`from yantrikdb import YantrikDB` then `db = YantrikDB(db_path, 64)`.
The 64 is the embedding dimension of the bundled embedder, which is what
packs use. The constructor creates the file if it does not exist. Always
`db.close()` when finished.

## Recording content that will become a pack

`db.record_text(text, namespace="mypack_ns", importance=0.6)` stores one
fact with an engine-generated embedding. The namespace argument is what
`seal_pack` later scopes to. Bulk content uses importance 0.6.

## seal_pack — turn a namespace into a pack file

```python
manifest = db.seal_pack(
    dest_path,                 # e.g. "physics.ydbpack"; must not exist yet
    name="physics",
    version="1.0.0",
    origin="acme/physics",     # publisher-scoped identity
    namespace="physics_ns",    # which namespace to export
    description="optional human summary",
)
```
Returns the sealed manifest as a dict including `pack_id` (origin@version),
`corpus_rows` and `content_digest`. Raises if `dest_path` already exists.

## generate_pack_keypair — publisher identity

`secret_hex, public_hex = YantrikDB.generate_pack_keypair()` — a static
method returning the Ed25519 keypair as two hex strings, secret first.
The secret is the publisher identity; the public half is shared with
buyers.

## sign_pack — sign a sealed pack file

`pubkey = YantrikDB.sign_pack(pack_path, secret_hex)` — static method,
called after sealing. Writes the publisher public key and signature into
the pack's manifest and returns the public key hex.

## mount_pack — transient attach

`pack_id = db.mount_pack(pack_path)` mounts a sealed pack read-only for
this process and returns its pack id string. It never writes to the host
database, and the mount is gone when the process exits. Raises
`PackEmbedderMismatch` if the pack's embedding space does not match, and
`PackAlreadyMounted` if the same pack id is already mounted.

## install_pack — durable attach

`pack_id = db.install_pack(pack_path)` copies the pack file into a
directory beside the database, mounts it, and records it so every future
open of that database re-mounts it automatically with no API call.
`db.uninstall_pack(pack_id)` reverses all of it.

## unmount_pack — detach

`db.unmount_pack(pack_id)` drops a transient mount, returning True if it
was mounted. The host database file is byte-identical after.

## mounted_packs — what is attached right now

`db.mounted_packs()` returns a list of dicts, one per mounted pack, with
keys `pack_id`, `name`, `version`, `origin`, `trust` ("signed",
"unsigned" or "unverified"), `rows` and `tier_multiplier`. An empty list
means nothing is mounted.

## installed_packs — what is durable

`db.installed_packs()` returns a list of dicts for packs recorded as
installed (whether or not currently mounted), with keys `pack_id`,
`file_name`, `name`, `version`, `content_digest`, `installed_at`.

## trust_publisher — earn the signed tier

`db.trust_publisher(public_hex, "Acme Corp")` records that this host
trusts the key. A pack validly signed by a trusted key mounts at trust
"signed"; the same pack without prior trust mounts at "unsigned".
`db.untrust_publisher(public_hex)` reverses it.

## recall_text — query across host and mounted packs

`hits = db.recall_text("what are gluons", top_k=5)` returns a list of
dicts with `text`, `score`, and `scores` (whose `similarity` field is
used for relevance gating). With a pack mounted, pack rows appear in
these results automatically.

## read_pack_manifest — inspect without mounting

`YantrikDB.read_pack_manifest(path)` — static; returns the manifest dict
(`pack_id`, `corpus_rows`, `content_digest`, `embedder`, ...) without
mounting the file.

## pack_context — the prompt block for mounted packs

`db.pack_context()` returns a string block describing every mounted
pack's coverage and rules for placement in a system prompt, or None if
nothing mounted declares them.
