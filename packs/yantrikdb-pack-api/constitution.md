# yantrikdb-pack-api constitution

## Construction and dimensions

Open or create a database with `db = YantrikDB(path, 64)` — the second
argument is the embedding dimension, and 64 is the bundled embedder's
dimension, which packs use. Import with `from yantrikdb import
YantrikDB`. Close with `db.close()` when done.

## Record before sealing

A pack is sealed FROM a namespace, so content must be recorded first:
`db.record_text(text, namespace=..., importance=0.6)`. Sealing an empty
namespace produces an empty pack.

## seal_pack refuses to overwrite

`seal_pack` raises if the destination file already exists. Seal to a
fresh path, or delete the old file first.

## Signing happens after sealing, on the file

`YantrikDB.sign_pack(path, secret_hex)` is a static method that operates
on an already-sealed pack file. Keys come from
`YantrikDB.generate_pack_keypair()`, which returns the tuple
`(secret_hex, public_hex)` in that order.

## Trust is a host decision, made before mounting

A valid signature from an unknown key mounts at the "unsigned" trust
tier. To get the "signed" tier, the host database must first call
`db.trust_publisher(public_hex, label)` — then mount.

## mount is transient, install is durable

`db.mount_pack(path)` lasts only for the current process and never
writes to the host database. `db.install_pack(path)` copies the pack
beside the database and re-mounts it automatically on every future
open. Check state with `db.mounted_packs()` and `db.installed_packs()`,
both returning lists of dicts.
