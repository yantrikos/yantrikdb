//! Secure credential vault baked into YantrikDB.
//!
//! Credentials are encrypted with AES-256-GCM under a vault-specific data key. Only the service
//! name and metadata are stored in the clear, for listing and search.
//!
//! # What the key is protected by, and what it used to be
//!
//! The data key was previously written to the `vault_security` table as plain base64 — in the
//! same SQLite file as the ciphertext it decrypts. That is not encryption at rest; it is
//! obfuscation with an extra step. Anyone holding the file held every credential: a stolen
//! laptop, a copied VM disk, a backup, a snapshot.
//!
//! The PIN did not help. It was an unsalted single-round BLAKE3 hash, stored in the same table,
//! and it never derived anything — `verify_pin` returned `true` when no PIN was set, so
//! `DELETE FROM vault_security WHERE key='pin_hash'` was the whole bypass, with no cryptography
//! involved at any point.
//!
//! Now the data key is *wrapped*: encrypted under a key-encryption key derived from the user's
//! passphrase with Argon2id, and only the wrapped form is stored. The file alone is no longer
//! enough. A wrong passphrase produces a KEK that fails AEAD authentication rather than
//! plausible-looking garbage, so there is nothing to brute-force offline except Argon2id itself.
//!
//! [`unlock`] holds the unwrapped key in memory for the life of the process and nowhere else.
//! [`lock`] overwrites it.
//!
//! # Vaults created before this
//!
//! They still open, because refusing would lock people out of their own credentials. But
//! [`is_protected`] reports false for them and [`set_passphrase`] migrates them in place: it
//! wraps the existing key and deletes the plaintext row, so the entries survive and the exposure
//! does not.

use std::sync::Mutex;

use base64::{engine::general_purpose::STANDARD as B64, Engine};
use rand::Rng;
use rusqlite::Connection;

use super::encryption::{self, EncryptionProvider};
use super::error::{Result, YantrikDbError};

/// The unwrapped data key for this process, once a passphrase has opened the vault.
///
/// In memory, never on disk. A caller that has not unlocked gets an error rather than data,
/// which is the point of wrapping the key in the first place.
static UNLOCKED: Mutex<Option<[u8; 32]>> = Mutex::new(None);

/// How hard it is to turn a passphrase into a key.
///
/// The defaults from the Argon2 RFC's second recommended configuration: 19 MiB, 2 passes. Chosen
/// for the threat that matters here — someone with the file, guessing offline — where the cost per
/// guess is the only thing standing between a six-character passphrase and every credential.
/// A single unlock costs a fraction of a second; a billion guesses cost a very long time.
const KDF_MEMORY_KIB: u32 = 19 * 1024;
const KDF_PASSES: u32 = 2;
const KDF_LANES: u32 = 1;

/// Derive the key-encryption key from a passphrase and the vault's salt.
fn derive_kek(passphrase: &str, salt: &[u8]) -> Result<[u8; 32]> {
    use argon2::{Algorithm, Argon2, Params, Version};

    let params = Params::new(KDF_MEMORY_KIB, KDF_PASSES, KDF_LANES, Some(32))
        .map_err(|e| YantrikDbError::Encryption(format!("argon2 params: {e}")))?;
    let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

    let mut kek = [0u8; 32];
    argon
        .hash_password_into(passphrase.as_bytes(), salt, &mut kek)
        .map_err(|e| YantrikDbError::Encryption(format!("argon2: {e}")))?;
    Ok(kek)
}

fn read_row(conn: &Connection, key: &str) -> Option<String> {
    conn.query_row(
        "SELECT value FROM vault_security WHERE key = ?1",
        rusqlite::params![key],
        |row| row.get(0),
    )
    .ok()
}

fn write_row(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO vault_security (key, value) VALUES (?1, ?2)",
        rusqlite::params![key, value],
    )?;
    Ok(())
}

/// Whether this vault's key is wrapped under a passphrase.
///
/// False means the key is still sitting in the file next to the ciphertext, which is the state
/// every vault created before this was in. Reported rather than assumed, so a caller can tell the
/// user the truth about their own vault.
pub fn is_protected(conn: &Connection) -> bool {
    read_row(conn, "dek_wrapped").is_some()
}

/// Whether this process currently holds the unwrapped key.
pub fn is_unlocked() -> bool {
    UNLOCKED.lock().map(|g| g.is_some()).unwrap_or(false)
}

/// Open the vault with a passphrase, and keep the key for this process.
///
/// A wrong passphrase fails here, at AEAD authentication, rather than producing a key that
/// silently decrypts to nonsense.
pub fn unlock(conn: &Connection, passphrase: &str) -> Result<EncryptionProvider> {
    let salt_b64 = read_row(conn, "kdf_salt").ok_or_else(|| {
        YantrikDbError::Encryption(
            "this vault has no passphrase; call set_passphrase to protect it".into(),
        )
    })?;
    let wrapped_b64 = read_row(conn, "dek_wrapped")
        .ok_or_else(|| YantrikDbError::Encryption("this vault has no wrapped key".into()))?;

    let salt = B64
        .decode(&salt_b64)
        .map_err(|e| YantrikDbError::Encryption(format!("salt decode: {e}")))?;
    let wrapped = B64
        .decode(&wrapped_b64)
        .map_err(|e| YantrikDbError::Encryption(format!("wrapped key decode: {e}")))?;

    let kek = derive_kek(passphrase, &salt)?;
    let dek = encryption::unwrap_dek(&kek, &wrapped)
        .map_err(|_| YantrikDbError::Encryption("wrong passphrase".into()))?;

    if let Ok(mut guard) = UNLOCKED.lock() {
        *guard = Some(dek);
    }
    Ok(EncryptionProvider::from_dek(&dek))
}

/// Forget the key. The vault cannot be read again without the passphrase.
pub fn lock() {
    if let Ok(mut guard) = UNLOCKED.lock() {
        if let Some(dek) = guard.as_mut() {
            // Overwritten rather than dropped: a freed buffer keeps its contents until something
            // else claims the page, and a key is worth the four lines.
            dek.iter_mut().for_each(|b| *b = 0);
        }
        *guard = None;
    }
}

/// Protect this vault with a passphrase, migrating an unprotected one in place.
///
/// Existing entries are preserved: the data key they were encrypted under is kept and wrapped,
/// rather than replaced. What goes away is the copy of it stored in the clear.
pub fn set_passphrase(conn: &Connection, passphrase: &str) -> Result<()> {
    if passphrase.trim().is_empty() {
        return Err(YantrikDbError::Encryption(
            "a passphrase cannot be empty".into(),
        ));
    }

    // The key to keep: whatever is currently in use, so nothing already stored becomes unreadable.
    let dek = if let Some(existing) = current_dek(conn) {
        existing
    } else {
        encryption::generate_key()
    };

    let mut salt = [0u8; 16];
    rand::thread_rng().fill(&mut salt);
    let kek = derive_kek(passphrase, &salt)?;
    let wrapped = encryption::wrap_dek(&kek, &dek)?;

    write_row(conn, "kdf_salt", &B64.encode(salt))?;
    write_row(conn, "dek_wrapped", &B64.encode(&wrapped))?;
    write_row(conn, "kdf", "argon2id")?;

    // The whole point. Until this row is gone the file still contains the key.
    conn.execute("DELETE FROM vault_security WHERE key = 'vault_dek'", [])?;
    // And the old PIN hash, which protected nothing and would only mislead anyone reading the
    // table about what is guarding this vault.
    conn.execute("DELETE FROM vault_security WHERE key = 'pin_hash'", [])?;

    if let Ok(mut guard) = UNLOCKED.lock() {
        *guard = Some(dek);
    }
    Ok(())
}

/// The data key currently in force, from memory or from a legacy plaintext row.
fn current_dek(conn: &Connection) -> Option<[u8; 32]> {
    if let Ok(guard) = UNLOCKED.lock() {
        if let Some(dek) = *guard {
            return Some(dek);
        }
    }
    let b64 = read_row(conn, "vault_dek")?;
    let bytes = B64.decode(&b64).ok()?;
    if bytes.len() != 32 {
        return None;
    }
    let mut dek = [0u8; 32];
    dek.copy_from_slice(&bytes);
    Some(dek)
}

/// Get or create the vault's own EncryptionProvider.
///
/// The vault DEK is stored in the `vault_security` table, auto-generated on first access.
/// This is independent of the DB-level encryption — works even when the DB is opened
/// without a master key.
pub fn vault_encryption(conn: &Connection) -> Result<EncryptionProvider> {
    // Unlocked in this process: use the key we already hold.
    if let Ok(guard) = UNLOCKED.lock() {
        if let Some(dek) = *guard {
            return Ok(EncryptionProvider::from_dek(&dek));
        }
    }

    // Protected but not unlocked. Refused, and told why — this is the case the wrapping exists
    // for, and quietly falling back to anything else would undo it.
    if is_protected(conn) {
        return Err(YantrikDbError::Encryption(
            "the vault is locked; unlock it with its passphrase".into(),
        ));
    }

    let existing: Option<String> = conn
        .query_row(
            "SELECT value FROM vault_security WHERE key = 'vault_dek'",
            [],
            |row| row.get(0),
        )
        .ok();

    if let Some(b64_dek) = existing {
        let dek_bytes = B64.decode(&b64_dek).map_err(|e| {
            super::error::YantrikDbError::Encryption(format!("vault DEK decode: {e}"))
        })?;
        if dek_bytes.len() != 32 {
            return Err(super::error::YantrikDbError::Encryption(format!(
                "vault DEK wrong length: {}",
                dek_bytes.len()
            )));
        }
        let mut dek = [0u8; 32];
        dek.copy_from_slice(&dek_bytes);
        Ok(EncryptionProvider::from_dek(&dek))
    } else {
        let dek = encryption::generate_key();
        let b64 = B64.encode(dek);
        conn.execute(
            "INSERT OR REPLACE INTO vault_security (key, value) VALUES ('vault_dek', ?1)",
            rusqlite::params![b64],
        )?;
        Ok(EncryptionProvider::from_dek(&dek))
    }
}

/// Initialize vault tables. Called during DB setup.
pub fn init_tables(conn: &Connection) {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS vault_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            service TEXT NOT NULL,
            username_enc TEXT NOT NULL,
            password_enc TEXT NOT NULL,
            url TEXT,
            notes_enc TEXT,
            category TEXT NOT NULL DEFAULT 'general',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(service, url)
        );
        CREATE INDEX IF NOT EXISTS idx_vault_service ON vault_entries(service);
        CREATE INDEX IF NOT EXISTS idx_vault_category ON vault_entries(category);
        CREATE TABLE IF NOT EXISTS vault_security (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );",
    )
    .expect("vault schema creation");
}

/// A decrypted vault entry.
#[derive(Debug, Clone)]
pub struct VaultEntry {
    pub id: i64,
    pub service: String,
    pub username: String,
    pub password: String,
    pub url: Option<String>,
    pub notes: Option<String>,
    pub category: String,
    pub created_at: f64,
    pub updated_at: f64,
}

/// A vault entry without sensitive fields (for listing).
#[derive(Debug, Clone)]
pub struct VaultListEntry {
    pub id: i64,
    pub service: String,
    pub url: Option<String>,
    pub category: String,
    pub updated_at: f64,
}

/// Store a credential in the vault.
pub fn store(
    conn: &Connection,
    enc: &EncryptionProvider,
    service: &str,
    username: &str,
    password: &str,
    url: Option<&str>,
    notes: Option<&str>,
    category: Option<&str>,
) -> Result<i64> {
    let now = crate::time::now_secs();

    let username_enc = enc.encrypt_string(username)?;
    let password_enc = enc.encrypt_string(password)?;
    let notes_enc = notes.map(|n| enc.encrypt_string(n)).transpose()?;
    let cat = category.unwrap_or("general");

    // Try UPDATE first for upsert — ON CONFLICT doesn't match NULLs in url
    let updated = conn.execute(
        "UPDATE vault_entries SET username_enc = ?1, password_enc = ?2, \
         notes_enc = ?3, category = ?4, updated_at = ?5 \
         WHERE service = ?6 AND (url IS ?7)",
        rusqlite::params![
            username_enc,
            password_enc,
            notes_enc,
            cat,
            now,
            service,
            url
        ],
    )?;
    if updated == 0 {
        conn.execute(
            "INSERT INTO vault_entries \
             (service, username_enc, password_enc, url, notes_enc, category, created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
            rusqlite::params![service, username_enc, password_enc, url, notes_enc, cat, now],
        )?;
    }

    let id = conn.last_insert_rowid();
    Ok(id)
}

/// Retrieve a credential by service name (decrypted).
pub fn get(conn: &Connection, enc: &EncryptionProvider, service: &str) -> Result<Vec<VaultEntry>> {
    let mut stmt = conn.prepare(
        "SELECT id, service, username_enc, password_enc, url, notes_enc, category, created_at, updated_at
         FROM vault_entries WHERE service = ?1 ORDER BY updated_at DESC"
    )?;

    let rows = stmt.query_map(rusqlite::params![service], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, Option<String>>(4)?,
            row.get::<_, Option<String>>(5)?,
            row.get::<_, String>(6)?,
            row.get::<_, f64>(7)?,
            row.get::<_, f64>(8)?,
        ))
    })?;

    let mut entries = Vec::new();
    for row in rows {
        let (id, svc, u_enc, p_enc, url, n_enc, cat, created, updated) = row?;

        let username = enc.decrypt_string(&u_enc)?;
        let password = enc.decrypt_string(&p_enc)?;
        let notes = n_enc.map(|n| enc.decrypt_string(&n)).transpose()?;

        entries.push(VaultEntry {
            id,
            service: svc,
            username,
            password,
            url,
            notes,
            category: cat,
            created_at: created,
            updated_at: updated,
        });
    }
    Ok(entries)
}

/// Search vault entries by service name pattern (case-insensitive).
pub fn search(conn: &Connection, enc: &EncryptionProvider, query: &str) -> Result<Vec<VaultEntry>> {
    let pattern = format!("%{query}%");
    let mut stmt = conn.prepare(
        "SELECT id, service, username_enc, password_enc, url, notes_enc, category, created_at, updated_at
         FROM vault_entries WHERE service LIKE ?1 ORDER BY updated_at DESC LIMIT 20"
    )?;

    let rows = stmt.query_map(rusqlite::params![pattern], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, Option<String>>(4)?,
            row.get::<_, Option<String>>(5)?,
            row.get::<_, String>(6)?,
            row.get::<_, f64>(7)?,
            row.get::<_, f64>(8)?,
        ))
    })?;

    let mut entries = Vec::new();
    for row in rows {
        let (id, svc, u_enc, p_enc, url, n_enc, cat, created, updated) = row?;

        let username = enc.decrypt_string(&u_enc)?;
        let password = enc.decrypt_string(&p_enc)?;
        let notes = n_enc.map(|n| enc.decrypt_string(&n)).transpose()?;

        entries.push(VaultEntry {
            id,
            service: svc,
            username,
            password,
            url,
            notes,
            category: cat,
            created_at: created,
            updated_at: updated,
        });
    }
    Ok(entries)
}

/// List all vault entries (without decrypted sensitive fields).
pub fn list(conn: &Connection) -> Result<Vec<VaultListEntry>> {
    let mut stmt = conn.prepare(
        "SELECT id, service, url, category, updated_at
         FROM vault_entries ORDER BY service ASC",
    )?;

    let rows = stmt.query_map([], |row| {
        Ok(VaultListEntry {
            id: row.get(0)?,
            service: row.get(1)?,
            url: row.get(2)?,
            category: row.get(3)?,
            updated_at: row.get(4)?,
        })
    })?;

    rows.into_iter().map(|r| Ok(r?)).collect()
}

/// Delete a vault entry by ID.
pub fn delete(conn: &Connection, id: i64) -> Result<bool> {
    let changed = conn.execute(
        "DELETE FROM vault_entries WHERE id = ?1",
        rusqlite::params![id],
    )?;
    Ok(changed > 0)
}

/// Delete a vault entry by service name.
pub fn delete_by_service(conn: &Connection, service: &str) -> Result<usize> {
    let changed = conn.execute(
        "DELETE FROM vault_entries WHERE service = ?1",
        rusqlite::params![service],
    )?;
    Ok(changed)
}

// ── Security PIN ──

/// Set or update the vault security PIN. Stores a blake3 hash.
/// Set the vault's PIN, which is now the passphrase that protects it.
///
/// One concept rather than two. It used to be a separate thing: a hash in a table that gated the
/// tools while the actual key sat beside it in the clear, so the PIN could be deleted and the
/// vault read anyway. Now the PIN *is* what the key is wrapped under, and there is nothing to
/// delete that would help.
///
/// A vault that already has entries keeps them: see [`set_passphrase`].
pub fn set_pin(conn: &Connection, pin: &str) -> Result<()> {
    set_passphrase(conn, pin)
}

/// Check a PIN by using it: it either unwraps the vault's key or it does not.
///
/// This is the difference between the old version and this one. Before, it compared a hash and
/// the answer had no bearing on whether the data could be read — the key was readable either way.
/// Now a correct PIN is the only thing that produces a usable key, and a wrong one fails AEAD
/// authentication rather than returning a plausible-looking wrong answer.
///
/// A side effect worth knowing about: success leaves the vault unlocked for this process, so a
/// caller that verifies and then reads does not have to pass the PIN twice.
///
/// Still returns true for a vault with no passphrase, because there is nothing to check and
/// refusing would lock people out of vaults created before this existed. [`is_protected`] is how
/// a caller tells the two apart.
pub fn verify_pin(conn: &Connection, pin: &str) -> bool {
    if !is_protected(conn) {
        return true;
    }
    unlock(conn, pin).is_ok()
}

/// Whether this vault is protected by a PIN.
///
/// Now the same question as [`is_protected`]: a PIN that does not wrap the key is not protection,
/// so there is only one thing left to report.
pub fn has_pin(conn: &Connection) -> bool {
    is_protected(conn)
}

/// Remove the vault's protection, putting its key back in the file in the clear.
///
/// It has to be said plainly rather than done quietly: with no passphrase there is nothing to
/// wrap the key under, so removing the PIN means writing the key next to the data it decrypts.
/// Anyone who can read the file can then read every credential. That is what this used to be all
/// the time; now it is a choice someone makes.
///
/// Requires the vault to be unlocked, because the key has to come from somewhere.
pub fn remove_pin(conn: &Connection) -> Result<()> {
    if !is_protected(conn) {
        // Nothing to remove. The legacy `pin_hash` row, if one survives, protects nothing.
        conn.execute("DELETE FROM vault_security WHERE key = 'pin_hash'", [])?;
        return Ok(());
    }

    let dek = current_dek(conn).ok_or_else(|| {
        YantrikDbError::Encryption("unlock the vault before removing its protection".into())
    })?;

    write_row(conn, "vault_dek", &B64.encode(dek))?;
    conn.execute("DELETE FROM vault_security WHERE key = 'dek_wrapped'", [])?;
    conn.execute("DELETE FROM vault_security WHERE key = 'kdf_salt'", [])?;
    conn.execute("DELETE FROM vault_security WHERE key = 'kdf'", [])?;
    Ok(())
}

/// Generate a cryptographically secure random password.
pub fn generate_password(length: usize, include_special: bool) -> String {
    let length = length.clamp(8, 128);
    let mut rng = rand::thread_rng();

    let lowercase = b"abcdefghijkmnopqrstuvwxyz"; // no l (ambiguous)
    let uppercase = b"ABCDEFGHJKLMNPQRSTUVWXYZ"; // no I, O (ambiguous)
    let digits = b"23456789"; // no 0, 1 (ambiguous)
    let special = b"!@#$%^&*-_=+?";

    let mut charset: Vec<u8> = Vec::new();
    charset.extend_from_slice(lowercase);
    charset.extend_from_slice(uppercase);
    charset.extend_from_slice(digits);
    if include_special {
        charset.extend_from_slice(special);
    }

    // Ensure at least one of each category
    let mut password: Vec<u8> = Vec::with_capacity(length);
    password.push(lowercase[rng.gen_range(0..lowercase.len())]);
    password.push(uppercase[rng.gen_range(0..uppercase.len())]);
    password.push(digits[rng.gen_range(0..digits.len())]);
    if include_special {
        password.push(special[rng.gen_range(0..special.len())]);
    }

    while password.len() < length {
        password.push(charset[rng.gen_range(0..charset.len())]);
    }

    // Shuffle (Fisher-Yates)
    for i in (1..password.len()).rev() {
        let j = rng.gen_range(0..=i);
        password.swap(i, j);
    }

    String::from_utf8(password).expect("ASCII password")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encryption::{generate_key, EncryptionProvider};

    fn setup() -> (Connection, EncryptionProvider) {
        let conn = Connection::open_in_memory().unwrap();
        init_tables(&conn);
        let dek = generate_key();
        let enc = EncryptionProvider::from_dek(&dek);
        (conn, enc)
    }

    // ── What the wrapping is for ────────────────────────────────────
    //
    // These tests are written as attacks, because that is the only way to know whether a
    // protection protects anything. Each one is a thing that used to work.

    /// The bypass, exactly as it was: the PIN was a row in a table, and deleting the row was the
    /// whole of it. `verify_pin` returned true when no PIN was set, and no cryptography was
    /// involved at any point.
    #[test]
    fn deleting_the_pin_row_no_longer_opens_the_vault() {
        let conn = Connection::open_in_memory().unwrap();
        init_tables(&conn);
        lock();

        set_passphrase(&conn, "correct horse battery staple").unwrap();
        let enc = vault_encryption(&conn).unwrap();
        store(
            &conn,
            &enc,
            "reddit.com",
            "someone",
            "hunter2",
            None,
            None,
            None,
        )
        .unwrap();

        // An attacker with the file does the thing that used to work.
        conn.execute("DELETE FROM vault_security WHERE key = 'pin_hash'", [])
            .unwrap();
        lock(); // and comes to it fresh, as a new process would

        let denied = vault_encryption(&conn);
        assert!(
            denied.is_err(),
            "deleting a row must not open a wrapped vault"
        );
    }

    /// The file used to be enough on its own. It must not be.
    #[test]
    fn the_database_file_alone_does_not_contain_the_key() {
        let conn = Connection::open_in_memory().unwrap();
        init_tables(&conn);
        lock();

        set_passphrase(&conn, "a passphrase").unwrap();

        // The row the key used to live in is gone, and nothing else holds it in the clear.
        let plaintext_key: Option<String> = conn
            .query_row(
                "SELECT value FROM vault_security WHERE key = 'vault_dek'",
                [],
                |r| r.get(0),
            )
            .ok();
        assert!(
            plaintext_key.is_none(),
            "the unwrapped key must not be in the file"
        );

        // And what is left cannot be turned back into it without the passphrase.
        lock();
        assert!(vault_encryption(&conn).is_err());
    }

    /// The property, stated directly: nothing left in the file is the key.
    ///
    /// The previous test checks that one named row is gone. This one checks the thing that row
    /// being gone was supposed to achieve — that no value anywhere in the security table can be
    /// used as a key to read an entry. It is the test that would have failed loudest before, and
    /// the one that would catch a future change putting the key back by another name.
    #[test]
    fn no_value_left_in_the_file_can_decrypt_an_entry() {
        let conn = Connection::open_in_memory().unwrap();
        init_tables(&conn);
        lock();

        set_passphrase(&conn, "a real passphrase").unwrap();
        let enc = vault_encryption(&conn).unwrap();
        store(
            &conn,
            &enc,
            "bank.example",
            "me",
            "the-secret",
            None,
            None,
            None,
        )
        .unwrap();
        lock();

        let mut stmt = conn.prepare("SELECT value FROM vault_security").unwrap();
        let values: Vec<String> = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .filter_map(|v| v.ok())
            .collect();
        assert!(
            !values.is_empty(),
            "the table should still hold the salt and the wrapped key"
        );

        for value in &values {
            let Ok(bytes) = B64.decode(value) else {
                continue;
            };
            if bytes.len() != 32 {
                continue;
            }
            let mut candidate = [0u8; 32];
            candidate.copy_from_slice(&bytes);
            let guess = EncryptionProvider::from_dek(&candidate);
            assert!(
                get(&conn, &guess, "bank.example").is_err()
                    || get(&conn, &guess, "bank.example").unwrap().is_empty(),
                "a value stored in the file decrypted a credential"
            );
        }
    }

    #[test]
    fn a_wrong_passphrase_fails_rather_than_producing_nonsense() {
        let conn = Connection::open_in_memory().unwrap();
        init_tables(&conn);
        lock();

        set_passphrase(&conn, "the right one").unwrap();
        lock();

        let wrong = unlock(&conn, "the wrong one");
        assert!(wrong.is_err(), "a wrong passphrase must not unwrap the key");
        // It fails at authentication, so there is no plausible-looking output to sift through.
        assert!(format!("{:?}", wrong.err().unwrap()).contains("passphrase"));

        assert!(unlock(&conn, "the right one").is_ok());
    }

    /// Migration must not cost anyone their credentials, or nobody will run it.
    #[test]
    fn protecting_an_existing_vault_keeps_what_is_in_it() {
        let conn = Connection::open_in_memory().unwrap();
        init_tables(&conn);
        lock();

        // A vault as they exist today: key in the clear, entry stored under it.
        let enc = vault_encryption(&conn).unwrap();
        store(&conn, &enc, "x.com", "me", "s3cret", None, None, None).unwrap();
        assert!(!is_protected(&conn));

        set_passphrase(&conn, "now protected").unwrap();
        assert!(is_protected(&conn));

        lock();
        let enc = unlock(&conn, "now protected").unwrap();
        let entries = get(&conn, &enc, "x.com").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].password, "s3cret",
            "the entry must survive being protected"
        );
    }

    #[test]
    fn locking_forgets_the_key() {
        let conn = Connection::open_in_memory().unwrap();
        init_tables(&conn);
        lock();

        set_passphrase(&conn, "open sesame").unwrap();
        assert!(is_unlocked());
        assert!(vault_encryption(&conn).is_ok());

        lock();
        assert!(!is_unlocked());
        assert!(
            vault_encryption(&conn).is_err(),
            "a locked vault must stay shut"
        );
    }

    /// A vault made before any of this still opens, because refusing would lock people out of
    /// their own credentials — but it must not claim to be protected.
    #[test]
    fn a_legacy_vault_still_opens_and_says_it_is_unprotected() {
        let conn = Connection::open_in_memory().unwrap();
        init_tables(&conn);
        lock();

        let enc = vault_encryption(&conn).unwrap();
        store(&conn, &enc, "old.example", "user", "pw", None, None, None).unwrap();

        assert!(
            !is_protected(&conn),
            "an unwrapped key is not protection and must not read as it"
        );
        let again = vault_encryption(&conn).unwrap();
        assert_eq!(get(&conn, &again, "old.example").unwrap()[0].password, "pw");
    }

    #[test]
    fn the_same_passphrase_and_salt_always_give_the_same_key() {
        // If it did not, a vault would open once and never again.
        let salt = [7u8; 16];
        let a = derive_kek("passphrase", &salt).unwrap();
        let b = derive_kek("passphrase", &salt).unwrap();
        assert_eq!(a, b);

        // And a different salt must give a different key, or two vaults with the same passphrase
        // would share one.
        let c = derive_kek("passphrase", &[9u8; 16]).unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn an_empty_passphrase_is_refused() {
        let conn = Connection::open_in_memory().unwrap();
        init_tables(&conn);
        lock();
        assert!(set_passphrase(&conn, "   ").is_err());
    }

    #[test]
    fn test_store_and_get() {
        let (conn, enc) = setup();
        store(
            &conn,
            &enc,
            "github.com",
            "user123",
            "pass456",
            Some("https://github.com"),
            None,
            None,
        )
        .unwrap();
        let entries = get(&conn, &enc, "github.com").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].username, "user123");
        assert_eq!(entries[0].password, "pass456");
        assert_eq!(entries[0].url.as_deref(), Some("https://github.com"));
    }

    #[test]
    fn test_upsert() {
        let (conn, enc) = setup();
        store(
            &conn,
            &enc,
            "gmail",
            "old@gmail.com",
            "old_pass",
            None,
            None,
            None,
        )
        .unwrap();
        store(
            &conn,
            &enc,
            "gmail",
            "new@gmail.com",
            "new_pass",
            None,
            None,
            None,
        )
        .unwrap();
        let entries = get(&conn, &enc, "gmail").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].username, "new@gmail.com");
        assert_eq!(entries[0].password, "new_pass");
    }

    #[test]
    fn test_list_no_passwords() {
        let (conn, enc) = setup();
        store(
            &conn,
            &enc,
            "github.com",
            "user",
            "secret",
            None,
            None,
            Some("dev"),
        )
        .unwrap();
        store(
            &conn,
            &enc,
            "gmail.com",
            "user@gmail.com",
            "secret2",
            None,
            None,
            Some("email"),
        )
        .unwrap();
        let list_entries = list(&conn).unwrap();
        assert_eq!(list_entries.len(), 2);
    }

    #[test]
    fn test_search() {
        let (conn, enc) = setup();
        store(&conn, &enc, "github.com", "u", "p", None, None, None).unwrap();
        store(&conn, &enc, "gitlab.com", "u", "p", None, None, None).unwrap();
        store(&conn, &enc, "gmail.com", "u", "p", None, None, None).unwrap();
        let results = search(&conn, &enc, "git").unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_delete() {
        let (conn, enc) = setup();
        store(&conn, &enc, "test", "u", "p", None, None, None).unwrap();
        let entries = get(&conn, &enc, "test").unwrap();
        assert!(delete(&conn, entries[0].id).unwrap());
        assert!(get(&conn, &enc, "test").unwrap().is_empty());
    }

    #[test]
    fn test_generate_password() {
        let p1 = generate_password(20, true);
        assert_eq!(p1.len(), 20);
        let p2 = generate_password(20, true);
        assert_ne!(p1, p2);

        let p_no_special = generate_password(16, false);
        assert_eq!(p_no_special.len(), 16);
        assert!(!p_no_special.contains('!') && !p_no_special.contains('@'));
    }

    #[test]
    fn test_wrong_key_cant_decrypt() {
        let (conn, enc) = setup();
        store(
            &conn,
            &enc,
            "secret_service",
            "admin",
            "super_secret",
            None,
            None,
            None,
        )
        .unwrap();

        let other_key = generate_key();
        let other_enc = EncryptionProvider::from_dek(&other_key);
        assert!(get(&conn, &other_enc, "secret_service").is_err());
    }

    #[test]
    fn test_pin_set_verify() {
        let (conn, _) = setup();
        assert!(!has_pin(&conn));

        set_pin(&conn, "1234").unwrap();
        assert!(has_pin(&conn));
        assert!(verify_pin(&conn, "1234"));
        assert!(!verify_pin(&conn, "wrong"));
    }

    #[test]
    fn test_pin_remove() {
        let (conn, _) = setup();
        set_pin(&conn, "9999").unwrap();
        assert!(has_pin(&conn));
        remove_pin(&conn).unwrap();
        assert!(!has_pin(&conn));
    }

    #[test]
    fn test_pin_change() {
        let (conn, _) = setup();
        set_pin(&conn, "old_pin").unwrap();
        assert!(verify_pin(&conn, "old_pin"));
        set_pin(&conn, "new_pin").unwrap();
        assert!(!verify_pin(&conn, "old_pin"));
        assert!(verify_pin(&conn, "new_pin"));
    }
}
