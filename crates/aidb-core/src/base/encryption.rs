/// Encryption at rest using AES-256-GCM with envelope encryption.
///
/// Architecture:
///   - User provides a 32-byte master key (or derives one from a passphrase).
///   - On first open: a random Data Encryption Key (DEK) is generated.
///   - DEK is encrypted with the master key and stored in the `meta` table.
///   - All sensitive fields (text, metadata, embeddings) are encrypted with the DEK.
///   - In-memory indexes (HNSW, scoring cache, graph) operate on plaintext.
///   - Only the SQLite persistence layer stores ciphertext.

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use rand::Rng;

use crate::error::{AidbError, Result};

const NONCE_LEN: usize = 12;

/// Provides encrypt/decrypt operations using AES-256-GCM.
pub struct EncryptionProvider {
    cipher: Aes256Gcm,
}

impl EncryptionProvider {
    /// Create a provider from a raw 32-byte Data Encryption Key.
    pub fn from_dek(dek: &[u8; 32]) -> Self {
        let cipher = Aes256Gcm::new_from_slice(dek).expect("valid 32-byte key");
        Self { cipher }
    }

    /// Encrypt raw bytes. Returns nonce || ciphertext (with auth tag).
    pub fn encrypt_bytes(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        let nonce_bytes: [u8; NONCE_LEN] = rand::thread_rng().gen();
        let nonce = Nonce::from_slice(&nonce_bytes);
        let ciphertext = self
            .cipher
            .encrypt(nonce, plaintext)
            .map_err(|_| AidbError::Encryption("encrypt failed".into()))?;
        let mut out = Vec::with_capacity(NONCE_LEN + ciphertext.len());
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Decrypt raw bytes (expects nonce || ciphertext format).
    pub fn decrypt_bytes(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < NONCE_LEN {
            return Err(AidbError::Encryption("ciphertext too short".into()));
        }
        let (nonce_bytes, ciphertext) = data.split_at(NONCE_LEN);
        let nonce = Nonce::from_slice(nonce_bytes);
        self.cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| AidbError::Encryption("decrypt failed (wrong key?)".into()))
    }

    /// Encrypt a string, returning a base64-encoded ciphertext suitable for TEXT columns.
    pub fn encrypt_string(&self, plaintext: &str) -> Result<String> {
        let encrypted = self.encrypt_bytes(plaintext.as_bytes())?;
        Ok(B64.encode(&encrypted))
    }

    /// Decrypt a base64-encoded ciphertext back to a string.
    pub fn decrypt_string(&self, b64_ciphertext: &str) -> Result<String> {
        let encrypted = B64
            .decode(b64_ciphertext)
            .map_err(|e| AidbError::Encryption(format!("base64 decode: {e}")))?;
        let plaintext = self.decrypt_bytes(&encrypted)?;
        String::from_utf8(plaintext)
            .map_err(|e| AidbError::Encryption(format!("invalid UTF-8: {e}")))
    }
}

/// Generate a random 32-byte key.
pub fn generate_key() -> [u8; 32] {
    rand::thread_rng().gen()
}

/// Encrypt a DEK with the master key (for storage in meta table).
pub fn wrap_dek(master_key: &[u8; 32], dek: &[u8; 32]) -> Result<Vec<u8>> {
    let provider = EncryptionProvider::from_dek(master_key);
    provider.encrypt_bytes(dek)
}

/// Decrypt a DEK using the master key.
pub fn unwrap_dek(master_key: &[u8; 32], wrapped: &[u8]) -> Result<[u8; 32]> {
    let provider = EncryptionProvider::from_dek(master_key);
    let dek_bytes = provider.decrypt_bytes(wrapped)?;
    if dek_bytes.len() != 32 {
        return Err(AidbError::Encryption(format!(
            "DEK wrong length: expected 32, got {}",
            dek_bytes.len()
        )));
    }
    let mut dek = [0u8; 32];
    dek.copy_from_slice(&dek_bytes);
    Ok(dek)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_bytes() {
        let dek = generate_key();
        let provider = EncryptionProvider::from_dek(&dek);
        let plaintext = b"Hello, AIDB!";
        let encrypted = provider.encrypt_bytes(plaintext).unwrap();
        assert_ne!(&encrypted[NONCE_LEN..], plaintext);
        let decrypted = provider.decrypt_bytes(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_encrypt_decrypt_string() {
        let dek = generate_key();
        let provider = EncryptionProvider::from_dek(&dek);
        let text = "Alice works at Anthropic and prefers Earl Grey tea.";
        let encrypted = provider.encrypt_string(text).unwrap();
        assert_ne!(encrypted, text);
        let decrypted = provider.decrypt_string(&encrypted).unwrap();
        assert_eq!(decrypted, text);
    }

    #[test]
    fn test_wrong_key_fails() {
        let dek1 = generate_key();
        let dek2 = generate_key();
        let p1 = EncryptionProvider::from_dek(&dek1);
        let p2 = EncryptionProvider::from_dek(&dek2);
        let encrypted = p1.encrypt_string("secret").unwrap();
        assert!(p2.decrypt_string(&encrypted).is_err());
    }

    #[test]
    fn test_wrap_unwrap_dek() {
        let master_key = generate_key();
        let dek = generate_key();
        let wrapped = wrap_dek(&master_key, &dek).unwrap();
        let unwrapped = unwrap_dek(&master_key, &wrapped).unwrap();
        assert_eq!(dek, unwrapped);
    }

    #[test]
    fn test_wrong_master_key_fails() {
        let mk1 = generate_key();
        let mk2 = generate_key();
        let dek = generate_key();
        let wrapped = wrap_dek(&mk1, &dek).unwrap();
        assert!(unwrap_dek(&mk2, &wrapped).is_err());
    }

    #[test]
    fn test_encrypt_empty_string() {
        let dek = generate_key();
        let provider = EncryptionProvider::from_dek(&dek);
        let encrypted = provider.encrypt_string("").unwrap();
        let decrypted = provider.decrypt_string(&encrypted).unwrap();
        assert_eq!(decrypted, "");
    }

    #[test]
    fn test_encrypt_large_payload() {
        let dek = generate_key();
        let provider = EncryptionProvider::from_dek(&dek);
        let large = "x".repeat(100_000);
        let encrypted = provider.encrypt_string(&large).unwrap();
        let decrypted = provider.decrypt_string(&encrypted).unwrap();
        assert_eq!(decrypted, large);
    }

    #[test]
    fn test_each_encryption_unique() {
        let dek = generate_key();
        let provider = EncryptionProvider::from_dek(&dek);
        let text = "same plaintext";
        let e1 = provider.encrypt_string(text).unwrap();
        let e2 = provider.encrypt_string(text).unwrap();
        // Random nonces mean different ciphertexts
        assert_ne!(e1, e2);
        // Both decrypt to same plaintext
        assert_eq!(provider.decrypt_string(&e1).unwrap(), text);
        assert_eq!(provider.decrypt_string(&e2).unwrap(), text);
    }
}
