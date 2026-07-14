/// Serialize a Vec<f32> to bytes (little-endian f32 array) for sqlite-vec.
pub fn serialize_f32(vector: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(vector.len() * 4);
    for &v in vector {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Deserialize bytes to Vec<f32>.
pub fn deserialize_f32(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Lowercase hex encoding (v0.10 Phase 0: HLC bytes carried in JSON
/// replication payloads so followers persist the leader's exact edge
/// identity/order).
pub fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Decode lowercase/uppercase hex back to bytes; None on malformed input.
pub fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let v = vec![1.0f32, -2.5, 0.0, 3.14159, f32::MAX, f32::MIN];
        let blob = serialize_f32(&v);
        let back = deserialize_f32(&blob);
        assert_eq!(v, back);
    }

    #[test]
    fn test_empty() {
        let v: Vec<f32> = vec![];
        let blob = serialize_f32(&v);
        assert!(blob.is_empty());
        let back = deserialize_f32(&blob);
        assert!(back.is_empty());
    }

    #[test]
    fn test_single_element() {
        let v = vec![42.0f32];
        let blob = serialize_f32(&v);
        assert_eq!(blob.len(), 4);
        let back = deserialize_f32(&blob);
        assert_eq!(v, back);
    }
}
