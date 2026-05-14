use std::collections::HashMap;
use std::path::PathBuf;

use crate::error::{Result, YantrikDbError};

use super::YantrikDB;

/// Per-tenant configuration.
pub struct TenantConfig {
    /// Optional 32-byte master key for envelope encryption.
    pub encryption_key: Option<[u8; 32]>,
    /// Optional custom embedding dimension (overrides manager default).
    pub embedding_dim: Option<usize>,
}

impl Default for TenantConfig {
    fn default() -> Self {
        Self {
            encryption_key: None,
            embedding_dim: None,
        }
    }
}

/// Multi-tenant manager that provides isolated YantrikDB instances per tenant.
///
/// Each tenant gets a separate SQLite database file under `base_dir/`,
/// ensuring complete data isolation at the storage layer.
/// Optional per-tenant encryption keys provide defense-in-depth.
pub struct TenantManager {
    base_dir: PathBuf,
    default_embedding_dim: usize,
    /// Per-tenant configs (registered before or at first access).
    configs: HashMap<String, TenantConfig>,
    /// Cache of open YantrikDB instances.
    instances: HashMap<String, YantrikDB>,
}

impl TenantManager {
    /// Create a new TenantManager.
    ///
    /// - `base_dir`: directory where `{tenant_id}.db` files are stored.
    /// - `default_embedding_dim`: default dimension for all tenants.
    pub fn new(base_dir: &str, default_embedding_dim: usize) -> Result<Self> {
        let path = PathBuf::from(base_dir);
        std::fs::create_dir_all(&path).map_err(|e| {
            YantrikDbError::SyncError(format!("failed to create tenant base dir: {e}"))
        })?;
        Ok(Self {
            base_dir: path,
            default_embedding_dim,
            configs: HashMap::new(),
            instances: HashMap::new(),
        })
    }

    /// Register (or update) a tenant's configuration before first access.
    pub fn register_tenant(&mut self, tenant_id: &str, config: TenantConfig) {
        self.configs.insert(tenant_id.to_string(), config);
    }

    /// Get an YantrikDB instance for a tenant, creating one if needed.
    pub fn get(&mut self, tenant_id: &str) -> Result<&YantrikDB> {
        if !self.instances.contains_key(tenant_id) {
            let db = self.open_tenant(tenant_id)?;
            self.instances.insert(tenant_id.to_string(), db);
        }
        Ok(self.instances.get(tenant_id).unwrap())
    }

    /// Get a mutable YantrikDB instance for a tenant.
    pub fn get_mut(&mut self, tenant_id: &str) -> Result<&mut YantrikDB> {
        if !self.instances.contains_key(tenant_id) {
            let db = self.open_tenant(tenant_id)?;
            self.instances.insert(tenant_id.to_string(), db);
        }
        Ok(self.instances.get_mut(tenant_id).unwrap())
    }

    /// Close and remove a tenant's instance from the cache.
    pub fn close_tenant(&mut self, tenant_id: &str) -> Result<bool> {
        if let Some(db) = self.instances.remove(tenant_id) {
            db.close()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// List all tenant IDs that have been opened.
    pub fn active_tenants(&self) -> Vec<&str> {
        self.instances.keys().map(|s| s.as_str()).collect()
    }

    /// List all tenant DB files discovered in the base directory.
    pub fn discovered_tenants(&self) -> Result<Vec<String>> {
        let mut tenants = Vec::new();
        let entries = std::fs::read_dir(&self.base_dir)
            .map_err(|e| YantrikDbError::SyncError(format!("failed to read tenant dir: {e}")))?;
        for entry in entries {
            let entry = entry
                .map_err(|e| YantrikDbError::SyncError(format!("failed to read dir entry: {e}")))?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.ends_with(".db") {
                tenants.push(name_str.trim_end_matches(".db").to_string());
            }
        }
        tenants.sort();
        Ok(tenants)
    }

    /// Close all tenant instances.
    pub fn close_all(&mut self) -> Result<()> {
        let ids: Vec<String> = self.instances.keys().cloned().collect();
        for id in ids {
            self.close_tenant(&id)?;
        }
        Ok(())
    }

    fn db_path(&self, tenant_id: &str) -> String {
        self.base_dir
            .join(format!("{tenant_id}.db"))
            .to_string_lossy()
            .to_string()
    }

    fn open_tenant(&self, tenant_id: &str) -> Result<YantrikDB> {
        let path = self.db_path(tenant_id);
        let config = self.configs.get(tenant_id);

        let dim = config
            .and_then(|c| c.embedding_dim)
            .unwrap_or(self.default_embedding_dim);

        match config.and_then(|c| c.encryption_key.as_ref()) {
            Some(key) => YantrikDB::new_encrypted(&path, dim, key),
            None => YantrikDB::new(&path, dim),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_isolation() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = TenantManager::new(dir.path().to_str().unwrap(), 8).unwrap();

        // Two tenants get separate DBs
        let db_a = mgr.get("tenant-a").unwrap();
        assert_eq!(db_a.stats(None).unwrap().active_memories, 0);

        let db_b = mgr.get("tenant-b").unwrap();
        assert_eq!(db_b.stats(None).unwrap().active_memories, 0);

        // Record in tenant-a
        let emb: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();
        let db_a = mgr.get("tenant-a").unwrap();
        db_a.record(
            "a-memory",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({}),
            &emb,
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

        // Tenant-b should still be empty
        let db_b = mgr.get("tenant-b").unwrap();
        assert_eq!(db_b.stats(None).unwrap().active_memories, 0);

        // Tenant-a has 1
        let db_a = mgr.get("tenant-a").unwrap();
        assert_eq!(db_a.stats(None).unwrap().active_memories, 1);
    }

    #[test]
    fn test_tenant_with_encryption() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = TenantManager::new(dir.path().to_str().unwrap(), 8).unwrap();

        let mut key = [0u8; 32];
        key[0] = 42;
        mgr.register_tenant(
            "secure",
            TenantConfig {
                encryption_key: Some(key),
                ..Default::default()
            },
        );

        let db = mgr.get("secure").unwrap();
        assert!(db.is_encrypted());
    }

    #[test]
    fn test_discovered_tenants() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = TenantManager::new(dir.path().to_str().unwrap(), 8).unwrap();

        mgr.get("alpha").unwrap();
        mgr.get("beta").unwrap();

        let discovered = mgr.discovered_tenants().unwrap();
        assert!(discovered.contains(&"alpha".to_string()));
        assert!(discovered.contains(&"beta".to_string()));
    }

    #[test]
    fn test_close_tenant() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = TenantManager::new(dir.path().to_str().unwrap(), 8).unwrap();

        mgr.get("temp").unwrap();
        assert_eq!(mgr.active_tenants().len(), 1);

        mgr.close_tenant("temp").unwrap();
        assert_eq!(mgr.active_tenants().len(), 0);

        // Can reopen
        mgr.get("temp").unwrap();
        assert_eq!(mgr.active_tenants().len(), 1);
    }
}
