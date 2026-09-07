//! Issue #225 — a second SQLite library with the store open in THIS process.
//!
//! The engine links its own SQLite (`rusqlite` `bundled`). When another
//! SQLite library in the same process — Python's stdlib `sqlite3`, a
//! system `libsqlite3` pulled in by some extension — opens the same store,
//! the result is silent page aliasing, not a locking error: both
//! libraries serialise writers with POSIX advisory (`fcntl`) locks, and
//! the kernel scopes those locks per PROCESS, so the foreign library's
//! unlock releases the engine's locks and the two writers interleave
//! their WAL commits (sqlite.org/howtocorrupt.html, "multiple copies of
//! SQLite linked into the same application"). "Sequential per record"
//! does not save anyone: the materializer keeps writing after `record()`
//! returns. Measured on our own CI on 2026-09-06 (an `entities` page
//! written where a `memories` page belongs) and reported from the field
//! on 2026-09-07 after a day of blaming hooks and concurrency.
//!
//! The engine cannot make a foreign library respect its locks. It CAN
//! notice that a second SQLite instance has the store open in this
//! process, and stop committing while that is true — which is enough,
//! because a single writer at a time is exactly what the locks were
//! meant to guarantee.
//!
//! **Detector (Linux).** Every SQLite instance maps the store's `-shm`
//! file itself, region by region, with `mmap(offset = i × region)`. One
//! library shares one shm node across all of its connections, so each
//! offset appears at most once per library in `/proc/self/maps`. The same
//! `<store>-shm` path mapped at the same offset TWICE means two libraries
//! have the store open in this process. That is the hazard exactly, read
//! in a few hundred microseconds, with no way for it to fire on the
//! engine's own connections. macOS has no `/proc`: the same rule is read
//! through libproc (`proc_pidinfo(PROC_PIDREGIONPATHINFO)`, one call per
//! region, so it is rescanned less often). Windows locks are per handle
//! and does not have the problem; there the detector reports
//! `supported = false` and the mode is inert.
//!
//! **Commits from outside this engine (all platforms).** A second engine
//! process, the `sqlite3` CLI, a backup tool: legitimate, serialised by
//! the kernel — and still the only way a store changes under this engine
//! without going through it. SQLite's `PRAGMA data_version` on the writer
//! connection changes only when ANOTHER connection commits, and every
//! engine write goes through that one connection, so a change is exactly
//! "someone else committed". Each is counted
//! (`stats().foreign_commits_detected_since_boot`) and asks for one
//! `PRAGMA quick_check`, run off the writer by the materializer (or
//! `integrity_check()` on demand); a failed check taints the store the
//! same way a foreign instance does, because writing onto a corrupt file
//! only spreads the damage.
//!
//! **Modes**, durable in `meta.foreign_sqlite_mode`, default `refuse`:
//! `off` never scans; `warn` scans, logs the transition and counts
//! (`stats().foreign_sqlite_detected_since_boot`); `refuse` additionally
//! makes every engine write fail with `ForeignSqliteInstance` while the
//! condition holds — a pre-check at the public write entry points for a
//! typed error, and a commit hook on the writer connection as the hard
//! guarantee that no engine commit, the materializer's included, lands
//! in an interleaved WAL. Reads continue.
//!
//! **Why a detection latches until the engine is reopened.** Measured on
//! WSL 2026-09-07: when the foreign connection closed, its library —
//! believing itself the last user, because its lock table is its own —
//! UNLINKED the `-shm` file (and, as SQLite does on last close, is free to
//! checkpoint and delete the WAL). The engine was left mapping a deleted
//! shm: consistent with itself, but any other process would now create a
//! fresh one and diverge. So after a detection the store is tainted for
//! this engine instance, writes stay refused, and the operator reopens
//! the engine (which recovers the WAL/shm cleanly) after checking
//! integrity. `stats().foreign_sqlite_tainted` says so.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, Instant};

use crate::error::{Result, YantrikDbError};

/// How often the commit hook rescans. A commit inside this window reuses
/// the last verdict; the first commit after it pays the scan. Linux reads
/// one file; macOS makes one libproc call per region, hence the longer
/// window there.
#[cfg(target_os = "macos")]
const RESCAN_AFTER: Duration = Duration::from_millis(1000);
#[cfg(not(target_os = "macos"))]
const RESCAN_AFTER: Duration = Duration::from_millis(200);

/// `SQLITE_CONSTRAINT_COMMITHOOK`: the extended result code SQLite returns
/// when a commit hook aborts the commit. `SQLITE_CONSTRAINT | (2 << 8)` =
/// 531 (pinned by a test below; the docs' table is easy to misread).
pub const SQLITE_CONSTRAINT_COMMITHOOK: i32 = 531;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForeignSqliteMode {
    Off,
    Warn,
    Refuse,
}

impl ForeignSqliteMode {
    /// A malformed persisted or caller-supplied mode is a typed error,
    /// never a silent `Off`.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "off" => Ok(Self::Off),
            "warn" => Ok(Self::Warn),
            "refuse" => Ok(Self::Refuse),
            other => Err(YantrikDbError::InvalidInput(format!(
                "foreign_sqlite_mode: expected off|warn|refuse, got {other:?}"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Warn => "warn",
            Self::Refuse => "refuse",
        }
    }

    pub fn as_u8(self) -> u8 {
        match self {
            Self::Off => 0,
            Self::Warn => 1,
            Self::Refuse => 2,
        }
    }

    pub fn from_u8(v: u8) -> Self {
        match v {
            2 => Self::Refuse,
            1 => Self::Warn,
            _ => Self::Off,
        }
    }
}

/// Shared between the engine and the writer connection's commit hook.
pub(crate) struct ForeignSqliteGuard {
    /// The store's `-shm` path as `/proc/self/maps` prints it (canonical).
    /// `None` for `:memory:` or an unsupported platform: the guard is inert.
    shm_path: Option<PathBuf>,
    /// The store path for messages.
    store: String,
    mode: AtomicU8,
    /// The last scan found a foreign instance.
    active: AtomicBool,
    /// A scan found one at some point since this engine opened. Latched:
    /// the foreign library's close may have unlinked the shm/WAL under us.
    tainted: AtomicBool,
    detected_since_boot: AtomicU64,
    refused_since_boot: AtomicU64,
    last_scan: parking_lot::Mutex<Option<Instant>>,
    /// `PRAGMA data_version` of the writer connection at the last check;
    /// -1 until the first read.
    last_data_version: AtomicI64,
    foreign_commits_detected_since_boot: AtomicU64,
    /// A foreign commit was seen and no integrity check has run since.
    integrity_check_pending: AtomicBool,
    integrity_checks_since_boot: AtomicU64,
    /// The last `PRAGMA quick_check` result (`ok`, or the first problem).
    last_integrity: parking_lot::Mutex<Option<String>>,
}

impl ForeignSqliteGuard {
    pub(crate) fn new(db_path: &str, mode: ForeignSqliteMode) -> Self {
        let shm_path =
            if db_path == ":memory:" || !cfg!(any(target_os = "linux", target_os = "macos")) {
                None
            } else {
                std::fs::canonicalize(db_path).ok().map(|p| {
                    let mut s = p.into_os_string();
                    s.push("-shm");
                    PathBuf::from(s)
                })
            };
        Self {
            shm_path,
            store: db_path.to_string(),
            mode: AtomicU8::new(mode.as_u8()),
            active: AtomicBool::new(false),
            tainted: AtomicBool::new(false),
            detected_since_boot: AtomicU64::new(0),
            refused_since_boot: AtomicU64::new(0),
            last_scan: parking_lot::Mutex::new(None),
            last_data_version: AtomicI64::new(-1),
            foreign_commits_detected_since_boot: AtomicU64::new(0),
            integrity_check_pending: AtomicBool::new(false),
            integrity_checks_since_boot: AtomicU64::new(0),
            last_integrity: parking_lot::Mutex::new(None),
        }
    }

    pub(crate) fn foreign_commits_detected_since_boot(&self) -> u64 {
        self.foreign_commits_detected_since_boot
            .load(Ordering::Relaxed)
    }

    pub(crate) fn integrity_check_pending(&self) -> bool {
        self.integrity_check_pending.load(Ordering::Relaxed)
    }

    pub(crate) fn integrity_checks_since_boot(&self) -> u64 {
        self.integrity_checks_since_boot.load(Ordering::Relaxed)
    }

    pub(crate) fn last_integrity(&self) -> Option<String> {
        self.last_integrity.lock().clone()
    }

    /// Read the writer connection's `PRAGMA data_version` and compare it
    /// with the last reading. A change means a commit that did not go
    /// through this engine. Returns whether one was seen.
    pub(crate) fn note_data_version(&self, conn: &rusqlite::Connection) -> bool {
        let Ok(v) = conn.query_row("PRAGMA data_version", [], |r| r.get::<_, i64>(0)) else {
            return false;
        };
        let prev = self.last_data_version.swap(v, Ordering::Relaxed);
        if prev >= 0 && prev != v {
            self.foreign_commits_detected_since_boot
                .fetch_add(1, Ordering::Relaxed);
            self.integrity_check_pending.store(true, Ordering::Relaxed);
            tracing::info!(
                store = %self.store,
                "a commit reached this store without going through this engine \
                 (another process, most likely); an integrity check is queued"
            );
            return true;
        }
        false
    }

    /// Record a `PRAGMA quick_check` result. Anything but `ok` taints the
    /// store: writing onto a corrupt file only spreads the damage.
    pub(crate) fn note_integrity(&self, result: &str) {
        self.integrity_check_pending.store(false, Ordering::Relaxed);
        self.integrity_checks_since_boot
            .fetch_add(1, Ordering::Relaxed);
        *self.last_integrity.lock() = Some(result.to_string());
        if result != "ok" {
            self.tainted.store(true, Ordering::Relaxed);
            tracing::error!(
                store = %self.store,
                result = %result,
                "integrity check failed after a commit from outside this engine; \
                 writes are refused until the store is repaired and the engine reopened"
            );
        }
    }

    pub(crate) fn mode(&self) -> ForeignSqliteMode {
        ForeignSqliteMode::from_u8(self.mode.load(Ordering::Relaxed))
    }

    pub(crate) fn set_mode(&self, mode: ForeignSqliteMode) {
        self.mode.store(mode.as_u8(), Ordering::Relaxed);
    }

    /// Whether this platform and store can be scanned at all.
    pub(crate) fn supported(&self) -> bool {
        self.shm_path.is_some()
    }

    pub(crate) fn active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    pub(crate) fn tainted(&self) -> bool {
        self.tainted.load(Ordering::Relaxed)
    }

    pub(crate) fn detected_since_boot(&self) -> u64 {
        self.detected_since_boot.load(Ordering::Relaxed)
    }

    pub(crate) fn refused_since_boot(&self) -> u64 {
        self.refused_since_boot.load(Ordering::Relaxed)
    }

    /// Scan now. Returns whether a foreign instance has the store open.
    /// Logs on the transition into the condition, once, at error level.
    pub(crate) fn scan(&self) -> bool {
        let Some(shm) = &self.shm_path else {
            return false;
        };
        if self.mode() == ForeignSqliteMode::Off {
            return false;
        }
        let found = foreign_shm_mapped(shm);
        *self.last_scan.lock() = Some(Instant::now());
        let was = self.active.swap(found, Ordering::Relaxed);
        if found {
            self.detected_since_boot.fetch_add(1, Ordering::Relaxed);
            self.tainted.store(true, Ordering::Relaxed);
            if !was {
                tracing::error!(
                    store = %self.store,
                    mode = self.mode().as_str(),
                    "another SQLite library has this store open in this process \
                     (issue #225): POSIX locks are per process, so its unlock releases \
                     the engine's and the two writers would interleave WAL commits. \
                     Writes are refused until this engine is reopened: close the foreign \
                     connection, check integrity, reopen."
                );
            }
        } else if was {
            tracing::warn!(
                store = %self.store,
                "foreign SQLite instance gone, but its close may have unlinked the shm/WAL \
                 under this engine: writes stay refused until the engine is reopened"
            );
        }
        found
    }

    /// The commit-hook cadence: rescan at most every `RESCAN_AFTER`.
    pub(crate) fn scan_cached(&self) -> bool {
        let due = self
            .last_scan
            .lock()
            .is_none_or(|t| t.elapsed() >= RESCAN_AFTER);
        if due {
            self.scan()
        } else {
            self.active()
        }
    }

    /// Commit-hook body: `true` aborts the commit.
    pub(crate) fn commit_should_abort(&self) -> bool {
        self.scan_cached();
        let abort = decide(self.mode(), self.tainted());
        if abort {
            self.refused_since_boot.fetch_add(1, Ordering::Relaxed);
        }
        abort
    }

    /// The typed pre-check at a public write entry point.
    pub(crate) fn check_write(&self) -> Result<()> {
        self.scan_cached();
        if decide(self.mode(), self.tainted()) {
            self.refused_since_boot.fetch_add(1, Ordering::Relaxed);
            return Err(YantrikDbError::ForeignSqliteInstance {
                path: self.store.clone(),
            });
        }
        Ok(())
    }
}

/// Is this SQLite error the writer connection's commit hook refusing a
/// commit? Used by the error conversion so the abort reaches callers as
/// the typed `ForeignSqliteInstance`, in Rust and across pyo3 alike.
pub(crate) fn is_commit_hook_abort(err: &rusqlite::Error) -> bool {
    matches!(err, rusqlite::Error::SqliteFailure(e, _) if e.extended_code == SQLITE_CONSTRAINT_COMMITHOOK)
}

/// The one rule: only `refuse` turns a detection into an abort.
fn decide(mode: ForeignSqliteMode, found: bool) -> bool {
    matches!(mode, ForeignSqliteMode::Refuse) && found
}

#[cfg(target_os = "linux")]
fn foreign_shm_mapped(shm: &Path) -> bool {
    match std::fs::read_to_string("/proc/self/maps") {
        Ok(maps) => duplicate_shm_offsets(&maps, shm) > 0,
        Err(_) => false,
    }
}

#[cfg(target_os = "macos")]
fn foreign_shm_mapped(shm: &Path) -> bool {
    duplicate_shm_offsets_macos(shm) > 0
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn foreign_shm_mapped(_shm: &Path) -> bool {
    false
}

/// macOS: walk this process's regions with libproc and count the file
/// offsets at which `shm_path` is mapped more than once — the same rule as
/// the Linux reader. Layouts follow XNU's `sys/proc_info.h`
/// (`proc_regioninfo`, `proc_regionwithpathinfo`, flavor 8). A walk that
/// stops advancing ends the scan, so a misbehaving kernel call can only
/// under-report, never invent a duplicate.
#[cfg(target_os = "macos")]
fn duplicate_shm_offsets_macos(shm_path: &Path) -> usize {
    use std::collections::HashMap;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct ProcRegionInfo {
        pri_protection: u32,
        pri_max_protection: u32,
        pri_inheritance: u32,
        pri_flags: u32,
        pri_offset: u64,
        pri_behavior: u32,
        pri_user_wired_count: u32,
        pri_user_tag: u32,
        pri_pages_resident: u32,
        pri_pages_shared_now_private: u32,
        pri_pages_swapped_out: u32,
        pri_pages_dirtied: u32,
        pri_ref_count: u32,
        pri_shadow_depth: u32,
        pri_share_mode: u32,
        pri_private_pages_resident: u32,
        pri_shared_pages_resident: u32,
        pri_obj_id: u32,
        pri_depth: u32,
        pri_address: u64,
        pri_size: u64,
    }
    #[repr(C)]
    struct ProcRegionWithPathInfo {
        prp_prinfo: ProcRegionInfo,
        prp_vip: libc::vnode_info_path,
    }
    const PROC_PIDREGIONPATHINFO: libc::c_int = 8;
    const MAX_REGIONS: usize = 50_000;

    let target = shm_path.to_string_lossy();
    let pid = std::process::id() as libc::c_int;
    let mut seen: HashMap<u64, usize> = HashMap::new();
    let mut address: u64 = 0;
    for _ in 0..MAX_REGIONS {
        let mut info = std::mem::MaybeUninit::<ProcRegionWithPathInfo>::zeroed();
        let size = std::mem::size_of::<ProcRegionWithPathInfo>() as libc::c_int;
        // SAFETY: libproc fills at most `size` bytes of a properly sized,
        // zeroed buffer; a non-positive return means no more regions.
        let got = unsafe {
            libc::proc_pidinfo(
                pid,
                PROC_PIDREGIONPATHINFO,
                address,
                info.as_mut_ptr() as *mut libc::c_void,
                size,
            )
        };
        if got <= 0 {
            break;
        }
        // SAFETY: the call succeeded and wrote the struct.
        let info = unsafe { info.assume_init() };
        let next = info
            .prp_prinfo
            .pri_address
            .saturating_add(info.prp_prinfo.pri_size);
        if next <= address {
            break; // not advancing: stop rather than loop
        }
        address = next;
        // SAFETY: vip_path is a NUL-terminated C string inside the struct.
        let path = unsafe {
            std::ffi::CStr::from_ptr(info.prp_vip.vip_path.as_ptr() as *const libc::c_char)
        }
        .to_string_lossy();
        if path == target {
            *seen.entry(info.prp_prinfo.pri_offset).or_insert(0) += 1;
        }
    }
    seen.values().filter(|&&n| n > 1).count()
}

/// Count the file offsets at which `shm_path` is mapped more than once in
/// a `/proc/self/maps` listing. One SQLite library maps each shm region
/// once; a second library maps the same offsets again.
///
/// Line shape: `start-end perms offset dev inode      pathname`, the
/// pathname possibly containing spaces and possibly suffixed
/// ` (deleted)` (an unlinked shm is not the live one — ignored).
pub(crate) fn duplicate_shm_offsets(maps: &str, shm_path: &Path) -> usize {
    let target = shm_path.to_string_lossy();
    let mut seen: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for line in maps.lines() {
        let Some((offset, path)) = maps_fields(line) else {
            continue;
        };
        if path != &*target {
            continue; // a different file, a longer path, or ` (deleted)`
        }
        *seen.entry(offset).or_insert(0) += 1;
    }
    seen.values().filter(|&&n| n > 1).count()
}

/// `(offset, pathname)` of one `/proc/self/maps` line: the third
/// whitespace-separated field, and everything after the fifth (the
/// pathname keeps its internal spaces).
fn maps_fields(line: &str) -> Option<(&str, &str)> {
    let mut rest = line;
    let mut offset = None;
    for i in 0..5 {
        let trimmed = rest.trim_start();
        let end = trimmed.find(char::is_whitespace).unwrap_or(trimmed.len());
        if end == 0 {
            return None;
        }
        if i == 2 {
            offset = Some(&trimmed[..end]);
        }
        rest = &trimmed[end..];
    }
    let path = rest.trim_start().trim_end();
    if path.is_empty() {
        return None;
    }
    offset.map(|o| (o, path))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE_LIBRARY: &str = "\
7f0000000000-7f0000008000 rw-s 00000000 08:01 42   /var/lib/y/memory.db-shm
7f0000008000-7f0000010000 rw-s 00008000 08:01 42   /var/lib/y/memory.db-shm
7f0000010000-7f0000018000 rw-s 00010000 08:01 42   /var/lib/y/memory.db-shm
7f0000020000-7f0000028000 rw-s 00000000 08:01 43   /var/lib/y/other.db-shm
7f0000030000-7f0000038000 rw-s 00000000 08:01 44   /var/lib/y/memory.db-shm (deleted)
7f0000040000-7f0000048000 r-xp 00000000 08:01 45   /usr/lib/libsqlite3.so.0
";

    #[test]
    fn one_library_maps_each_region_once() {
        assert_eq!(
            duplicate_shm_offsets(ONE_LIBRARY, Path::new("/var/lib/y/memory.db-shm")),
            0
        );
    }

    #[test]
    fn a_second_library_maps_the_same_offset_again() {
        let two = format!(
            "{ONE_LIBRARY}7f0000050000-7f0000058000 rw-s 00000000 08:01 42   /var/lib/y/memory.db-shm\n"
        );
        assert_eq!(
            duplicate_shm_offsets(&two, Path::new("/var/lib/y/memory.db-shm")),
            1
        );
        // A path with spaces still resolves exactly.
        let spaced = "7f0-7f1 rw-s 00000000 08:01 9   /home/a b/m.db-shm\n\
                      7f2-7f3 rw-s 00000000 08:01 9   /home/a b/m.db-shm\n\
                      7f4-7f5 rw-s 00000000 08:01 9   /home/xa b/m.db-shm\n";
        assert_eq!(
            duplicate_shm_offsets(spaced, Path::new("/home/a b/m.db-shm")),
            1
        );
        assert_eq!(duplicate_shm_offsets(spaced, Path::new("b/m.db-shm")), 0);
    }

    #[test]
    fn only_refuse_aborts_and_a_bad_mode_is_loud() {
        assert!(!decide(ForeignSqliteMode::Off, true));
        assert!(!decide(ForeignSqliteMode::Warn, true));
        assert!(decide(ForeignSqliteMode::Refuse, true));
        assert!(!decide(ForeignSqliteMode::Refuse, false));
        for (s, m) in [
            ("off", ForeignSqliteMode::Off),
            ("warn", ForeignSqliteMode::Warn),
            ("refuse", ForeignSqliteMode::Refuse),
        ] {
            assert_eq!(ForeignSqliteMode::parse(s).unwrap(), m);
            assert_eq!(m.as_str(), s);
            assert_eq!(ForeignSqliteMode::from_u8(m.as_u8()), m);
        }
        assert!(ForeignSqliteMode::parse("enforce").is_err());
    }

    /// Pins the assumption the error mapping rests on: an aborting commit
    /// hook surfaces as SQLITE_CONSTRAINT_COMMITHOOK (531).
    #[test]
    fn an_aborting_commit_hook_surfaces_as_the_commithook_code() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE t (x INTEGER)").unwrap();
        conn.commit_hook(Some(|| true));
        let err = conn.execute("INSERT INTO t VALUES (1)", []).unwrap_err();
        match err {
            rusqlite::Error::SqliteFailure(e, msg) => {
                assert_eq!(
                    e.extended_code, SQLITE_CONSTRAINT_COMMITHOOK,
                    "{e:?} {msg:?}"
                );
            }
            other => panic!("unexpected error shape: {other:?}"),
        }
        assert!(is_commit_hook_abort(
            &conn.execute("INSERT INTO t VALUES (2)", []).unwrap_err()
        ));
    }

    #[test]
    fn a_memory_store_is_never_scanned() {
        let g = ForeignSqliteGuard::new(":memory:", ForeignSqliteMode::Refuse);
        assert!(!g.supported());
        assert!(!g.scan());
        assert!(g.check_write().is_ok());
        assert!(!g.commit_should_abort());
    }
}
