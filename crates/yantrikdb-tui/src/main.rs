//! `yantrikdb-tui <store.db>`: a terminal explorer for one YantrikDB store.
//!
//! Three panes: namespaces, memories (newest first, or the results of a
//! semantic search), and an inspector for the selected memory with its
//! text, metadata, linked entities, the claims it backs, and its revision
//! history; the namespace's tasks sit under the inspector.
//!
//! THE SOURCE IS NEVER OPENED BY THE ENGINE. An ordinary engine open is not
//! read-only: it switches the journal to WAL, runs schema migrations and
//! entity/source-turn backfills, and may rewrite oplog payloads. So the
//! explorer takes a consistent SNAPSHOT first — a plain read-only connection
//! of the engine's own SQLite library runs the online backup into a private
//! temporary file — and constructs the engine on that copy. The source's
//! bytes, journal mode and schema stamp are untouched, an agent writing to it
//! from another process is undisturbed, and `r` takes a fresh snapshot.
//!
//! On the snapshot every read goes through non-reinforcing paths
//! (`recall(.., skip_reinforce = true, ..)`, `list_memories`, `get`, the
//! `engine::inspect` reads). Search embeds queries with the model the store
//! recorded, verified by digest; if that model cannot be attached, search is
//! disabled and the status line says why.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;
use yantrikdb::engine::materializer::{
    recommended_worker_count, spawn_all_workers, AllWorkerGuards,
};
use yantrikdb::YantrikDB;

const PAGE_SIZE: usize = 200;
const SEARCH_TOP_K: usize = 50;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pane {
    Namespaces,
    Memories,
    Inspector,
}

struct MemoryRow {
    rid: String,
    text: String,
    memory_type: String,
    importance: f64,
    created_at: f64,
    /// `Some(score)` for a search hit, `None` for a listing row.
    score: Option<f64>,
}

struct Inspector {
    rid: String,
    header: Vec<Line<'static>>,
    text: String,
    entities: Vec<String>,
    claims: Vec<String>,
    revisions: Vec<String>,
}

struct App {
    /// Background workers (materializers + compactor) for the engine on the
    /// COPY, exactly as the Python binding spawns them: without the
    /// compactor the delta tier fills at 256 and pending materialization on
    /// the copy would never run. Declared FIRST so they stop before the
    /// engine drops.
    workers: Option<AllWorkerGuards>,
    db: Arc<YantrikDB>,
    store: String,
    namespaces: Vec<(String, i64)>,
    ns_state: ListState,
    memories: Vec<MemoryRow>,
    mem_state: ListState,
    total: usize,
    page: usize,
    query: String,
    typing: bool,
    /// The query the current memory list came from, if it is a search.
    active_query: Option<String>,
    inspector: Option<Inspector>,
    inspector_scroll: u16,
    tasks: Vec<String>,
    pane: Pane,
    status: String,
    /// The store the user named; never opened by the engine.
    source: PathBuf,
    /// The private copy the engine is constructed on. Declared AFTER `db`
    /// so the engine drops before the directory is removed.
    snapshot: SnapshotGuard,
    snapshot_at: f64,
    search_state: SearchState,
    /// The namespace scope in force (set on Enter in the namespace pane),
    /// kept BY NAME so a refresh cannot silently move it.
    scope: Option<String>,
}

enum SearchState {
    Ready(String),
    Disabled(String),
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// A private, EXCLUSIVELY created directory that owns one snapshot copy.
///
/// Dropping the guard removes the directory: on normal exit, on refresh,
/// and on every error path of `take_snapshot`/`open_snapshot` (the guard is
/// created first and any failure returns through it). Removal is best
/// effort: a process killed outright never runs Drop, and a removal that
/// still fails after the retries leaves the directory; either way it carries
/// the dead process's id in its name and is never reused, because names are
/// random and created exclusively. The engine on the copy must be closed
/// before the guard drops; `App` declares the engine before the guard so
/// field-drop order does that on the panic path, and `App::close`/`refresh`
/// do it explicitly.
struct SnapshotGuard {
    dir: PathBuf,
}

impl SnapshotGuard {
    fn create() -> Result<Self, Box<dyn Error>> {
        let root = std::env::temp_dir().join("yantrikdb-tui");
        std::fs::create_dir_all(&root)?;
        for _ in 0..16 {
            // `create_dir` (not `create_dir_all`) is the exclusive step: it
            // fails if the name exists, so a leftover from a crashed process
            // or a concurrent explorer can never be reused.
            let dir = root.join(format!(
                "{}-{:016x}",
                std::process::id(),
                rand::random::<u64>()
            ));
            match std::fs::create_dir(&dir) {
                Ok(()) => return Ok(SnapshotGuard { dir }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        }
        Err("could not create a private snapshot directory".into())
    }

    fn db_path(&self) -> PathBuf {
        self.dir.join("snapshot.db")
    }
}

/// Remove `dir` and everything under it, best effort with retries (up to
/// ~2 s): on Windows the engine's file handles are released on the last
/// reference drop, which can trail `close()` while its worker threads wind
/// down. A process killed outright cannot run this at all.
fn remove_dir_with_retries(dir: &Path) {
    for attempt in 0..40 {
        match std::fs::remove_dir_all(dir) {
            Ok(()) => return,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
            Err(_) if attempt < 39 => std::thread::sleep(Duration::from_millis(50)),
            Err(_) => {}
        }
    }
}

impl Drop for SnapshotGuard {
    fn drop(&mut self) {
        // Leftovers from a killed process carry the dead pid in their name and
        // are never reused (names are random and created exclusively).
        remove_dir_with_retries(&self.dir);
    }
}

/// Consistent copy of `source` via SQLite's online backup, read through a
/// plain READ-ONLY connection of the engine's own library, into the guard's
/// directory. Nothing in the source is modified: no journal switch, no
/// migration, no backfill.
fn take_snapshot(source: &Path, guard: &SnapshotGuard) -> Result<PathBuf, Box<dyn Error>> {
    let dest = guard.db_path();
    let src = rusqlite::Connection::open_with_flags(
        source,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    let mut dst = rusqlite::Connection::open(&dest)?;
    {
        let backup = rusqlite::backup::Backup::new(&src, &mut dst)?;
        // ALL pages in ONE step. A paged backup is restarted by SQLite each
        // time another connection commits to the source between steps, so
        // against a busy writer it never finishes; a single step copies the
        // whole file under one read transaction (WAL readers block nobody)
        // and yields exactly one moment in time.
        match backup.step(-1)? {
            rusqlite::backup::StepResult::Done => {}
            other => return Err(format!("snapshot did not complete in one step: {other:?}").into()),
        }
    }
    dst.close().map_err(|(_, e)| e)?;
    src.close().map_err(|(_, e)| e)?;
    Ok(dest)
}

/// Everything one snapshot needs: the engine on the copy (with its worker
/// pool), the directory guard, and the search state.
struct Opened {
    db: Arc<YantrikDB>,
    workers: AllWorkerGuards,
    guard: SnapshotGuard,
    search_state: SearchState,
}

/// Snapshot, construct the engine on the copy, attach the store's recorded
/// embedder, spawn the engine's background workers. Any failure returns
/// through the guard, which removes whatever was copied.
fn open_snapshot(source: &Path) -> Result<Opened, Box<dyn Error>> {
    let guard = SnapshotGuard::create()?;
    let copy = take_snapshot(source, &guard)?;
    let mut db = YantrikDB::with_default(&copy.to_string_lossy())?;
    let search_state = attach_store_embedder(&mut db);
    let db = Arc::new(db);
    let workers = spawn_all_workers(&db, recommended_worker_count());
    Ok(Opened {
        db,
        workers,
        guard,
        search_state,
    })
}

/// Stop the workers, then close the engine. Dropping the guards signals the
/// worker threads and JOINS them, so by the time `try_unwrap` runs no worker
/// holds the engine; `close` consumes it, so the `Arc` must be unique. An
/// `Err` here therefore means some other owner still holds the engine, which
/// is a bug in the caller rather than worker lag; the fallback drops our
/// reference so the last owner's drop closes it.
fn shutdown(workers: Option<AllWorkerGuards>, db: Arc<YantrikDB>) -> Result<(), Box<dyn Error>> {
    drop(workers);
    match Arc::try_unwrap(db) {
        Ok(db) => db.close()?,
        Err(shared) => drop(shared),
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let store = match args.next() {
        Some(p) if p != "--help" && p != "-h" => p,
        _ => {
            eprintln!("usage: yantrikdb-tui <store.db>\n\n\
                       Read-only terminal explorer for one YantrikDB store.\n\
                       keys: Tab/Shift-Tab panes · ↑/↓ or j/k move · Enter open · / search · Esc clear · \
                       n/p page · PgUp/PgDn scroll · r refresh · q quit");
            std::process::exit(2);
        }
    };
    if !std::path::Path::new(&store).is_file() {
        eprintln!("no store file at {store}");
        std::process::exit(2);
    }
    let mut app = App::open(PathBuf::from(store))?;

    let mut terminal = ratatui::init();
    let result = run(&mut terminal, &mut app);
    ratatui::restore();
    // Close the engine, then drop the snapshot directory, whatever `run` did.
    let closed = app.close();
    result.and(closed)
}

fn run(terminal: &mut ratatui::DefaultTerminal, app: &mut App) -> Result<(), Box<dyn Error>> {
    loop {
        terminal.draw(|f| ui(f, app))?;
        if !event::poll(Duration::from_millis(250))? {
            continue;
        }
        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            if app.typing {
                match key.code {
                    KeyCode::Esc => {
                        app.typing = false;
                        app.query.clear();
                        app.active_query = None;
                        app.load_page(0);
                    }
                    KeyCode::Enter => {
                        app.typing = false;
                        app.search();
                    }
                    KeyCode::Backspace => {
                        app.query.pop();
                    }
                    KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                        app.query.push(c);
                    }
                    _ => {}
                }
                continue;
            }
            match key.code {
                KeyCode::Char('q') => return Ok(()),
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    return Ok(())
                }
                KeyCode::Tab => app.pane = next_pane(app.pane, true),
                KeyCode::BackTab => app.pane = next_pane(app.pane, false),
                KeyCode::Char('/') => {
                    app.typing = true;
                    app.pane = Pane::Memories;
                }
                KeyCode::Esc => {
                    if app.active_query.is_some() {
                        app.query.clear();
                        app.active_query = None;
                        app.load_page(0);
                    }
                }
                KeyCode::Char('r') => app.refresh(),
                KeyCode::Char('n') => {
                    if app.active_query.is_none() && (app.page + 1) * PAGE_SIZE < app.total {
                        app.load_page(app.page + 1);
                    }
                }
                KeyCode::Char('p') => {
                    if app.active_query.is_none() && app.page > 0 {
                        app.load_page(app.page - 1);
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => app.move_selection(1),
                KeyCode::Up | KeyCode::Char('k') => app.move_selection(-1),
                KeyCode::PageDown => app.inspector_scroll = app.inspector_scroll.saturating_add(10),
                KeyCode::PageUp => app.inspector_scroll = app.inspector_scroll.saturating_sub(10),
                KeyCode::Enter => app.activate(),
                _ => {}
            }
        }
    }
}

/// Search must embed queries with the SAME model that built the store's
/// vectors, and a matching dimension does not prove that. The store records
/// its embedder's name, digest and dimension; `adopt_embedder_identity`
/// compares the attached embedder's digest with the recorded one and
/// returns Ok only on a match (it never persists when an identity is
/// already recorded, and it is only called then). So: verify the attached
/// embedder; if it differs, attach the recorded model by name and verify
/// again; anything short of a verified match DISABLES search with the
/// reason, rather than serving a differently-spaced similarity as if it
/// were degraded-but-comparable.
fn attach_store_embedder(db: &mut YantrikDB) -> SearchState {
    let verify = |db: &YantrikDB| {
        db.adopt_embedder_identity()
            .map(|_| ())
            .map_err(|e| e.to_string())
    };
    // Capture the recorded identity BEFORE touching the embedder, and check
    // afterwards that attaching/verifying did not rewrite it: a verification
    // that reads back a record it just replaced would prove nothing.
    let recorded = match db.embedder_identity() {
        Ok(v) => v,
        Err(e) => {
            return SearchState::Disabled(format!(
                "search disabled: embedder identity unreadable: {e}"
            ))
        }
    };
    let Some((name, digest, dim)) = recorded else {
        return SearchState::Ready(
            "no recorded embedder identity · engine default attached, unverified".to_string(),
        );
    };
    let label = name.clone().unwrap_or_else(|| "bundled".to_string());
    let outcome = if verify(db).is_ok() {
        SearchState::Ready(format!("embedder {label} · {dim}-d · verified"))
    } else if let Some(n) = name {
        match db.set_embedder_named(&n) {
            Err(e) => SearchState::Disabled(format!(
                "search disabled: store needs embedder {n} ({dim}-d), which could not be \
                 attached: {e}"
            )),
            Ok(()) => match verify(db) {
                Ok(()) => SearchState::Ready(format!("embedder {n} · {dim}-d · verified")),
                Err(e) => SearchState::Disabled(format!(
                    "search disabled: {n} attached but its digest differs from the store's: {e}"
                )),
            },
        }
    } else {
        SearchState::Disabled(format!(
            "search disabled: the store's embedder is unnamed ({digest}, {dim}-d) and is \
             not the one attached"
        ))
    };
    // The record must be exactly what we started from.
    match db.embedder_identity() {
        Ok(Some((_, d2, dim2))) if d2 == digest && dim2 == dim => outcome,
        other => SearchState::Disabled(format!(
            "search disabled: the recorded embedder identity changed while attaching \
             (was {digest}/{dim}-d, now {other:?})"
        )),
    }
}

fn next_pane(p: Pane, forward: bool) -> Pane {
    match (p, forward) {
        (Pane::Namespaces, true) | (Pane::Inspector, false) => Pane::Memories,
        (Pane::Memories, true) | (Pane::Namespaces, false) => Pane::Inspector,
        (Pane::Inspector, true) | (Pane::Memories, false) => Pane::Namespaces,
    }
}

impl App {
    /// Snapshot `source` and build the explorer on the copy.
    fn open(source: PathBuf) -> Result<Self, Box<dyn Error>> {
        let Opened {
            db,
            workers,
            guard: snapshot,
            search_state,
        } = open_snapshot(&source)?;
        let mut app = App {
            workers: Some(workers),
            db,
            store: source.display().to_string(),
            namespaces: Vec::new(),
            ns_state: ListState::default(),
            memories: Vec::new(),
            mem_state: ListState::default(),
            total: 0,
            page: 0,
            query: String::new(),
            typing: false,
            active_query: None,
            inspector: None,
            inspector_scroll: 0,
            tasks: Vec::new(),
            pane: Pane::Memories,
            status: String::new(),
            source,
            snapshot,
            snapshot_at: now_secs(),
            search_state,
            scope: None,
        };
        app.load_namespaces()?;
        app.ns_state.select(Some(0));
        app.load_page(0);
        app.load_tasks();
        Ok(app)
    }

    /// "(all)" first, then every namespace: those with live (non-tombstoned)
    /// memories carry their count, and namespaces that only hold tasks are
    /// listed with 0, so the list is complete. Read through the engine's own
    /// connection on the SNAPSHOT: same library, same process.
    fn load_namespaces(&mut self) -> Result<(), Box<dyn Error>> {
        let mut counts: std::collections::BTreeMap<String, i64> = Default::default();
        {
            let conn = self.db.conn();
            let mut stmt = conn.prepare(
                "SELECT namespace, COUNT(*) FROM memories \
                 WHERE COALESCE(consolidation_status, 'active') != 'tombstoned' \
                 GROUP BY namespace",
            )?;
            for row in stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))? {
                let (name, n) = row?;
                counts.insert(name, n);
            }
            let has_tasks: i64 = conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'tasks'",
                [],
                |r| r.get(0),
            )?;
            if has_tasks > 0 {
                let mut stmt = conn.prepare("SELECT DISTINCT namespace FROM tasks")?;
                for row in stmt.query_map([], |r| r.get::<_, String>(0))? {
                    counts.entry(row?).or_insert(0);
                }
            }
        }
        let all: i64 = counts.values().sum();
        self.namespaces = std::iter::once(("(all)".to_string(), all))
            .chain(counts)
            .collect();
        Ok(())
    }

    /// The scope in force: `None` is every namespace.
    fn selected_namespace(&self) -> Option<String> {
        self.scope.clone()
    }

    /// The namespace under the cursor in the namespace pane.
    fn namespace_at_cursor(&self) -> Option<String> {
        let i = self.ns_state.selected().unwrap_or(0);
        if i == 0 {
            None
        } else {
            self.namespaces.get(i).map(|(n, _)| n.clone())
        }
    }

    /// Close the engine, then drop the snapshot directory. Field-drop order
    /// gives the same sequence on the panic path; this makes it explicit
    /// and reports a close failure.
    fn close(self) -> Result<(), Box<dyn Error>> {
        let App {
            workers,
            db,
            snapshot,
            ..
        } = self;
        shutdown(workers, db)?;
        drop(snapshot);
        Ok(())
    }

    fn header_note(&self) -> String {
        let search = match &self.search_state {
            SearchState::Ready(note) | SearchState::Disabled(note) => note.as_str(),
        };
        format!("snapshot {} · {search}", fmt_clock(self.snapshot_at))
    }

    fn load_page(&mut self, page: usize) {
        let ns = self.selected_namespace();
        match self.db.list_memories(
            PAGE_SIZE,
            page * PAGE_SIZE,
            None,
            None,
            ns.as_deref(),
            "created_at",
        ) {
            Ok((mems, total)) => {
                self.memories = mems
                    .into_iter()
                    .map(|m| MemoryRow {
                        rid: m.rid,
                        text: m.text,
                        memory_type: m.memory_type,
                        importance: m.importance,
                        created_at: m.created_at,
                        score: None,
                    })
                    .collect();
                self.total = total;
                self.page = page;
                self.active_query = None;
                self.mem_state.select(if self.memories.is_empty() {
                    None
                } else {
                    Some(0)
                });
                self.status = format!(
                    "{} memories · page {}/{}",
                    total,
                    page + 1,
                    total.div_ceil(PAGE_SIZE).max(1)
                );
            }
            Err(e) => self.status = format!("list failed: {e}"),
        }
        self.open_selected();
    }

    /// Semantic search over the selected namespace, never reinforcing what
    /// it touches (`skip_reinforce = true`), never expanding entities.
    fn search(&mut self) {
        let q = self.query.trim().to_string();
        if q.is_empty() {
            self.load_page(0);
            return;
        }
        if let SearchState::Disabled(reason) = &self.search_state {
            self.status = reason.clone();
            self.query.clear();
            return;
        }
        let ns = self.selected_namespace();
        let embedding = match self.db.embed(&q) {
            Ok(e) => e,
            Err(e) => {
                self.status = format!("embed failed: {e}");
                return;
            }
        };
        match self.db.recall(
            &embedding,
            SEARCH_TOP_K,
            None,
            None,
            false,
            false,
            Some(&q),
            true,
            ns.as_deref(),
            None,
            None,
            None,
            None,
            false,
            None,
            None,
        ) {
            Ok(hits) => {
                self.memories = hits
                    .into_iter()
                    .map(|h| MemoryRow {
                        rid: h.rid,
                        text: h.text,
                        memory_type: h.memory_type,
                        importance: h.importance,
                        created_at: h.created_at,
                        score: Some(h.score),
                    })
                    .collect();
                self.status = format!("{} hits for \"{}\" (Esc clears)", self.memories.len(), q);
                self.active_query = Some(q);
                self.mem_state.select(if self.memories.is_empty() {
                    None
                } else {
                    Some(0)
                });
                self.open_selected();
            }
            Err(e) => self.status = format!("search failed: {e}"),
        }
    }

    fn load_tasks(&mut self) {
        let Some(ns) = self.selected_namespace() else {
            self.tasks.clear();
            return;
        };
        self.tasks = match self.db.task_list(&ns, None) {
            Ok(ts) => ts
                .into_iter()
                .map(|t| format!("[{}] {} · {}", t.status, t.title, t.priority))
                .collect(),
            Err(e) => vec![format!("tasks unavailable: {e}")],
        };
    }

    /// Take a fresh snapshot of the source and rebuild every view on it,
    /// keeping the namespace scope by NAME (a namespace that has vanished
    /// falls back to all).
    fn refresh(&mut self) {
        match open_snapshot(&self.source) {
            Ok(Opened {
                db,
                workers,
                guard,
                search_state,
            }) => {
                // Stop the old workers and close the old engine BEFORE its
                // directory goes away.
                let old_workers = self.workers.replace(workers);
                let old_db = std::mem::replace(&mut self.db, db);
                let _ = shutdown(old_workers, old_db);
                let old_guard = std::mem::replace(&mut self.snapshot, guard);
                drop(old_guard);
                self.snapshot_at = now_secs();
                self.search_state = search_state;
            }
            Err(e) => {
                self.status = format!("refresh failed: {e}");
                return;
            }
        }
        if let Err(e) = self.load_namespaces() {
            self.status = format!("refresh failed: {e}");
        }
        let idx = match &self.scope {
            Some(name) => self.namespaces.iter().position(|(n, _)| n == name),
            None => Some(0),
        };
        if idx.is_none() {
            self.scope = None;
        }
        self.ns_state.select(idx.or(Some(0)));
        if self.active_query.is_some() {
            self.search();
        } else {
            self.load_page(
                self.page
                    .min(self.total.div_ceil(PAGE_SIZE).saturating_sub(1)),
            );
        }
        self.load_tasks();
    }

    fn move_selection(&mut self, delta: i32) {
        match self.pane {
            Pane::Namespaces => {
                let n = self.namespaces.len();
                if n == 0 {
                    return;
                }
                let i = self.ns_state.selected().unwrap_or(0) as i32 + delta;
                self.ns_state
                    .select(Some(i.clamp(0, n as i32 - 1) as usize));
            }
            Pane::Memories => {
                let n = self.memories.len();
                if n == 0 {
                    return;
                }
                let i = self.mem_state.selected().unwrap_or(0) as i32 + delta;
                self.mem_state
                    .select(Some(i.clamp(0, n as i32 - 1) as usize));
                self.open_selected();
            }
            Pane::Inspector => {
                self.inspector_scroll = if delta > 0 {
                    self.inspector_scroll.saturating_add(1)
                } else {
                    self.inspector_scroll.saturating_sub(1)
                };
            }
        }
    }

    fn activate(&mut self) {
        match self.pane {
            Pane::Namespaces => {
                self.scope = self.namespace_at_cursor();
                self.query.clear();
                self.load_page(0);
                self.load_tasks();
                self.pane = Pane::Memories;
            }
            Pane::Memories => {
                self.open_selected();
                self.pane = Pane::Inspector;
            }
            Pane::Inspector => {}
        }
    }

    fn open_selected(&mut self) {
        self.inspector_scroll = 0;
        let Some(i) = self.mem_state.selected() else {
            self.inspector = None;
            return;
        };
        let Some(row) = self.memories.get(i) else {
            self.inspector = None;
            return;
        };
        let rid = row.rid.clone();
        self.inspector = Some(self.inspect(&rid));
    }

    /// Everything the store holds about one memory, via read-only engine
    /// APIs: the record, its entity links, the claims it backs and its
    /// revision history.
    fn inspect(&self, rid: &str) -> Inspector {
        let dim = Style::default().fg(Color::DarkGray);
        let key = Style::default().fg(Color::Cyan);
        let mut header = Vec::new();
        let mut text = String::new();
        match self.db.get(rid) {
            Ok(Some(m)) => {
                header.push(Line::from(vec![
                    Span::styled(m.memory_type.clone(), key),
                    Span::styled(" · ", dim),
                    Span::raw(m.namespace.clone()),
                    Span::styled(" · importance ", dim),
                    Span::raw(format!("{:.2}", m.importance)),
                    Span::styled(" · ", dim),
                    Span::raw(m.consolidation_status.clone()),
                ]));
                header.push(Line::from(vec![
                    Span::styled("created ", dim),
                    Span::raw(fmt_day(m.created_at)),
                    Span::styled("  domain ", dim),
                    Span::raw(m.domain.clone()),
                    Span::styled("  source ", dim),
                    Span::raw(m.source.clone()),
                ]));
                header.push(Line::from(vec![Span::styled(rid.to_string(), dim)]));
                if !m.metadata.is_null() && m.metadata != serde_json::json!({}) {
                    header.push(Line::from(vec![
                        Span::styled("metadata ", dim),
                        Span::raw(m.metadata.to_string()),
                    ]));
                }
                text = m.text;
            }
            Ok(None) => header.push(Line::from(Span::styled("(record not found)", dim))),
            Err(e) => header.push(Line::from(Span::styled(format!("get failed: {e}"), dim))),
        }
        let entities = self.db.memory_entities(rid).unwrap_or_default();
        let claims = match self.db.claims_for_memory(rid) {
            Ok(cs) => cs
                .iter()
                .map(|c| {
                    let s = |k: &str| c.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let neg = c.get("polarity").and_then(|v| v.as_i64()) == Some(-1);
                    let window = match (
                        c.get("valid_from").and_then(|v| v.as_f64()),
                        c.get("valid_to").and_then(|v| v.as_f64()),
                    ) {
                        (Some(a), Some(b)) => format!(" [{} → {}]", fmt_day(a), fmt_day(b)),
                        (Some(a), None) => format!(" [from {}]", fmt_day(a)),
                        (None, Some(b)) => format!(" [until {}]", fmt_day(b)),
                        (None, None) => String::new(),
                    };
                    format!(
                        "{} —{} {} → {}  ({}, {}){}",
                        s("src"),
                        if neg { " NOT" } else { "" },
                        s("rel_type"),
                        s("dst"),
                        s("status_suggestion"),
                        s("extractor"),
                        window
                    )
                })
                .collect(),
            Err(e) => vec![format!("claims unavailable: {e}")],
        };
        let revisions = match self.db.revision_history(rid) {
            Ok(rs) => rs
                .iter()
                .map(|r| {
                    format!(
                        "r{} · {} · {}\n    was: {}",
                        r.revision_num,
                        fmt_day(r.applied_at),
                        r.reason,
                        r.prior_text
                    )
                })
                .collect(),
            Err(e) => vec![format!("revisions unavailable: {e}")],
        };
        Inspector {
            rid: rid.to_string(),
            header,
            text,
            entities,
            claims,
            revisions,
        }
    }
}

// ── drawing ─────────────────────────────────────────────────────────

fn ui(f: &mut Frame, app: &mut App) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(5),
            Constraint::Length(1),
        ])
        .split(f.area());
    draw_title(f, outer[0], app);
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(24),
            Constraint::Percentage(38),
            Constraint::Min(30),
        ])
        .split(outer[1]);
    draw_namespaces(f, cols[0], app);
    draw_memories(f, cols[1], app);
    draw_inspector(f, cols[2], app);
    draw_footer(f, outer[2], app);
}

fn border(app: &App, pane: Pane, title: String) -> Block<'static> {
    let style = if app.pane == pane {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    Block::default()
        .borders(Borders::ALL)
        .border_style(style)
        .title(title)
}

fn draw_title(f: &mut Frame, area: Rect, app: &App) {
    let line = Line::from(vec![
        Span::styled("yantrikdb", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::styled(app.store.clone(), Style::default().fg(Color::DarkGray)),
        Span::raw("  "),
        Span::styled(app.header_note(), Style::default().fg(Color::DarkGray)),
        Span::raw("  · "),
        Span::styled(
            "Tab panes · ↑↓ move · Enter open · / search · Esc clear · n/p page · PgUp/PgDn scroll · r refresh · q quit",
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn draw_namespaces(f: &mut Frame, area: Rect, app: &mut App) {
    let items: Vec<ListItem> = app
        .namespaces
        .iter()
        .map(|(name, n)| {
            ListItem::new(Line::from(vec![
                Span::raw(format!("{name:<14}")),
                Span::styled(format!("{n:>6}"), Style::default().fg(Color::DarkGray)),
            ]))
        })
        .collect();
    let list = List::new(items)
        .block(border(app, Pane::Namespaces, " namespaces ".into()))
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED));
    f.render_stateful_widget(list, area, &mut app.ns_state);
}

fn draw_memories(f: &mut Frame, area: Rect, app: &mut App) {
    let width = area.width.saturating_sub(4) as usize;
    let items: Vec<ListItem> = app
        .memories
        .iter()
        .map(|m| {
            let lead = match m.score {
                Some(s) => format!("{s:.2} "),
                None => format!("{} ", fmt_day(m.created_at)),
            };
            let room = width.saturating_sub(lead.len() + 6);
            let text = first_line(&m.text, room);
            ListItem::new(Line::from(vec![
                Span::styled(lead, Style::default().fg(Color::DarkGray)),
                Span::raw(text),
                Span::styled(
                    format!(
                        " {}{:.1}",
                        &m.memory_type[..1.min(m.memory_type.len())],
                        m.importance
                    ),
                    Style::default().fg(Color::DarkGray),
                ),
            ]))
        })
        .collect();
    let title = if app.typing {
        format!(" search: {}▏ ", app.query)
    } else if let Some(q) = &app.active_query {
        format!(" search: {q} ")
    } else {
        " memories · newest first ".to_string()
    };
    let list = List::new(items)
        .block(border(app, Pane::Memories, title))
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED));
    f.render_stateful_widget(list, area, &mut app.mem_state);
}

fn draw_inspector(f: &mut Frame, area: Rect, app: &App) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(6)])
        .split(area);
    let dim = Style::default().fg(Color::DarkGray);
    let head = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let mut lines: Vec<Line> = Vec::new();
    match &app.inspector {
        None => lines.push(Line::from(Span::styled("select a memory", dim))),
        Some(ins) => {
            lines.extend(ins.header.iter().cloned());
            lines.push(Line::from(""));
            for l in ins.text.lines() {
                lines.push(Line::from(l.to_string()));
            }
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("ENTITIES · {}", ins.entities.len()),
                head,
            )));
            lines.push(Line::from(if ins.entities.is_empty() {
                Span::styled("none linked", dim)
            } else {
                Span::raw(ins.entities.join(" · "))
            }));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("CLAIMS BACKED BY THIS MEMORY · {}", ins.claims.len()),
                head,
            )));
            if ins.claims.is_empty() {
                lines.push(Line::from(Span::styled("none active", dim)));
            }
            for c in &ins.claims {
                lines.push(Line::from(format!("• {c}")));
            }
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("REVISION HISTORY · {}", ins.revisions.len()),
                head,
            )));
            if ins.revisions.is_empty() {
                lines.push(Line::from(Span::styled("never corrected", dim)));
            }
            for r in &ins.revisions {
                for (i, l) in r.lines().enumerate() {
                    lines.push(Line::from(if i == 0 {
                        format!("• {l}")
                    } else {
                        l.to_string()
                    }));
                }
            }
        }
    }
    let title = match &app.inspector {
        Some(ins) => format!(" memory {} ", &ins.rid[..8.min(ins.rid.len())]),
        None => " inspector ".to_string(),
    };
    let para = Paragraph::new(lines)
        .block(border(app, Pane::Inspector, title))
        .wrap(Wrap { trim: false })
        .scroll((app.inspector_scroll, 0));
    f.render_widget(para, rows[0]);

    let task_lines: Vec<Line> = if app.selected_namespace().is_none() {
        vec![Line::from(Span::styled(
            "select a namespace to see its tasks",
            dim,
        ))]
    } else if app.tasks.is_empty() {
        vec![Line::from(Span::styled("no tasks", dim))]
    } else {
        app.tasks.iter().map(|t| Line::from(t.clone())).collect()
    };
    let tasks = Paragraph::new(task_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(format!(" tasks · {} ", app.tasks.len())),
        )
        .wrap(Wrap { trim: true });
    f.render_widget(tasks, rows[1]);
}

fn draw_footer(f: &mut Frame, area: Rect, app: &App) {
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            app.status.clone(),
            Style::default().fg(Color::DarkGray),
        ))),
        area,
    );
}

// ── helpers ─────────────────────────────────────────────────────────

fn first_line(text: &str, width: usize) -> String {
    let line = text.lines().next().unwrap_or("");
    let mut out: String = line.chars().take(width.max(1)).collect();
    if line.chars().count() > width {
        out.push('…');
    }
    out
}

/// `HH:MM:SS` (UTC) from unix seconds.
fn fmt_clock(secs: f64) -> String {
    let s = secs.max(0.0) as u64 % 86_400;
    format!("{:02}:{:02}:{:02}Z", s / 3600, (s % 3600) / 60, s % 60)
}

/// `YYYY-MM-DD` from unix seconds, no calendar crate: days-to-civil per
/// Howard Hinnant's algorithm.
fn fmt_day(secs: f64) -> String {
    if !secs.is_finite() || secs <= 0.0 {
        return "—".to_string();
    }
    let days = (secs / 86_400.0).floor() as i64;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_day_matches_known_dates() {
        assert_eq!(fmt_day(0.0), "—");
        assert_eq!(fmt_day(86_400.0), "1970-01-02");
        assert_eq!(fmt_day(1_789_300_000.0), "2026-09-13");
    }

    #[test]
    fn first_line_truncates_with_ellipsis() {
        assert_eq!(first_line("hello world\nsecond", 5), "hello…");
        assert_eq!(first_line("short", 10), "short");
    }

    // ── headless app tests: the data path without a terminal ────────

    /// A fixture directory that removes itself when the test's binding goes
    /// out of scope — including on a panic, which is how 231 such directories
    /// (330 MB of stores) accumulated in %TEMP% while this crate was written.
    /// Bind it (`let dir = temp_dir("x");`) for as long as the files are
    /// needed; a temporary would be dropped at the end of the statement.
    struct TempDir(PathBuf);

    impl TempDir {
        fn path(&self) -> &Path {
            &self.0
        }

        fn join(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            remove_dir_with_retries(&self.0);
        }
    }

    fn temp_dir(name: &str) -> TempDir {
        let d = std::env::temp_dir().join(format!("yantrikdb-tui-{}-{}", std::process::id(), name));
        std::fs::create_dir_all(&d).unwrap();
        TempDir(d)
    }

    /// A writer engine WITH its worker pool, as the Python binding constructs
    /// one: without the compactor the delta tier fills at 256 and every
    /// later write is refused with Backpressure forever (that was the
    /// "engine stall" seen during this work). Derefs to the engine.
    struct Writer {
        workers: Option<AllWorkerGuards>,
        db: Arc<YantrikDB>,
    }

    impl std::ops::Deref for Writer {
        type Target = YantrikDB;
        fn deref(&self) -> &YantrikDB {
            &self.db
        }
    }

    impl Writer {
        fn close(self) {
            let Writer { workers, db } = self;
            shutdown(workers, db).unwrap();
        }
    }

    /// A writer engine on `path` at the bundled dimension: no model download
    /// in CI, and the store records the bundled embedder's identity.
    fn writer(path: &Path) -> Writer {
        let db = Arc::new(
            YantrikDB::new(
                &path.to_string_lossy(),
                yantrikdb::embedder::BUNDLED_EMBEDDER_DIM,
            )
            .unwrap(),
        );
        let workers = spawn_all_workers(&db, recommended_worker_count());
        Writer {
            workers: Some(workers),
            db,
        }
    }

    /// Record one memory, honouring the engine's typed backpressure (the
    /// delta queue is finite; a tight loop meets it) the way a client would.
    fn record(db: &YantrikDB, ns: &str, text: &str) -> String {
        loop {
            match db.record_text(
                text,
                "semantic",
                0.6,
                0.0,
                604800.0,
                &serde_json::json!({}),
                ns,
                0.8,
                "general",
                "user",
                None,
            ) {
                Ok(rid) => return rid,
                Err(yantrikdb::YantrikDbError::Backpressure { retry_after_ms, .. }) => {
                    std::thread::sleep(Duration::from_millis(retry_after_ms));
                }
                Err(e) => panic!("record failed: {e}"),
            }
        }
    }

    /// Build the SOURCE with a writer engine and close it: two namespaces,
    /// a memory with a stated claim, a correction, an explicit entity link,
    /// one task. Returns the source path and the Dana rid.
    fn build_source(dir: &Path) -> (PathBuf, String) {
        let path = dir.join("source.db");
        let db = writer(&path);
        let dana = record(
            &db,
            "work",
            "Dana Okafor leads the Data Platform team at Northwind Analytics.",
        );
        record(
            &db,
            "work",
            "Helios is the nightly feature pipeline at Northwind Analytics.",
        );
        record(
            &db,
            "personal",
            "Ari Vasquez lives in Lisbon and cycles to the office.",
        );
        db.attach_claims(
            &dana,
            &[yantrikdb::StatedClaim {
                src: "Dana Okafor".into(),
                rel_type: "leads".into(),
                dst: "Data Platform team".into(),
                polarity: 1,
                modality: None,
                valid_from: None,
                valid_to: None,
            }],
        )
        .unwrap();
        db.correct(&dana, None, None, Some(0.9), None, "importance bump")
            .unwrap();
        db.link_memory_entity(&dana, "Northwind Analytics").unwrap();
        db.task_add("work", "Ship the atlas TUI", "high", None)
            .unwrap();
        db.close();
        (path, dana)
    }

    /// The returned `TempDir` must be held for the life of the test: dropping
    /// it removes the source store.
    fn seeded(name: &str) -> (App, String, PathBuf, TempDir) {
        let dir = temp_dir(name);
        let (path, dana) = build_source(dir.path());
        (App::open(path.clone()).unwrap(), dana, path, dir)
    }

    fn file_hash(path: &Path) -> Option<u64> {
        use std::hash::{Hash, Hasher};
        let bytes = std::fs::read(path).ok()?;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut h);
        Some(h.finish())
    }

    /// Hash of the source file plus the byte length of its WAL sidecar (0 when
    /// absent). A read-only connection on a WAL-mode database may create an
    /// EMPTY `-wal`/`-shm` pair on open; that carries no frames and no state,
    /// so "untouched" means the main file's bytes are identical and the WAL
    /// holds no more bytes than before.
    fn source_state(path: &Path) -> (Option<u64>, u64) {
        let wal = path.with_file_name(format!(
            "{}-wal",
            path.file_name().unwrap().to_string_lossy()
        ));
        (
            file_hash(path),
            std::fs::metadata(&wal).map(|m| m.len()).unwrap_or(0),
        )
    }

    fn meta_value(path: &Path, key: &str) -> String {
        let c =
            rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
                .unwrap();
        c.query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| {
            r.get::<_, String>(0)
        })
        .unwrap()
    }

    #[test]
    fn app_lists_namespaces_memories_and_inspects_without_a_terminal() {
        let (mut app, dana, source, _dir) = seeded("lists");
        assert_ne!(
            app.snapshot.db_path(),
            source,
            "the engine runs on a private copy"
        );
        assert!(
            matches!(app.search_state, SearchState::Ready(_)),
            "{}",
            app.header_note()
        );
        let names: Vec<&str> = app.namespaces.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["(all)", "personal", "work"]);
        assert_eq!(app.namespaces[0].1, 3);
        assert_eq!(app.memories.len(), 3, "every namespace at start");

        // Select "work" in the namespace pane and open it.
        app.pane = Pane::Namespaces;
        app.move_selection(1);
        app.move_selection(1);
        app.activate();
        assert_eq!(app.pane, Pane::Memories);
        assert_eq!(app.scope.as_deref(), Some("work"));
        assert_eq!(app.memories.len(), 2);
        assert!(
            app.tasks.iter().any(|t| t.contains("Ship the atlas TUI")),
            "{:?}",
            app.tasks
        );

        // Inspect the corrected memory.
        let i = app.memories.iter().position(|m| m.rid == dana).unwrap();
        app.mem_state.select(Some(i));
        app.open_selected();
        let ins = app.inspector.as_ref().unwrap();
        assert!(ins.text.contains("Dana Okafor"));
        assert!(
            ins.entities.iter().any(|e| e == "Northwind Analytics"),
            "{:?}",
            ins.entities
        );
        assert!(
            ins.claims
                .iter()
                .any(|c| c.contains("leads") && c.contains("Data Platform team")),
            "{:?}",
            ins.claims
        );
        assert_eq!(ins.revisions.len(), 1);
        assert!(ins.revisions[0].contains("importance bump"));
        assert!(ins.header.len() >= 3);
        app.close().unwrap();
    }

    #[test]
    fn search_returns_scored_hits_and_clearing_restores_the_listing() {
        let (mut app, dana, _, _dir) = seeded("search");
        app.query = "who leads the data platform team".into();
        app.search();
        assert!(app.active_query.is_some(), "{}", app.status);
        assert!(!app.memories.is_empty());
        assert!(app.memories.iter().all(|m| m.score.is_some()));
        assert!(
            app.memories.iter().take(2).any(|m| m.rid == dana),
            "Dana in the top two"
        );
        assert!(app.inspector.is_some(), "the top hit is opened");

        app.query.clear();
        app.active_query = None;
        app.load_page(0);
        assert!(app.memories.iter().all(|m| m.score.is_none()));
        assert_eq!(app.memories.len(), 3);
        app.close().unwrap();
    }

    #[test]
    fn opening_never_touches_the_source_even_with_an_older_schema_stamp() {
        let dir = temp_dir("untouched");
        let (path, _) = build_source(dir.path());
        // Pretend the source predates the current schema: an ordinary engine
        // open would MAX-stamp it back to current and migrate it.
        {
            let c = rusqlite::Connection::open(&path).unwrap();
            c.execute(
                "UPDATE meta SET value = '40' WHERE key = 'schema_version'",
                [],
            )
            .unwrap();
            c.close().unwrap();
        }
        let before = source_state(&path);
        let app = App::open(path.clone()).unwrap();
        assert_eq!(
            source_state(&path),
            before,
            "source bytes and WAL unchanged by opening"
        );
        assert_eq!(
            meta_value(&path, "schema_version"),
            "40",
            "source schema stamp untouched"
        );
        assert_eq!(app.total, 3);
        let copy_version: i64 = meta_value(&app.snapshot.db_path(), "schema_version")
            .parse()
            .unwrap();
        assert!(copy_version > 40, "the COPY is migrated; the source is not");
        app.close().unwrap();
        assert_eq!(source_state(&path), before, "still untouched after close");
    }

    /// OPEN-WRITER / WAL snapshot: a writer engine holds the source open (its
    /// rows live in the WAL) while the explorer snapshots it. Writes here
    /// happen before and after the backup, not during it; the concurrent case
    /// is the next test.
    #[test]
    fn snapshot_with_an_open_writer_is_a_moment_in_time_and_the_writer_continues() {
        let dir = temp_dir("writer");
        let path = dir.join("source.db");
        let w = writer(&path);
        for i in 0..12 {
            record(
                &w,
                "work",
                &format!("Nightly run {i} completed with no failures."),
            );
        }
        let mut app = App::open(path.clone()).unwrap();
        assert_eq!(app.total, 12);
        for i in 12..18 {
            record(&w, "ops", &format!("Backup {i} rotated the logs."));
        }
        assert_eq!(app.total, 12, "a snapshot does not move");
        app.refresh();
        assert_eq!(app.total, 18, "refresh takes a new snapshot");
        assert!(app.namespaces.iter().any(|(n, _)| n == "ops"));
        record(&w, "work", "The writer is undisturbed by the explorer.");
        app.close().unwrap();
        w.close();
    }

    /// CONCURRENT writer: a background thread keeps committing rows while
    /// the backups run. Every snapshot must be a valid store whose count
    /// lies between what was committed before and after.
    #[test]
    fn snapshot_while_a_background_writer_commits_is_consistent() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        let dir = temp_dir("concurrent");
        let path = dir.join("source.db");
        let w = writer(&path);
        for i in 0..20 {
            record(&w, "work", &format!("Seed row {i}."));
        }
        let stop = AtomicBool::new(false);
        let written = AtomicUsize::new(20);
        let mut apps: Vec<App> = Vec::new();
        std::thread::scope(|s| {
            s.spawn(|| {
                // Commit steadily, just under the engine's backpressure limit:
                // the point is commits DURING the backups, not saturating the
                // engine (a saturated single writer occasionally wedges inside
                // the engine, which is an engine finding, not this test's).
                let mut i = 20;
                while !stop.load(Ordering::Relaxed) {
                    match w.record_text(
                        &format!("Concurrent row {i}."),
                        "semantic",
                        0.6,
                        0.0,
                        604800.0,
                        &serde_json::json!({}),
                        "work",
                        0.8,
                        "general",
                        "user",
                        None,
                    ) {
                        Ok(_) => {
                            written.fetch_add(1, Ordering::Relaxed);
                            i += 1;
                            std::thread::sleep(Duration::from_millis(2));
                        }
                        Err(yantrikdb::YantrikDbError::Backpressure { retry_after_ms, .. }) => {
                            std::thread::sleep(Duration::from_millis(retry_after_ms));
                        }
                        Err(e) => panic!("writer failed: {e}"),
                    }
                }
            });
            for k in 0..3 {
                eprintln!(
                    "[concurrent] opening snapshot {k} (written so far {})",
                    written.load(Ordering::Relaxed)
                );
                apps.push(App::open(path.clone()).unwrap());
                eprintln!("[concurrent] opened snapshot {k}: total {}", apps[k].total);
            }
            stop.store(true, Ordering::Relaxed);
            eprintln!("[concurrent] stop set; waiting for the writer thread");
        });
        eprintln!("[concurrent] writer thread joined");
        let after = written.load(Ordering::Relaxed);
        assert!(
            after > 20,
            "the writer committed during the backups ({after})"
        );
        for app in &apps {
            assert!(
                app.total >= 20 && app.total <= after,
                "snapshot count {} within [20, {after}]",
                app.total
            );
            assert_eq!(app.namespaces[0].1 as usize, app.total, "a consistent copy");
        }
        let mut app = apps.pop().unwrap();
        app.open_selected();
        assert!(app.inspector.is_some(), "the copy is a working store");
        eprintln!("[concurrent] assertions passed; closing snapshot engines");
        for (k, app) in apps.into_iter().enumerate() {
            app.close().unwrap();
            eprintln!("[concurrent] closed snapshot engine {k}");
        }
        app.close().unwrap();
        eprintln!("[concurrent] closed last snapshot engine; final write");
        record(&w, "work", "The writer is undisturbed by the explorer.");
        eprintln!("[concurrent] closing the writer");
        w.close();
        eprintln!("[concurrent] writer closed");
    }

    #[test]
    fn refresh_keeps_the_scope_by_name_and_lists_new_and_task_only_namespaces() {
        let dir = temp_dir("scope");
        let (path, _) = build_source(dir.path());
        let mut app = App::open(path.clone()).unwrap();
        let work = app
            .namespaces
            .iter()
            .position(|(n, _)| n == "work")
            .unwrap();
        app.pane = Pane::Namespaces;
        app.ns_state.select(Some(work));
        app.activate();
        assert_eq!(app.scope.as_deref(), Some("work"));
        assert_eq!(app.memories.len(), 2);

        // Meanwhile a writer adds a namespace that sorts BEFORE "work" and a
        // task-only namespace.
        let w = writer(&path);
        record(&w, "aaa", "A new namespace appears.");
        w.task_add("planning", "Task-only namespace", "low", None)
            .unwrap();
        w.close();

        let first_dir = app.snapshot.dir.clone();
        app.refresh();
        assert!(
            !first_dir.exists(),
            "the previous snapshot directory is removed on refresh"
        );
        assert_eq!(
            app.scope.as_deref(),
            Some("work"),
            "scope kept by name, not by index"
        );
        assert_eq!(app.namespaces[app.ns_state.selected().unwrap()].0, "work");
        assert_eq!(app.memories.len(), 2);
        assert!(
            app.namespaces.iter().any(|(n, c)| n == "aaa" && *c == 1),
            "{:?}",
            app.namespaces
        );
        assert!(
            app.namespaces
                .iter()
                .any(|(n, c)| n == "planning" && *c == 0),
            "{:?}",
            app.namespaces
        );
        app.close().unwrap();
    }

    #[test]
    fn closing_removes_the_snapshot_directory() {
        let (app, _, _, _dir) = seeded("cleanup");
        let dir = app.snapshot.dir.clone();
        assert!(dir.join("snapshot.db").is_file());
        app.close().unwrap();
        assert!(!dir.exists(), "snapshot directory removed on close");
    }

    #[test]
    fn a_failed_snapshot_leaves_nothing_behind() {
        // Backup from a missing source.
        let guard = SnapshotGuard::create().unwrap();
        let dir = guard.dir.clone();
        let missing = temp_dir("missing");
        assert!(take_snapshot(&missing.join("does-not-exist.db"), &guard).is_err());
        drop(guard);
        assert!(!dir.exists(), "directory removed after a failed backup");

        // Backup from a file that is not a database.
        let bogus_dir = temp_dir("bogus");
        let bogus = bogus_dir.join("not-a-db.db");
        std::fs::write(&bogus, b"not a database").unwrap();
        let guard = SnapshotGuard::create().unwrap();
        let dir = guard.dir.clone();
        assert!(take_snapshot(&bogus, &guard).is_err());
        drop(guard);
        assert!(
            !dir.exists(),
            "directory removed after a failed backup of a non-database"
        );

        // The whole open path: nothing left behind either.
        assert!(open_snapshot(&bogus).is_err());
    }

    #[test]
    fn snapshot_directories_are_created_exclusively() {
        let a = SnapshotGuard::create().unwrap();
        let b = SnapshotGuard::create().unwrap();
        assert_ne!(a.dir, b.dir);
        assert!(a.dir.is_dir() && b.dir.is_dir());
        let (da, db_) = (a.dir.clone(), b.dir.clone());
        drop(a);
        drop(b);
        assert!(!da.exists() && !db_.exists());
    }

    /// MISSING-WORKER-POOL REPRODUCER (ignored by default). What looked like
    /// an engine stall during this work was a Rust-constructed engine
    /// without `spawn_all_workers`: with no compactor the delta tier fills
    /// at 256 and every later write is refused with Backpressure forever;
    /// the "stall" was that refusal behind a retry loop without a stop
    /// check. The engine-stall claim is withdrawn. This test is retained as
    /// a regression check that a worker-backed engine keeps draining under
    /// an unthrottled single writer (measured 992–1,345 rows per 8 s with
    /// zero backpressure hits) while three snapshots are taken and torn
    /// down. A watchdog prints the last phase marker and aborts after 60 s
    /// so any future stall is located, not merely waited out. Run:
    /// `cargo test -p yantrikdb-tui -- --ignored --nocapture engine_stall`
    #[test]
    #[ignore]
    fn engine_stall_repro_unthrottled_writer() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::sync::{Arc, Mutex};
        let phase = Arc::new(Mutex::new(String::from("start")));
        let mark = {
            let phase = phase.clone();
            move |m: String| {
                eprintln!("[stall {:>8.3}s] {m}", now_secs() % 1000.0);
                *phase.lock().unwrap() = m;
            }
        };
        // Watchdog: abort the whole process with the last phase after 60 s.
        {
            let phase = phase.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(60));
                eprintln!(
                    "[stall] WATCHDOG: still running after 60 s; last phase = {}",
                    phase.lock().unwrap()
                );
                std::process::abort();
            });
        }
        let stall_dir = temp_dir("stall");
        let path = stall_dir.join("source.db");
        let w = writer(&path);
        mark("seed".into());
        for i in 0..20 {
            record(&w, "work", &format!("Seed row {i}."));
        }
        let stop = AtomicBool::new(false);
        let written = AtomicUsize::new(20);
        let backpressure_hits = AtomicUsize::new(0);
        let mut apps: Vec<App> = Vec::new();
        std::thread::scope(|s| {
            s.spawn(|| {
                let mut i = 20;
                while !stop.load(Ordering::Relaxed) {
                    // UNTHROTTLED: exactly the loop that stalled.
                    match w.record_text(
                        &format!("Concurrent row {i}."),
                        "semantic",
                        0.6,
                        0.0,
                        604800.0,
                        &serde_json::json!({}),
                        "work",
                        0.8,
                        "general",
                        "user",
                        None,
                    ) {
                        Ok(_) => {
                            written.fetch_add(1, Ordering::Relaxed);
                            i += 1;
                        }
                        Err(yantrikdb::YantrikDbError::Backpressure { retry_after_ms, .. }) => {
                            backpressure_hits.fetch_add(1, Ordering::Relaxed);
                            std::thread::sleep(Duration::from_millis(retry_after_ms));
                        }
                        Err(e) => panic!("writer failed: {e}"),
                    }
                }
                *phase.lock().unwrap() = "writer loop exited".into();
            });
            for k in 0..3 {
                mark(format!(
                    "opening snapshot {k} (written {}, backpressure hits {})",
                    written.load(Ordering::Relaxed),
                    backpressure_hits.load(Ordering::Relaxed)
                ));
                apps.push(App::open(path.clone()).unwrap());
                mark(format!("opened snapshot {k}: total {}", apps[k].total));
            }
            // Keep the writer saturated for a while after the backups: the
            // stalls seen during development came under SUSTAINED
            // backpressure (hundreds of retry hits), not from a brief burst.
            let t0 = std::time::Instant::now();
            while backpressure_hits.load(Ordering::Relaxed) < 300
                && t0.elapsed() < Duration::from_secs(8)
            {
                std::thread::sleep(Duration::from_millis(100));
            }
            mark(format!(
                "sustained phase over: written {}, backpressure hits {} in {:.1}s",
                written.load(Ordering::Relaxed),
                backpressure_hits.load(Ordering::Relaxed),
                t0.elapsed().as_secs_f64()
            ));
            stop.store(true, Ordering::Relaxed);
            mark("stop set; waiting for the writer thread (it is inside record_text or its retry sleep)".into());
        });
        mark(format!(
            "writer thread joined; written {}, backpressure hits {}",
            written.load(Ordering::Relaxed),
            backpressure_hits.load(Ordering::Relaxed)
        ));
        for (k, app) in apps.into_iter().enumerate() {
            mark(format!("closing snapshot engine {k}"));
            app.close().unwrap();
        }
        mark("closing the writer".into());
        w.close();
        mark("done".into());
    }

    /// Manual smoke against a real store, when one is named:
    /// `YANTRIKDB_TUI_SMOKE_STORE=path cargo test -p yantrikdb-tui -- --nocapture smoke`
    #[test]
    fn smoke_opens_the_store_named_by_env() {
        let Ok(path) = std::env::var("YANTRIKDB_TUI_SMOKE_STORE") else {
            return;
        };
        let mut app = App::open(PathBuf::from(path)).unwrap();
        eprintln!("{}", app.header_note());
        eprintln!("namespaces: {:?}", app.namespaces);
        eprintln!("first page: {} of {}", app.memories.len(), app.total);
        app.query = "who leads the data platform team".into();
        app.search();
        for m in app.memories.iter().take(3) {
            eprintln!(
                "  hit {:.3}  {}",
                m.score.unwrap_or(0.0),
                first_line(&m.text, 70)
            );
        }
        if let Some(ins) = &app.inspector {
            eprintln!(
                "inspector: entities={} claims={} revisions={}",
                ins.entities.len(),
                ins.claims.len(),
                ins.revisions.len()
            );
            for c in &ins.claims {
                eprintln!("  claim: {c}");
            }
        }
        assert!(!app.namespaces.is_empty());
        app.close().unwrap();
    }
}
