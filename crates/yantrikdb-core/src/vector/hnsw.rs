//! Rust-native HNSW (Hierarchical Navigable Small World) vector index.
//!
//! Implements the Malkov-Yashunin algorithm with cosine distance,
//! incremental insert, tombstone-based deletion, and configurable parameters.
//!
//! This is a purpose-built index for YantrikDB's cognitive memory engine —
//! single-threaded, derived from SQLite as source of truth, and rebuilt on startup.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::error::Result;

// ── Distance functions ──

/// One-pass dot product with f64 accumulation. Routed through the
/// runtime-dispatched SIMD kernels (AVX2+FMA when the CPU has them,
/// ILP-unrolled scalar otherwise) — see `vector::kernels` for the
/// dispatch policy and the numerical-equivalence contract.
#[inline]
pub(crate) fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    crate::vector::kernels::dot_f64(a, b)
}

/// Euclidean norm with f64 accumulation. Computed ONCE per stored vector at
/// insert time (v0.9.3 speed work) — recomputing it per distance call was
/// two-thirds of every comparison's flops.
#[inline]
pub(crate) fn norm_f64(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt()
}

/// Cosine distance assembled from a precomputed dot product and norms.
///
/// Guards zero AND NaN norms: `norm == 0.0` misses NaN (`NaN == 0.0` is
/// false), and `f64::clamp` preserves a NaN `self`, so a NaN norm would let
/// a NaN distance escape — which then poisons callers' sort comparators
/// (issue #60). `!(norm > 0.0)` is true for NaN, 0.0, and negatives (every
/// comparison with NaN is false), so it catches all three. A NaN in the dot
/// product with finite positive norms is impossible for finite inputs; NaN
/// *elements* make the corresponding norm NaN, which this guard catches.
#[inline]
pub(crate) fn dist_from(dot: f64, norm_a: f64, norm_b: f64) -> f64 {
    if !(norm_a > 0.0) || !(norm_b > 0.0) {
        return 1.0;
    }
    // Clamp to [0.0, 2.0] to handle floating-point rounding. With both norms
    // > 0.0 and finite, the ratio cannot be NaN here, so clamp is NaN-safe.
    (1.0 - (dot / (norm_a * norm_b))).clamp(0.0, 2.0)
}

/// Cosine distance: 1.0 - cosine_similarity. Convenience form for callers
/// without precomputed norms (the brute-force oracle + tests); the HNSW hot
/// path uses `dist_from` with stored node norms instead.
#[inline]
fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    dist_from(dot_f64(a, b), norm_f64(a), norm_f64(b))
}

// ── Heap helpers ──

/// An entry in the candidate/result heaps, ordered by distance.
#[derive(Clone)]
struct Candidate {
    idx: usize,
    distance: f64,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for Candidate {}

/// Min-heap ordering (smallest distance first).
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse: BinaryHeap is a max-heap, so reverse for min-heap behavior
        other.distance.total_cmp(&self.distance)
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Max-heap entry (largest distance first) for maintaining top-k.
#[derive(Clone)]
struct FarCandidate {
    idx: usize,
    distance: f64,
}

impl PartialEq for FarCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for FarCandidate {}

impl Ord for FarCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.total_cmp(&other.distance)
    }
}
impl PartialOrd for FarCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ── Node ──

#[derive(Clone)]
struct HnswNode {
    /// The embedding vector.
    embedding: Vec<f32>,
    /// Precomputed Euclidean norm of `embedding` (v0.9.3 speed work).
    /// Stored vectors never change without going through insert, so this is
    /// computed exactly once instead of on every distance comparison. The
    /// index is rebuilt from SQLite on open, so no migration is needed.
    norm: f64,
    /// Connections per layer: neighbors[layer] = vec of neighbor indices.
    neighbors: Vec<Vec<usize>>,
    /// Whether this node has been deleted.
    tombstoned: bool,
}

// ── HnswIndex ──

/// Fix (j): the production level-RNG seed. The VALUE is arbitrary — any
/// constant yields the same expected graph quality as an entropy draw —
/// what matters is that it never varies across opens of the same file.
/// Changing it changes every approximate result set by a hair, so it
/// moves only with a release note, never silently.
const HNSW_LEVEL_SEED: u64 = 0x11DB_5EED;

/// A Rust-native HNSW vector index.
#[derive(Clone)]
pub struct HnswIndex {
    dim: usize,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    ef_search: usize,
    ml: f64,
    entry_point: Option<usize>,
    max_layer: usize,
    nodes: Vec<HnswNode>,
    rid_to_idx: HashMap<String, usize>,
    idx_to_rid: Vec<String>,
    free_list: Vec<usize>,
    active_count: usize,
    rng: SmallRng,
    /// Layer-0 incoming edge count per node slot. Search always finishes
    /// with an ef-bounded exploration of layer 0, so a node with zero
    /// incoming layer-0 edges (and not the entry point) can never be
    /// returned — it exists in the index and is silently unfindable.
    /// Pure-distance pruning creates exactly that: in a dense cluster,
    /// every neighbor of a new node may immediately prune its backlink
    /// because it already holds `max_m` closer edges (found live: a
    /// 65-record pack dropped a different record per mount). Pruning
    /// consults this count and never removes an edge whose target it
    /// would orphan. Counts include edges FROM tombstoned nodes — the
    /// traversal expands through tombstones, so those edges still
    /// provide reachability.
    incoming0: Vec<usize>,
}

impl HnswIndex {
    /// Create a new HNSW index with default parameters.
    ///
    /// ef_search=200 ensures good recall quality for indices up to ~100K entries.
    pub fn new(dim: usize) -> Self {
        Self::with_params(dim, 16, 200, 200)
    }

    /// Create a new HNSW index with custom parameters.
    ///
    /// **v0.10 Phase 0 determinism seam:** under the NON-DEFAULT `testing`
    /// cargo feature, the `YANTRIKDB_HNSW_SEED` env var (if set) seeds the
    /// level RNG so trace tests can vary the draw. This check lives HERE —
    /// the single constructor every construction path funnels through
    /// (open rebuild, compaction, reembed) — so seeds reach all of them
    /// (sol Q6).
    ///
    /// **Fix (j), 2026-08-06 — the eleventh determinism source.** The level
    /// RNG was `from_entropy()` per instance, so every OPEN of the same
    /// file built a topologically different graph, and approximate search
    /// returned a different fetch_k pool TAIL (capture: 3 distinct
    /// 100-pools across 6 drift-free opens, swapped rids at positions
    /// 51–99 — invisible to the k=50 probe that once refuted this
    /// suspect; shared candidates bit-identical in distance, so no float
    /// mechanism at all). Boost/reserve lanes then lifted differing tail
    /// candidates into top-5: hermes determinism_burst arm B read 2–4
    /// orderings; with the RNG seeded it reads 1 (3/3 bursts — the
    /// conviction experiment is this fix's proof). Level randomization
    /// needs a good DISTRIBUTION, not unpredictability: a fixed seed keeps
    /// the same expected graph quality and makes same-file → same-graph
    /// hold, because the open path inserts ORDER BY rid (Phase 0 seam)
    /// with a fresh constructor RNG.
    pub fn with_params(dim: usize, m: usize, ef_construction: usize, ef_search: usize) -> Self {
        #[cfg(feature = "testing")]
        let rng = match std::env::var("YANTRIKDB_HNSW_SEED")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
        {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::seed_from_u64(HNSW_LEVEL_SEED),
        };
        #[cfg(not(feature = "testing"))]
        let rng = SmallRng::seed_from_u64(HNSW_LEVEL_SEED);
        Self {
            dim,
            m,
            m_max0: m * 2,
            ef_construction,
            ef_search,
            ml: 1.0 / (m as f64).ln(),
            entry_point: None,
            max_layer: 0,
            nodes: Vec::new(),
            rid_to_idx: HashMap::new(),
            idx_to_rid: Vec::new(),
            free_list: Vec::new(),
            active_count: 0,
            rng,
            incoming0: Vec::new(),
        }
    }

    /// Explicitly-seeded constructor (deterministic level assignment) for
    /// trace fixtures and reproducibility tests. Always available; the
    /// env-var path above is the way SEEDS reach the engine's internal
    /// construction sites without threading a parameter through every
    /// caller.
    pub fn with_params_seeded(
        dim: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        seed: u64,
    ) -> Self {
        let mut idx = Self::with_params(dim, m, ef_construction, ef_search);
        idx.rng = SmallRng::seed_from_u64(seed);
        idx
    }

    /// Number of active (non-tombstoned) entries.
    /// Embedding dimension this index was constructed with.
    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.active_count
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.active_count == 0
    }

    /// Insert a vector keyed by rid.
    pub fn insert(&mut self, rid: &str, embedding: &[f32]) -> Result<()> {
        assert_eq!(embedding.len(), self.dim, "embedding dimension mismatch");

        // If rid already exists and is tombstoned, resurrect it
        if let Some(&existing_idx) = self.rid_to_idx.get(rid) {
            let node = &mut self.nodes[existing_idx];
            if node.tombstoned {
                node.embedding = embedding.to_vec();
                node.norm = norm_f64(embedding);
                node.tombstoned = false;
                self.active_count += 1;
                // Re-connect by inserting into the graph at its existing layers
                let level = node.neighbors.len().saturating_sub(1);
                self.connect_node(existing_idx, level);
                return Ok(());
            }
            // Already exists and active — update embedding in-place
            node.embedding = embedding.to_vec();
            node.norm = norm_f64(embedding);
            return Ok(());
        }

        // Assign a random level
        let level = self.random_level();

        // Allocate node index
        let idx = if let Some(free_idx) = self.free_list.pop() {
            // Reuse a freed slot. incoming0[free_idx] is deliberately NOT
            // reset: stale edges from other nodes still point at this slot
            // and now reach the new occupant, so the count is still true.
            let neighbors = (0..=level).map(|_| Vec::new()).collect();
            self.nodes[free_idx] = HnswNode {
                embedding: embedding.to_vec(),
                norm: norm_f64(embedding),
                neighbors,
                tombstoned: false,
            };
            self.idx_to_rid[free_idx] = rid.to_string();
            free_idx
        } else {
            // Append new slot
            let neighbors = (0..=level).map(|_| Vec::new()).collect();
            self.nodes.push(HnswNode {
                embedding: embedding.to_vec(),
                norm: norm_f64(embedding),
                neighbors,
                tombstoned: false,
            });
            self.idx_to_rid.push(rid.to_string());
            self.incoming0.push(0);
            self.nodes.len() - 1
        };

        self.rid_to_idx.insert(rid.to_string(), idx);
        self.active_count += 1;

        // First node: set as entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(idx);
            self.max_layer = level;
            return Ok(());
        }

        self.connect_node(idx, level);

        // Update entry point if this node has a higher level
        if level > self.max_layer {
            self.entry_point = Some(idx);
            self.max_layer = level;
        }

        Ok(())
    }

    /// Distance from a query (with its precomputed norm) to a stored node.
    /// The node's norm comes from the insert-time cache — one pass over the
    /// vectors instead of the historical three.
    #[inline]
    fn dist_to(&self, query: &[f32], qnorm: f64, idx: usize) -> f64 {
        let node = &self.nodes[idx];
        dist_from(dot_f64(query, &node.embedding), qnorm, node.norm)
    }

    /// Connect a node into the graph at the given level.
    fn connect_node(&mut self, idx: usize, level: usize) {
        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return,
        };

        let query = self.nodes[idx].embedding.clone();
        // The "query" here is a stored node — its norm is already cached.
        let qnorm = self.nodes[idx].norm;
        let mut current_ep = ep;

        // Phase 1: Greedy descent from top layer to level+1
        for lc in (level + 1..=self.max_layer).rev() {
            current_ep = self.greedy_closest(&query, qnorm, current_ep, lc);
        }

        // Phase 2: Insert at layers min(level, max_layer) down to 0
        let insert_top = level.min(self.max_layer);
        let mut ep_candidates = vec![current_ep];

        for lc in (0..=insert_top).rev() {
            let max_m = if lc == 0 { self.m_max0 } else { self.m };
            let ef = self.ef_construction;

            // Search for neighbors at this layer
            let nearest = self.search_layer(&query, qnorm, &ep_candidates, ef, lc, Some(idx));

            // Select top M neighbors
            let selected: Vec<usize> = nearest.iter().take(max_m).map(|c| c.idx).collect();

            // Connect bidirectionally, keeping layer-0 incoming counts
            // true: decrement targets of any outgoing edges being
            // overwritten (the resurrection path re-connects a node that
            // already has edges), then count the new ones.
            if lc == 0 {
                for old_i in 0..self.nodes[idx].neighbors[0].len() {
                    let old = self.nodes[idx].neighbors[0][old_i];
                    self.incoming0[old] = self.incoming0[old].saturating_sub(1);
                }
            }
            self.nodes[idx].neighbors[lc] = selected.clone();
            if lc == 0 {
                for &t in &selected {
                    self.incoming0[t] += 1;
                }
            }
            for &neighbor_idx in &selected {
                if self.nodes[neighbor_idx].neighbors.len() > lc {
                    self.nodes[neighbor_idx].neighbors[lc].push(idx);
                    if lc == 0 {
                        self.incoming0[idx] += 1;
                    }
                    // Prune if over capacity
                    if self.nodes[neighbor_idx].neighbors[lc].len() > max_m {
                        self.prune_neighbors(neighbor_idx, lc, max_m);
                    }
                }
            }

            // Propagate entry points for next layer
            ep_candidates = selected;
            if ep_candidates.is_empty() {
                ep_candidates = vec![current_ep];
            }
        }
    }

    /// Prune a node's neighbors at a given layer to max_m connections.
    ///
    /// At layer 0 the prune is orphan-safe: an edge whose target has no
    /// OTHER incoming layer-0 edge is kept even over capacity, because
    /// dropping it would make that target unreachable to every future
    /// search — present in the index, silently unfindable. The over-
    /// capacity this allows is bounded and self-heals as later inserts
    /// give protected targets more incoming edges.
    fn prune_neighbors(&mut self, node_idx: usize, layer: usize, max_m: usize) {
        let node_emb = self.nodes[node_idx].embedding.clone();
        let node_norm = self.nodes[node_idx].norm;
        let mut neighbors_with_dist: Vec<(usize, f64)> = self.nodes[node_idx].neighbors[layer]
            .iter()
            .filter(|&&n| !self.nodes[n].tombstoned)
            .map(|&n| (n, self.dist_to(&node_emb, node_norm, n)))
            .collect();
        neighbors_with_dist.sort_by(|a, b| a.1.total_cmp(&b.1));

        if layer == 0 {
            // Tombstone-filtered edges vanish from the list; their targets
            // lose an incoming edge and the count must say so.
            for old_i in 0..self.nodes[node_idx].neighbors[0].len() {
                let old = self.nodes[node_idx].neighbors[0][old_i];
                if self.nodes[old].tombstoned {
                    self.incoming0[old] = self.incoming0[old].saturating_sub(1);
                }
            }
            let mut kept: Vec<usize> = Vec::with_capacity(max_m + 2);
            for (i, &(n, _)) in neighbors_with_dist.iter().enumerate() {
                if i < max_m || self.incoming0[n] <= 1 {
                    kept.push(n);
                } else {
                    self.incoming0[n] = self.incoming0[n].saturating_sub(1);
                }
            }
            self.nodes[node_idx].neighbors[0] = kept;
            return;
        }

        neighbors_with_dist.truncate(max_m);
        self.nodes[node_idx].neighbors[layer] =
            neighbors_with_dist.iter().map(|&(n, _)| n).collect();
    }

    /// Search for the k nearest neighbors of query.
    /// Returns (rid, distance) pairs sorted by distance (ascending).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f64)>> {
        if self.active_count == 0 || self.entry_point.is_none() {
            return Ok(vec![]);
        }

        let ep = self.entry_point.unwrap();
        let mut current_ep = ep;
        // Query norm computed ONCE per search (v0.9.3 speed work) instead of
        // inside every distance comparison the traversal makes.
        let qnorm = norm_f64(query);

        // Phase 1: Greedy descent from top layer to layer 1
        for lc in (1..=self.max_layer).rev() {
            current_ep = self.greedy_closest(query, qnorm, current_ep, lc);
        }

        // Phase 2: Search layer 0 with ef_search candidates
        // Ensure ef >= k so we can return enough results; also respect configured ef_search.
        let ef = self.ef_search.max(k * 2);
        let nearest = self.search_layer(query, qnorm, &[current_ep], ef, 0, None);

        // Return top-k non-tombstoned results
        let mut results: Vec<(String, f64)> = Vec::with_capacity(k);
        for c in &nearest {
            if !self.nodes[c.idx].tombstoned {
                results.push((self.idx_to_rid[c.idx].clone(), c.distance));
                if results.len() >= k {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Guarantee that every active node is reachable from the entry point
    /// over layer 0, force-linking any that are not. Returns how many
    /// nodes needed rescuing.
    ///
    /// The in-degree guard in `prune_neighbors` keeps pruning from
    /// orphaning nodes one edge at a time, but it is a LOCAL invariant:
    /// two nodes whose only incoming edges come from each other satisfy
    /// it while forming an island the search can never enter (found by
    /// the regression test below — the dense-cluster outliers paired up).
    /// Reachability is a property of the whole graph, so it has to be
    /// checked as one: BFS from the entry point (traversing through
    /// tombstones, which search also does), then link each unreachable
    /// node from its nearest reachable one. Called after bulk builds
    /// (host open, pack mount, compaction, reembed); incremental inserts
    /// between rebuilds are covered by the in-degree guard.
    pub fn ensure_all_reachable(&mut self) -> usize {
        let Some(ep) = self.entry_point else {
            return 0;
        };
        let n = self.nodes.len();

        // Pass 1: restore layer-0 symmetry.
        //
        // Reachability from the entry point is NOT the property search
        // needs. `search` first descends the upper layers greedily and
        // only then explores layer 0, starting from wherever that descent
        // landed — a different node for every query. Layer-0 edges are
        // directed, and pruning removes one direction at a time, so a
        // node can keep an outgoing edge into the graph while nothing
        // points back at it from where the descent begins. That is how
        // the regression test still failed (~25% of builds, always on a
        // far outlier) after the in-degree guard: connectivity from the
        // entry point held, and search still could not find the node.
        //
        // Making layer-0 edges symmetric collapses the distinction —
        // directed reachability becomes undirected reachability, so one
        // connected component means search finds the node no matter where
        // the descent lands. Insertion already links both ways; this only
        // repairs what pruning broke, and it runs at rebuild time.
        let mut restore: Vec<(usize, usize)> = Vec::new();
        for u in 0..n {
            if self.nodes[u].neighbors.is_empty() {
                continue;
            }
            for &v in &self.nodes[u].neighbors[0] {
                if v < n && !self.nodes[v].neighbors[0].contains(&u) {
                    restore.push((v, u));
                }
            }
        }
        for (v, u) in restore {
            if !self.nodes[v].neighbors[0].contains(&u) {
                self.nodes[v].neighbors[0].push(u);
                self.incoming0[u] += 1;
            }
        }

        let mut seen = vec![false; n];
        let mut stack = vec![ep];
        seen[ep] = true;
        while let Some(cur) = stack.pop() {
            if let Some(nbrs) = self.nodes[cur].neighbors.first() {
                for &n in nbrs {
                    if n < self.nodes.len() && !seen[n] {
                        seen[n] = true;
                        stack.push(n);
                    }
                }
            }
        }

        let mut rescued = 0;
        for i in 0..self.nodes.len() {
            if seen[i] || self.nodes[i].tombstoned {
                continue;
            }
            // Nearest reachable node by brute force — orphans are rare and
            // this runs at rebuild time, so exactness beats speed here.
            let emb = self.nodes[i].embedding.clone();
            let norm = self.nodes[i].norm;
            let mut best: Option<(usize, f64)> = None;
            for (j, seen_j) in seen.iter().enumerate() {
                if !seen_j {
                    continue;
                }
                let d = self.dist_to(&emb, norm, j);
                if best.is_none_or(|(_, bd)| d < bd) {
                    best = Some((j, d));
                }
            }
            let Some((j, _)) = best else {
                continue;
            };
            // Link both ways, or pass 1's symmetry invariant would be
            // broken by the very repair that depends on it.
            self.nodes[j].neighbors[0].push(i);
            self.incoming0[i] += 1;
            self.nodes[i].neighbors[0].push(j);
            self.incoming0[j] += 1;
            rescued += 1;
            // The rescued node's outgoing edges may reach further
            // stranded nodes — mark its whole component reachable so a
            // chain of orphans costs one rescue edge, not one each.
            seen[i] = true;
            let mut st = vec![i];
            while let Some(c) = st.pop() {
                if let Some(nbrs) = self.nodes[c].neighbors.first() {
                    for &n in nbrs {
                        if n < self.nodes.len() && !seen[n] {
                            seen[n] = true;
                            st.push(n);
                        }
                    }
                }
            }
        }
        rescued
    }

    /// Remove a vector by rid (tombstone-based).
    pub fn remove(&mut self, rid: &str) -> bool {
        if let Some(&idx) = self.rid_to_idx.get(rid) {
            if !self.nodes[idx].tombstoned {
                self.nodes[idx].tombstoned = true;
                self.active_count -= 1;
                self.free_list.push(idx);
                return true;
            }
        }
        false
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.rid_to_idx.clear();
        self.idx_to_rid.clear();
        self.free_list.clear();
        self.entry_point = None;
        self.max_layer = 0;
        self.active_count = 0;
        self.incoming0.clear();
    }

    // ── Internal helpers ──

    /// Assign a random level for a new node.
    fn random_level(&mut self) -> usize {
        let r: f64 = self.rng.gen();
        let level = (-r.ln() * self.ml).floor() as usize;
        level.min(32) // Cap at 32 layers
    }

    /// Greedy descent: find the closest node to query at a given layer.
    /// `qnorm` is the query's precomputed norm (one pass per public op).
    fn greedy_closest(&self, query: &[f32], qnorm: f64, entry: usize, layer: usize) -> usize {
        let mut current = entry;
        let mut current_dist = self.dist_to(query, qnorm, current);

        loop {
            let mut changed = false;
            if layer < self.nodes[current].neighbors.len() {
                for &neighbor in &self.nodes[current].neighbors[layer] {
                    if neighbor >= self.nodes.len() || self.nodes[neighbor].tombstoned {
                        continue;
                    }
                    let dist = self.dist_to(query, qnorm, neighbor);
                    if dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    /// Search a single layer starting from entry points.
    /// Returns candidates sorted by distance (ascending).
    /// `qnorm` is the query's precomputed norm (one pass per public op).
    fn search_layer(
        &self,
        query: &[f32],
        qnorm: f64,
        entry_points: &[usize],
        ef: usize,
        layer: usize,
        exclude_idx: Option<usize>,
    ) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        // Min-heap of candidates to explore
        let mut candidates = BinaryHeap::new();
        // Max-heap of best results so far
        let mut results = BinaryHeap::<FarCandidate>::new();

        for &ep in entry_points {
            if ep >= self.nodes.len() || visited.contains(&ep) {
                continue;
            }
            visited.insert(ep);
            let dist = self.dist_to(query, qnorm, ep);

            if exclude_idx != Some(ep) && !self.nodes[ep].tombstoned {
                candidates.push(Candidate {
                    idx: ep,
                    distance: dist,
                });
                results.push(FarCandidate {
                    idx: ep,
                    distance: dist,
                });
            } else {
                // Still add to candidates for traversal but not to results
                candidates.push(Candidate {
                    idx: ep,
                    distance: dist,
                });
            }
        }

        while let Some(closest) = candidates.pop() {
            // Check if the closest candidate is farther than the worst result
            let worst_dist = results.peek().map(|r| r.distance).unwrap_or(f64::MAX);
            if closest.distance > worst_dist && results.len() >= ef {
                break;
            }

            // Expand neighbors at this layer
            let node = &self.nodes[closest.idx];
            if layer < node.neighbors.len() {
                for &neighbor in &node.neighbors[layer] {
                    if neighbor >= self.nodes.len() || visited.contains(&neighbor) {
                        continue;
                    }
                    visited.insert(neighbor);
                    let dist = self.dist_to(query, qnorm, neighbor);
                    let worst_dist = results.peek().map(|r| r.distance).unwrap_or(f64::MAX);

                    if dist < worst_dist || results.len() < ef {
                        candidates.push(Candidate {
                            idx: neighbor,
                            distance: dist,
                        });

                        if exclude_idx != Some(neighbor) && !self.nodes[neighbor].tombstoned {
                            results.push(FarCandidate {
                                idx: neighbor,
                                distance: dist,
                            });
                            if results.len() > ef {
                                results.pop(); // Remove farthest
                            }
                        }
                    }
                }
            }
        }

        // Convert max-heap to sorted vec (ascending distance)
        let mut sorted: Vec<Candidate> = results
            .into_iter()
            .map(|fc| Candidate {
                idx: fc.idx,
                distance: fc.distance,
            })
            .collect();
        sorted.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        sorted
    }
}

// ── BruteForceIndex (correctness oracle for testing) ──

/// Brute-force vector index for testing HNSW recall quality.
pub struct BruteForceIndex {
    dim: usize,
    entries: Vec<(String, Vec<f32>, bool)>, // (rid, embedding, tombstoned)
    rid_to_idx: HashMap<String, usize>,
}

impl BruteForceIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entries: Vec::new(),
            rid_to_idx: HashMap::new(),
        }
    }

    pub fn insert(&mut self, rid: &str, embedding: &[f32]) {
        assert_eq!(embedding.len(), self.dim);
        if let Some(&idx) = self.rid_to_idx.get(rid) {
            self.entries[idx].1 = embedding.to_vec();
            self.entries[idx].2 = false;
        } else {
            let idx = self.entries.len();
            self.entries
                .push((rid.to_string(), embedding.to_vec(), false));
            self.rid_to_idx.insert(rid.to_string(), idx);
        }
    }

    pub fn remove(&mut self, rid: &str) -> bool {
        if let Some(&idx) = self.rid_to_idx.get(rid) {
            if !self.entries[idx].2 {
                self.entries[idx].2 = true;
                return true;
            }
        }
        false
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f64)> {
        let mut scored: Vec<(String, f64)> = self
            .entries
            .iter()
            .filter(|(_, _, tombstoned)| !tombstoned)
            .map(|(rid, emb, _)| (rid.clone(), cosine_distance(query, emb)))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.truncate(k);
        scored
    }

    /// Embedding dimension this index was constructed with.
    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.entries.iter().filter(|(_, _, t)| !t).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_construction_is_deterministic() {
        // v0.10 Phase 0 determinism seam: same seed + same insertion order
        // => identical graphs => identical approximate result sets.
        let build = |seed: u64| {
            let mut idx = HnswIndex::with_params_seeded(16, 16, 200, 200, seed);
            for i in 0..200 {
                idx.insert(&format!("rid-{i}"), &vec_seed(i as f32, 16))
                    .unwrap();
            }
            idx.search(&vec_seed(7.0, 16), 10).unwrap()
        };
        let a = build(42);
        let b = build(42);
        assert_eq!(a, b, "same seed must reproduce identical results");
    }

    #[test]
    fn default_construction_is_deterministic_across_opens() {
        // Fix (j) regression gate — the eleventh determinism source. Two
        // independently constructed DEFAULT indices over the same
        // insertion sequence must agree on the ENTIRE result pool, tail
        // included: the convicted defect lived at pool positions 51–99,
        // where per-open entropy levels swapped 3–4 approximate
        // neighbors and the k=50 comparison that once cleared this
        // suspect could not see it.
        let build = || {
            let mut idx = HnswIndex::new(16);
            for i in 0..200 {
                idx.insert(&format!("rid-{i:03}"), &vec_seed(i as f32, 16))
                    .unwrap();
            }
            idx
        };
        let (a, b) = (build(), build());
        for q in 0..8 {
            let qa = a.search(&vec_seed(q as f32 * 3.7, 16), 100).unwrap();
            let qb = b.search(&vec_seed(q as f32 * 3.7, 16), 100).unwrap();
            assert_eq!(
                qa, qb,
                "query {q}: fresh default constructions disagree — the \
                 level RNG is drawing per-instance entropy again"
            );
        }
    }

    #[test]
    fn every_insert_is_reachable_by_its_own_vector() {
        // The mount-drop bug: pure-distance pruning in a dense cluster
        // removed every incoming layer-0 edge of some node, making it
        // present-but-unfindable — a 65-record pack lost a different
        // record per mount (RNG-dependent), and a 377-record pack lost 2.
        // The invariant: searching a stored vector itself must return its
        // rid. Fix (j) pinned the production level RNG, so varied draws
        // now come from explicit per-round seeds — the coverage this test
        // wants (no orphan-producing construction across level layouts)
        // is preserved, not inherited from entropy.
        for round in 0..5u64 {
            let mut idx = HnswIndex::with_params_seeded(16, 16, 200, 200, round * 7919 + 1);
            // A dense cluster (tiny perturbations of one direction) plus
            // scattered outliers — the shape that saturates max_m and
            // makes distance-only pruning drop backlinks.
            for i in 0..120 {
                let mut v = vec_seed(1.0, 16);
                v[i % 16] += 0.001 * ((i as f32) + 1.0);
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let v: Vec<f32> = v.iter().map(|x| x / norm).collect();
                idx.insert(&format!("dense-{round}-{i}"), &v).unwrap();
            }
            for i in 0..30 {
                idx.insert(&format!("far-{round}-{i}"), &vec_seed(100.0 + i as f32, 16))
                    .unwrap();
            }
            // The repair pass is part of every bulk build; a second call
            // must find nothing left to rescue.
            idx.ensure_all_reachable();
            assert_eq!(
                idx.ensure_all_reachable(),
                0,
                "round {round}: repair must be idempotent"
            );
            // Every rid must come back for a search of its own vector.
            let rids: Vec<String> = idx.idx_to_rid.clone();
            for (i, rid) in rids.iter().enumerate() {
                let emb = idx.nodes[i].embedding.clone();
                let hits = idx.search(&emb, 5).unwrap();
                assert!(
                    hits.iter().any(|(r, _)| r == rid),
                    "round {round}: {rid} is stored but unreachable by its \
                     own vector — orphaned by pruning"
                );
            }
        }
    }

    #[test]
    fn cosine_distance_guards_nan_and_zero_norms() {
        // Issue #60: the zero-norm guard used `norm == 0.0`, which misses NaN
        // (`NaN == 0.0` is false) and lets a NaN distance escape (`clamp`
        // preserves NaN). A NaN norm arises from a NaN-valued embedding
        // component. The distance must always come back finite so it can never
        // poison a caller's sort comparator.
        let nan_vec = vec![f32::NAN, 1.0, 0.0];
        let finite = vec![1.0f32, 0.0, 0.0];
        let zero = vec![0.0f32, 0.0, 0.0];

        let d_nan = cosine_distance(&nan_vec, &finite);
        assert!(
            d_nan.is_finite(),
            "NaN embedding must not yield NaN distance"
        );
        assert_eq!(d_nan, 1.0);

        // Symmetric: NaN on the other side.
        assert_eq!(cosine_distance(&finite, &nan_vec), 1.0);
        // Zero norm still guarded (unchanged behavior).
        assert_eq!(cosine_distance(&zero, &finite), 1.0);
        // Sanity: a normal pair is finite and within the clamped range.
        let d = cosine_distance(&finite, &finite);
        assert!(d.is_finite() && (0.0..=2.0).contains(&d));
    }

    /// Generate a deterministic unit-norm embedding.
    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim)
            .map(|i| ((seed + i as f32) * 0.7123 + (i as f32) * 0.3171).sin())
            .collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return vec![1.0 / (dim as f32).sqrt(); dim];
        }
        raw.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_cosine_distance_identical() {
        let v = vec_seed(1.0, 8);
        let d = cosine_distance(&v, &v);
        assert!(d.abs() < 1e-6, "distance to self should be ~0, got {d}");
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0];
        let d = cosine_distance(&a, &b);
        assert!(
            (d - 1.0).abs() < 1e-6,
            "orthogonal distance should be ~1, got {d}"
        );
    }

    #[test]
    fn test_empty_index() {
        let index = HnswIndex::new(8);
        let results = index.search(&vec_seed(1.0, 8), 10).unwrap();
        assert!(results.is_empty());
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_single_insert_search() {
        let mut index = HnswIndex::new(8);
        index.insert("a", &vec_seed(1.0, 8)).unwrap();
        assert_eq!(index.len(), 1);

        let results = index.search(&vec_seed(1.0, 8), 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "a");
        assert!(results[0].1 < 1e-6); // Distance to self should be ~0
    }

    #[test]
    fn test_insert_search_nearest() {
        let dim = 64;
        let mut index = HnswIndex::new(dim);

        // Insert 100 vectors
        for i in 0..100 {
            index
                .insert(&format!("v{i}"), &vec_seed(i as f32 * 0.37, dim))
                .unwrap();
        }
        assert_eq!(index.len(), 100);

        // Search for the nearest to vec_seed(0.0, dim) — should be "v0"
        let query = vec_seed(0.0, dim);
        let results = index.search(&query, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "v0");
    }

    #[test]
    fn test_tombstone_excludes_from_search() {
        let dim = 8;
        let mut index = HnswIndex::new(dim);

        index.insert("a", &vec_seed(1.0, dim)).unwrap();
        index.insert("b", &vec_seed(2.0, dim)).unwrap();
        assert_eq!(index.len(), 2);

        // Remove "a"
        assert!(index.remove("a"));
        assert_eq!(index.len(), 1);

        // Search should not return "a"
        let results = index.search(&vec_seed(1.0, dim), 10).unwrap();
        assert!(!results.iter().any(|(rid, _)| rid == "a"));
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut index = HnswIndex::new(8);
        assert!(!index.remove("nonexistent"));
    }

    #[test]
    fn test_free_list_reuse() {
        let dim = 8;
        let mut index = HnswIndex::new(dim);

        index.insert("a", &vec_seed(1.0, dim)).unwrap();
        let initial_nodes = index.nodes.len();

        index.remove("a");
        index.insert("b", &vec_seed(2.0, dim)).unwrap();

        // Should reuse the freed slot
        assert_eq!(index.nodes.len(), initial_nodes);
        assert_eq!(index.len(), 1);

        let results = index.search(&vec_seed(2.0, dim), 10).unwrap();
        assert_eq!(results[0].0, "b");
    }

    #[test]
    fn test_clear() {
        let dim = 8;
        let mut index = HnswIndex::new(dim);
        for i in 0..50 {
            index
                .insert(&format!("v{i}"), &vec_seed(i as f32, dim))
                .unwrap();
        }
        assert_eq!(index.len(), 50);

        index.clear();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert!(index.search(&vec_seed(1.0, dim), 10).unwrap().is_empty());
    }

    #[test]
    fn test_duplicate_insert_updates() {
        let dim = 8;
        let mut index = HnswIndex::new(dim);

        index.insert("a", &vec_seed(1.0, dim)).unwrap();
        index.insert("a", &vec_seed(2.0, dim)).unwrap();
        assert_eq!(index.len(), 1);

        // Search should find "a" near vec_seed(2.0) not vec_seed(1.0)
        let results = index.search(&vec_seed(2.0, dim), 1).unwrap();
        assert_eq!(results[0].0, "a");
        assert!(results[0].1 < 0.01);
    }

    #[test]
    fn test_resurrect_tombstoned() {
        let dim = 8;
        let mut index = HnswIndex::new(dim);

        index.insert("a", &vec_seed(1.0, dim)).unwrap();
        index.remove("a");
        assert_eq!(index.len(), 0);

        // Re-insert same rid
        index.insert("a", &vec_seed(2.0, dim)).unwrap();
        assert_eq!(index.len(), 1);

        let results = index.search(&vec_seed(2.0, dim), 1).unwrap();
        assert_eq!(results[0].0, "a");
    }

    #[test]
    fn test_recall_quality_dim64() {
        let dim = 64;
        let n = 1000;
        let k = 10;

        let mut hnsw = HnswIndex::with_params(dim, 16, 200, 50);
        let mut brute = BruteForceIndex::new(dim);

        for i in 0..n {
            let emb = vec_seed(i as f32 * 0.37, dim);
            hnsw.insert(&format!("v{i}"), &emb).unwrap();
            brute.insert(&format!("v{i}"), &emb);
        }

        // Test recall with 20 different queries
        let mut total_recall = 0.0;
        let num_queries = 20;
        for q in 0..num_queries {
            let query = vec_seed(q as f32 * 7.13 + 100.0, dim);
            let hnsw_results: HashSet<String> = hnsw
                .search(&query, k)
                .unwrap()
                .into_iter()
                .map(|(rid, _)| rid)
                .collect();
            let brute_results: HashSet<String> = brute
                .search(&query, k)
                .into_iter()
                .map(|(rid, _)| rid)
                .collect();

            let intersection = hnsw_results.intersection(&brute_results).count();
            total_recall += intersection as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall > 0.90,
            "recall@{k} should be > 0.90, got {avg_recall:.3}"
        );
    }

    #[test]
    fn test_recall_quality_dim384() {
        let dim = 384;
        let n = 500;
        let k = 10;

        let mut hnsw = HnswIndex::with_params(dim, 16, 200, 50);
        let mut brute = BruteForceIndex::new(dim);

        for i in 0..n {
            let emb = vec_seed(i as f32 * 0.37, dim);
            hnsw.insert(&format!("v{i}"), &emb).unwrap();
            brute.insert(&format!("v{i}"), &emb);
        }

        let mut total_recall = 0.0;
        let num_queries = 10;
        for q in 0..num_queries {
            let query = vec_seed(q as f32 * 7.13 + 100.0, dim);
            let hnsw_results: HashSet<String> = hnsw
                .search(&query, k)
                .unwrap()
                .into_iter()
                .map(|(rid, _)| rid)
                .collect();
            let brute_results: HashSet<String> = brute
                .search(&query, k)
                .into_iter()
                .map(|(rid, _)| rid)
                .collect();

            let intersection = hnsw_results.intersection(&brute_results).count();
            total_recall += intersection as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall > 0.85,
            "recall@{k} at dim=384 should be > 0.85, got {avg_recall:.3}"
        );
    }

    #[test]
    fn test_search_results_sorted_by_distance() {
        let dim = 64;
        let mut index = HnswIndex::new(dim);
        for i in 0..200 {
            index
                .insert(&format!("v{i}"), &vec_seed(i as f32 * 0.37, dim))
                .unwrap();
        }

        let query = vec_seed(999.0, dim);
        let results = index.search(&query, 20).unwrap();

        for i in 1..results.len() {
            assert!(
                results[i - 1].1 <= results[i].1 + 1e-10,
                "results not sorted: {} > {}",
                results[i - 1].1,
                results[i].1
            );
        }
    }

    #[test]
    fn test_large_insert_search() {
        let dim = 64;
        let n = 5000;
        let mut index = HnswIndex::new(dim);

        for i in 0..n {
            index
                .insert(&format!("v{i}"), &vec_seed(i as f32 * 0.37, dim))
                .unwrap();
        }
        assert_eq!(index.len(), n);

        let results = index.search(&vec_seed(999.0, dim), 10).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_search_with_many_tombstones() {
        let dim = 32;
        let mut index = HnswIndex::new(dim);

        // Insert 100, tombstone 90
        for i in 0..100 {
            index
                .insert(&format!("v{i}"), &vec_seed(i as f32, dim))
                .unwrap();
        }
        for i in 0..90 {
            index.remove(&format!("v{i}"));
        }
        assert_eq!(index.len(), 10);

        let results = index.search(&vec_seed(95.0, dim), 5).unwrap();
        // All results should be from v90-v99
        for (rid, _) in &results {
            let num: usize = rid[1..].parse().unwrap();
            assert!(num >= 90, "got tombstoned result {rid}");
        }
    }

    #[test]
    fn test_brute_force_index() {
        let dim = 8;
        let mut bf = BruteForceIndex::new(dim);
        bf.insert("a", &vec_seed(1.0, dim));
        bf.insert("b", &vec_seed(2.0, dim));
        bf.insert("c", &vec_seed(3.0, dim));

        assert_eq!(bf.len(), 3);
        bf.remove("b");
        assert_eq!(bf.len(), 2);

        let results = bf.search(&vec_seed(1.0, dim), 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "a"); // Closest to seed 1.0
    }
}
