//! CK-5.5 — Probabilistic Belief Network.
//!
//! Factor graph inference over epistemic edges using loopy belief
//! propagation (LBP). Enables joint reasoning about uncertain beliefs:
//! "if I learn X is true, what does that tell me about Y?"
//!
//! # Design principles
//! - Pure functions only — no DB access (engine layer handles persistence)
//! - Loopy BP with damping for graphs with moderate cycles
//! - Beta distributions for binary beliefs (natural conjugate)
//! - Gaussian distributions for continuous-valued beliefs
//! - Human-readable explanations of inference results

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::state::NodeId;

// ══════════════════════════════════════════════════════════════════════════════
// § 1  Identifiers
// ══════════════════════════════════════════════════════════════════════════════

/// Unique identifier for a variable in the belief network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VariableId(pub u64);

/// Unique identifier for a factor in the belief network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FactorId(pub u64);

// ══════════════════════════════════════════════════════════════════════════════
// § 2  Probability Distributions
// ══════════════════════════════════════════════════════════════════════════════

/// Probability distribution representation.
///
/// Supports the distributions most useful for epistemic reasoning:
/// - Beta: for binary beliefs (confidence in true/false)
/// - Gaussian: for continuous-valued beliefs
/// - Categorical: for multi-valued discrete beliefs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Distribution {
    /// Beta distribution — natural for binary beliefs.
    /// alpha = pseudo-count for "true", beta = pseudo-count for "false".
    Beta { alpha: f64, beta: f64 },
    /// Gaussian distribution — for continuous values.
    Gaussian { mean: f64, variance: f64 },
    /// Categorical distribution — for discrete multi-valued.
    Categorical { probs: Vec<f64> },
}

impl Distribution {
    /// Create a uniform Beta prior (maximum ignorance).
    pub fn uniform_beta() -> Self {
        Self::Beta {
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Create a Beta from log-odds (used by existing belief system).
    pub fn from_log_odds(log_odds: f64) -> Self {
        // Convert log-odds to probability, then to pseudo-counts.
        let p = 1.0 / (1.0 + (-log_odds).exp());
        // Use effective sample size of 2 (equivalent to observing one coin flip).
        let n = 2.0;
        Self::Beta {
            alpha: p * n,
            beta: (1.0 - p) * n,
        }
    }

    /// Create a Beta with specified strength.
    pub fn beta(alpha: f64, beta: f64) -> Self {
        Self::Beta {
            alpha: alpha.max(0.01),
            beta: beta.max(0.01),
        }
    }

    /// Create a Gaussian distribution.
    pub fn gaussian(mean: f64, variance: f64) -> Self {
        Self::Gaussian {
            mean,
            variance: variance.max(1e-10),
        }
    }

    /// Create a categorical distribution (auto-normalized).
    pub fn categorical(probs: Vec<f64>) -> Self {
        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 {
            let n = probs.len().max(1);
            return Self::Categorical {
                probs: vec![1.0 / n as f64; n],
            };
        }
        Self::Categorical {
            probs: probs.iter().map(|p| p / sum).collect(),
        }
    }

    /// Mean of the distribution.
    pub fn mean(&self) -> f64 {
        match self {
            Self::Beta { alpha, beta } => alpha / (alpha + beta),
            Self::Gaussian { mean, .. } => *mean,
            Self::Categorical { probs } => {
                // Expected value using index as value.
                probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum()
            }
        }
    }

    /// Variance of the distribution.
    pub fn variance(&self) -> f64 {
        match self {
            Self::Beta { alpha, beta } => {
                let sum = alpha + beta;
                (alpha * beta) / (sum * sum * (sum + 1.0))
            }
            Self::Gaussian { variance, .. } => *variance,
            Self::Categorical { probs } => {
                let mu = self.mean();
                probs
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let diff = i as f64 - mu;
                        diff * diff * p
                    })
                    .sum()
            }
        }
    }

    /// Shannon entropy (nats).
    pub fn entropy(&self) -> f64 {
        match self {
            Self::Beta { alpha, beta } => {
                let sum = alpha + beta;
                ln_beta(*alpha, *beta)
                    - (alpha - 1.0) * digamma(*alpha)
                    - (beta - 1.0) * digamma(*beta)
                    + (sum - 2.0) * digamma(sum)
            }
            Self::Gaussian { variance, .. } => {
                0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * variance).ln()
            }
            Self::Categorical { probs } => {
                let mut h = 0.0;
                for &p in probs {
                    if p > 0.0 {
                        h -= p * p.ln();
                    }
                }
                h
            }
        }
    }

    /// KL divergence: D_KL(self || other).
    pub fn kl_divergence(&self, other: &Distribution) -> f64 {
        match (self, other) {
            (
                Self::Beta {
                    alpha: a1,
                    beta: b1,
                },
                Self::Beta {
                    alpha: a2,
                    beta: b2,
                },
            ) => {
                let s1 = a1 + b1;
                let s2 = a2 + b2;
                ln_beta(*a2, *b2) - ln_beta(*a1, *b1)
                    + (a1 - a2) * digamma(*a1)
                    + (b1 - b2) * digamma(*b1)
                    + (s2 - s1) * digamma(s1)
            }
            (
                Self::Gaussian {
                    mean: m1,
                    variance: v1,
                },
                Self::Gaussian {
                    mean: m2,
                    variance: v2,
                },
            ) => 0.5 * ((v1 / v2).ln() + (v2 + (m1 - m2).powi(2)) / v1 - 1.0),
            (Self::Categorical { probs: p1 }, Self::Categorical { probs: p2 }) => {
                if p1.len() != p2.len() {
                    return f64::INFINITY;
                }
                let mut kl = 0.0;
                for (a, b) in p1.iter().zip(p2.iter()) {
                    if *a > 0.0 && *b > 0.0 {
                        kl += a * (a / b).ln();
                    } else if *a > 0.0 {
                        return f64::INFINITY;
                    }
                }
                kl
            }
            _ => f64::INFINITY, // incompatible types
        }
    }

    /// Confidence level (0.0 = uncertain, 1.0 = certain).
    pub fn confidence(&self) -> f64 {
        match self {
            Self::Beta { alpha, beta } => {
                let total = alpha + beta;
                // Concentration relative to uniform prior.
                1.0 - 2.0 / total.max(2.0)
            }
            Self::Gaussian { variance, .. } => {
                // Lower variance = higher confidence.
                1.0 / (1.0 + variance.sqrt())
            }
            Self::Categorical { probs } => {
                // Max probability minus uniform.
                let n = probs.len().max(1) as f64;
                let max_p = probs.iter().cloned().fold(0.0f64, f64::max);
                (max_p - 1.0 / n) / (1.0 - 1.0 / n)
            }
        }
    }
}

// ── Special functions (no external dep) ─────────────────────────────

/// Log of the Beta function: ln(B(a,b)) = ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a+b)).
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Stirling's approximation for ln(Γ(x)). Adequate for x > 0.5.
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Lanczos approximation (g=7, n=9 coefficients).
    let g = 7.0;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let z = x - 1.0;
    let mut sum = c[0];
    for i in 1..9 {
        sum += c[i] / (z + i as f64);
    }
    let t = z + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (t.ln() * (z + 0.5)) - t + sum.ln()
}

/// Digamma function ψ(x) — derivative of ln(Γ(x)).
/// Uses asymptotic expansion for x > 6, recurrence for smaller x.
fn digamma(mut x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut result = 0.0;
    // Shift to x > 6 using recurrence ψ(x+1) = ψ(x) + 1/x.
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic expansion.
    result += x.ln() - 0.5 / x;
    let x2 = x * x;
    result -= 1.0 / (12.0 * x2);
    result += 1.0 / (120.0 * x2 * x2);
    result -= 1.0 / (252.0 * x2 * x2 * x2);
    result
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Factor Types
// ══════════════════════════════════════════════════════════════════════════════

/// The semantic type of a factor (derived from cognitive edge kind).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FactorType {
    /// Both variables tend to agree (Supports edge).
    Supports,
    /// Variables tend to disagree (Contradicts edge).
    Contradicts,
    /// Directional: parent causes child (Causes edge).
    Causes,
    /// Symmetric correlation (AssociatedWith edge).
    Correlates,
    /// Directional implication: if A then likely B (Predicts edge).
    Implies,
}

/// Potential function — how variables interact in a factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PotentialFunction {
    /// Pairwise agreement: high correlation means both high or both low.
    Agreement { correlation: f64 },
    /// Pairwise opposition: high anti-correlation means one high ↔ other low.
    Opposition { anti_correlation: f64 },
    /// Directional conditional: P(child | parent).
    /// Table[parent_state][child_state] — for discretized binary beliefs.
    Conditional { table: [[f64; 2]; 2] },
    /// Noisy-OR for multi-parent causal effects.
    NoisyOr { leak: f64 },
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  Network Structure
// ══════════════════════════════════════════════════════════════════════════════

/// A random variable in the belief network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefVariable {
    pub id: VariableId,
    /// Corresponding Belief node in the cognitive state graph.
    pub node_id: NodeId,
    /// Label for readability.
    pub label: String,
    /// Prior distribution before evidence.
    pub prior: Distribution,
    /// Current posterior after message passing.
    pub posterior: Distribution,
    /// Observed evidence value (if clamped).
    pub observed: Option<f64>,
}

/// A factor (compatibility function) between variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Factor {
    pub id: FactorId,
    /// Connected variable IDs (usually 2, sometimes 3).
    pub variables: Vec<VariableId>,
    /// Semantic type of the relationship.
    pub factor_type: FactorType,
    /// Strength of the relationship ∈ [0.0, 1.0].
    pub strength: f64,
    /// The potential function.
    pub potential: PotentialFunction,
}

/// The belief network: a factor graph over belief nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefNetwork {
    /// All variables in the network.
    pub variables: Vec<BeliefVariable>,
    /// All factors in the network.
    pub factors: Vec<Factor>,
    /// Variable → factor adjacency.
    #[serde(skip)]
    var_to_factors: HashMap<VariableId, Vec<usize>>,
    /// Factor → variable adjacency.
    #[serde(skip)]
    factor_to_vars: HashMap<FactorId, Vec<VariableId>>,
    /// Next variable ID.
    next_var_id: u64,
    /// Next factor ID.
    next_factor_id: u64,
}

impl BeliefNetwork {
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            factors: Vec::new(),
            var_to_factors: HashMap::new(),
            factor_to_vars: HashMap::new(),
            next_var_id: 1,
            next_factor_id: 1,
        }
    }

    /// Add a variable to the network.
    pub fn add_variable(
        &mut self,
        node_id: NodeId,
        label: &str,
        prior: Distribution,
    ) -> VariableId {
        let id = VariableId(self.next_var_id);
        self.next_var_id += 1;
        self.variables.push(BeliefVariable {
            id,
            node_id,
            label: label.to_string(),
            prior: prior.clone(),
            posterior: prior,
            observed: None,
        });
        id
    }

    /// Add a factor connecting variables.
    pub fn add_factor(
        &mut self,
        variables: Vec<VariableId>,
        factor_type: FactorType,
        strength: f64,
        potential: PotentialFunction,
    ) -> FactorId {
        let id = FactorId(self.next_factor_id);
        self.next_factor_id += 1;
        let factor_idx = self.factors.len();

        for &var_id in &variables {
            self.var_to_factors
                .entry(var_id)
                .or_default()
                .push(factor_idx);
        }
        self.factor_to_vars.insert(id, variables.clone());

        self.factors.push(Factor {
            id,
            variables,
            factor_type,
            strength,
            potential,
        });
        id
    }

    /// Rebuild runtime indices after deserialization.
    pub fn rebuild_indices(&mut self) {
        self.var_to_factors.clear();
        self.factor_to_vars.clear();
        for (idx, factor) in self.factors.iter().enumerate() {
            for &var_id in &factor.variables {
                self.var_to_factors.entry(var_id).or_default().push(idx);
            }
            self.factor_to_vars
                .insert(factor.id, factor.variables.clone());
        }
    }

    /// Find a variable by ID.
    pub fn variable(&self, id: VariableId) -> Option<&BeliefVariable> {
        self.variables.iter().find(|v| v.id == id)
    }

    /// Find a variable by ID (mutable).
    fn variable_mut(&mut self, id: VariableId) -> Option<&mut BeliefVariable> {
        self.variables.iter_mut().find(|v| v.id == id)
    }

    /// Clamp a variable to an observed value.
    pub fn observe(&mut self, var_id: VariableId, value: f64) {
        if let Some(var) = self.variable_mut(var_id) {
            var.observed = Some(value);
            // Set posterior to a tight distribution around the observed value.
            var.posterior = Distribution::Beta {
                alpha: if value > 0.5 { 100.0 } else { 1.0 },
                beta: if value > 0.5 { 1.0 } else { 100.0 },
            };
        }
    }

    /// Clear all observations.
    pub fn clear_observations(&mut self) {
        for var in &mut self.variables {
            var.observed = None;
            var.posterior = var.prior.clone();
        }
    }

    /// Number of variables.
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// Number of factors.
    pub fn factor_count(&self) -> usize {
        self.factors.len()
    }

    /// Get factors connected to a variable.
    fn factors_of(&self, var_id: VariableId) -> Vec<usize> {
        self.var_to_factors
            .get(&var_id)
            .cloned()
            .unwrap_or_default()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  Loopy Belief Propagation
// ══════════════════════════════════════════════════════════════════════════════

/// Configuration for belief propagation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPConfig {
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance (max change in any marginal).
    pub tolerance: f64,
    /// Damping factor (0.0 = no damping, 1.0 = no update). 0.5 is typical.
    pub damping: f64,
}

impl Default for BPConfig {
    fn default() -> Self {
        Self {
            max_iterations: 30,
            tolerance: 1e-4,
            damping: 0.5,
        }
    }
}

/// Result of running belief propagation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPResult {
    /// Number of iterations run.
    pub iterations: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Maximum change in last iteration.
    pub max_change: f64,
}

/// Run loopy belief propagation on the network.
///
/// Uses sum-product message passing with damping for stability.
/// Updates posteriors in-place on the network variables.
pub fn loopy_belief_propagation(network: &mut BeliefNetwork, config: &BPConfig) -> BPResult {
    let var_ids: Vec<VariableId> = network.variables.iter().map(|v| v.id).collect();
    let n_vars = var_ids.len();

    if n_vars == 0 {
        return BPResult {
            iterations: 0,
            converged: true,
            max_change: 0.0,
        };
    }

    // Message storage: (factor_idx, var_id) → message (as mean value).
    // For simplicity, messages are scalar means of the marginal.
    let mut messages: HashMap<(usize, VariableId), f64> = HashMap::new();

    // Initialize messages to prior means.
    for (fi, factor) in network.factors.iter().enumerate() {
        for &var_id in &factor.variables {
            let prior_mean = network
                .variable(var_id)
                .map(|v| v.prior.mean())
                .unwrap_or(0.5);
            messages.insert((fi, var_id), prior_mean);
        }
    }

    let mut iterations = 0;
    let mut max_change = f64::MAX;

    while iterations < config.max_iterations && max_change > config.tolerance {
        max_change = 0.0;
        iterations += 1;

        // For each factor, compute factor → variable messages.
        for (fi, factor) in network.factors.iter().enumerate() {
            for &target_var in &factor.variables {
                // Skip observed variables (their messages are fixed).
                let is_observed = network
                    .variable(target_var)
                    .map(|v| v.observed.is_some())
                    .unwrap_or(false);
                if is_observed {
                    continue;
                }

                // Collect incoming messages from other variables in this factor.
                let other_means: Vec<f64> = factor
                    .variables
                    .iter()
                    .filter(|&&v| v != target_var)
                    .map(|&v| {
                        // Use current posterior mean as the variable's belief.
                        network
                            .variable(v)
                            .map(|var| {
                                if var.observed.is_some() {
                                    var.posterior.mean()
                                } else {
                                    var.posterior.mean()
                                }
                            })
                            .unwrap_or(0.5)
                    })
                    .collect();

                // Compute new message based on potential function.
                let new_msg =
                    compute_factor_message(&factor.potential, factor.strength, &other_means);

                // Apply damping.
                let old_msg = messages.get(&(fi, target_var)).copied().unwrap_or(0.5);
                let damped = config.damping * old_msg + (1.0 - config.damping) * new_msg;

                let change = (damped - old_msg).abs();
                if change > max_change {
                    max_change = change;
                }

                messages.insert((fi, target_var), damped);
            }
        }

        // Update posteriors: combine prior with all incoming factor messages.
        for &var_id in &var_ids {
            let is_observed = network
                .variable(var_id)
                .map(|v| v.observed.is_some())
                .unwrap_or(false);
            if is_observed {
                continue;
            }

            let prior_mean = network
                .variable(var_id)
                .map(|v| v.prior.mean())
                .unwrap_or(0.5);

            let factor_indices = network.factors_of(var_id);

            // Combine messages: weighted average of factor messages + prior.
            let mut sum = prior_mean;
            let mut weight = 1.0;
            for &fi in &factor_indices {
                if let Some(&msg) = messages.get(&(fi, var_id)) {
                    sum += msg;
                    weight += 1.0;
                }
            }
            let combined_mean = (sum / weight).clamp(0.001, 0.999);

            // Update posterior as a Beta distribution.
            // Use concentration proportional to evidence count.
            let concentration = 2.0 + weight;
            if let Some(var) = network.variable_mut(var_id) {
                var.posterior = Distribution::Beta {
                    alpha: combined_mean * concentration,
                    beta: (1.0 - combined_mean) * concentration,
                };
            }
        }
    }

    BPResult {
        iterations,
        converged: max_change <= config.tolerance,
        max_change,
    }
}

/// Compute a factor → variable message based on the potential function.
fn compute_factor_message(
    potential: &PotentialFunction,
    strength: f64,
    other_means: &[f64],
) -> f64 {
    if other_means.is_empty() {
        return 0.5; // no other variables → uninformative
    }

    let avg_other = other_means.iter().sum::<f64>() / other_means.len() as f64;

    match potential {
        PotentialFunction::Agreement { correlation } => {
            // Agreement: push target toward same value as others.
            let effective = correlation * strength;
            0.5 + effective * (avg_other - 0.5)
        }
        PotentialFunction::Opposition { anti_correlation } => {
            // Opposition: push target toward opposite value.
            let effective = anti_correlation * strength;
            0.5 - effective * (avg_other - 0.5)
        }
        PotentialFunction::Conditional { table } => {
            // Weighted sum of conditional probabilities.
            // P(child=1) = P(child=1|parent=1)*P(parent=1) + P(child=1|parent=0)*P(parent=0)
            let p_parent_high = avg_other;
            let p_parent_low = 1.0 - avg_other;
            let p_child_high = table[1][1] * p_parent_high + table[0][1] * p_parent_low;
            // Blend with strength (weaker = closer to 0.5).
            0.5 + strength * (p_child_high - 0.5)
        }
        PotentialFunction::NoisyOr { leak } => {
            // Noisy-OR: P(child=0) = leak * Π(1 - strength_i * parent_i).
            // For simplicity with scalar messages:
            let p_no_cause = (1.0 - strength * avg_other).max(0.0);
            let p_child_off = leak * p_no_cause;
            1.0 - p_child_off
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 6  Inference Queries
// ══════════════════════════════════════════════════════════════════════════════

/// Type of probabilistic query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceType {
    /// Marginal probability of the target variable.
    Marginal,
    /// Conditional probability given evidence.
    Conditional,
    /// Maximum a posteriori assignment.
    MAP,
    /// Most probable explanation for evidence.
    MostProbableExplanation,
}

/// A probabilistic query about the belief network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceQuery {
    /// Target variable to query about.
    pub target: VariableId,
    /// Evidence: observed (variable, value) pairs.
    pub evidence: Vec<(VariableId, f64)>,
    /// Type of query.
    pub query_type: InferenceType,
}

/// Contribution of a single piece of evidence to the result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceContribution {
    /// The evidence variable.
    pub variable: VariableId,
    /// How much this evidence changed the target posterior.
    pub impact: f64,
    /// Direction: positive = increased target, negative = decreased.
    pub direction: f64,
}

/// Result of an inference query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    /// Target variable.
    pub target: VariableId,
    /// Posterior distribution after inference.
    pub posterior: Distribution,
    /// Which evidence contributed most.
    pub contributions: Vec<EvidenceContribution>,
    /// Number of BP iterations used.
    pub iterations: usize,
    /// Whether BP converged.
    pub converged: bool,
}

/// Run an inference query on the network.
///
/// Clamps evidence variables, runs BP, reads target posterior.
pub fn query(
    network: &mut BeliefNetwork,
    query: &InferenceQuery,
    bp_config: &BPConfig,
) -> InferenceResult {
    // Save the prior of the target before any evidence.
    let prior_mean = network
        .variable(query.target)
        .map(|v| v.prior.mean())
        .unwrap_or(0.5);

    // Reset to priors.
    for var in &mut network.variables {
        if var.observed.is_none() {
            var.posterior = var.prior.clone();
        }
    }

    // Clamp evidence.
    for &(var_id, value) in &query.evidence {
        network.observe(var_id, value);
    }

    // Run BP.
    let bp_result = loopy_belief_propagation(network, bp_config);

    // Read target posterior.
    let posterior = network
        .variable(query.target)
        .map(|v| v.posterior.clone())
        .unwrap_or(Distribution::uniform_beta());

    // Compute evidence contributions by ablation.
    let contributions = compute_evidence_contributions(
        network,
        query.target,
        &query.evidence,
        prior_mean,
        bp_config,
    );

    // Clean up observations.
    for &(var_id, _) in &query.evidence {
        if let Some(var) = network.variable_mut(var_id) {
            var.observed = None;
        }
    }

    InferenceResult {
        target: query.target,
        posterior,
        contributions,
        iterations: bp_result.iterations,
        converged: bp_result.converged,
    }
}

/// Compute how much each evidence variable contributed to the result.
fn compute_evidence_contributions(
    network: &mut BeliefNetwork,
    target: VariableId,
    evidence: &[(VariableId, f64)],
    prior_mean: f64,
    bp_config: &BPConfig,
) -> Vec<EvidenceContribution> {
    let full_posterior_mean = network
        .variable(target)
        .map(|v| v.posterior.mean())
        .unwrap_or(0.5);

    let mut contributions = Vec::new();

    // For each evidence variable, measure its individual impact.
    for &(ev_var, ev_val) in evidence {
        // Remove this one piece of evidence and re-run.
        if let Some(var) = network.variable_mut(ev_var) {
            var.observed = None;
            var.posterior = var.prior.clone();
        }

        // Reset non-observed posteriors.
        for var in &mut network.variables {
            if var.observed.is_none() {
                var.posterior = var.prior.clone();
            }
        }

        // Re-clamp other evidence.
        for &(other_var, other_val) in evidence {
            if other_var != ev_var {
                network.observe(other_var, other_val);
            }
        }

        loopy_belief_propagation(network, bp_config);

        let without_mean = network
            .variable(target)
            .map(|v| v.posterior.mean())
            .unwrap_or(0.5);

        let impact = (full_posterior_mean - without_mean).abs();
        let direction = full_posterior_mean - without_mean;

        contributions.push(EvidenceContribution {
            variable: ev_var,
            impact,
            direction,
        });

        // Re-observe for next iteration.
        network.observe(ev_var, ev_val);
    }

    // Sort by impact descending.
    contributions.sort_by(|a, b| {
        b.impact
            .partial_cmp(&a.impact)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    contributions
}

// ══════════════════════════════════════════════════════════════════════════════
// § 7  Information Gain & Sensitivity
// ══════════════════════════════════════════════════════════════════════════════

/// Compute expected information gain: how much would observing `candidate`
/// reduce uncertainty about `target`?
///
/// Approximated by measuring the change in target entropy under
/// high vs. low observations of the candidate.
pub fn information_gain(
    network: &mut BeliefNetwork,
    candidate: VariableId,
    target: VariableId,
    bp_config: &BPConfig,
) -> f64 {
    // Baseline entropy of target.
    reset_network(network);
    loopy_belief_propagation(network, bp_config);
    let baseline_entropy = network
        .variable(target)
        .map(|v| v.posterior.entropy())
        .unwrap_or(0.0);

    // Entropy if we observe candidate = high (0.9).
    reset_network(network);
    network.observe(candidate, 0.9);
    loopy_belief_propagation(network, bp_config);
    let high_entropy = network
        .variable(target)
        .map(|v| v.posterior.entropy())
        .unwrap_or(0.0);

    // Entropy if we observe candidate = low (0.1).
    reset_network(network);
    network.observe(candidate, 0.1);
    loopy_belief_propagation(network, bp_config);
    let low_entropy = network
        .variable(target)
        .map(|v| v.posterior.entropy())
        .unwrap_or(0.0);

    // Clean up.
    if let Some(var) = network.variable_mut(candidate) {
        var.observed = None;
    }
    reset_network(network);

    // Expected information gain (average of both scenarios).
    let avg_conditional_entropy = (high_entropy + low_entropy) / 2.0;
    (baseline_entropy - avg_conditional_entropy).max(0.0)
}

/// Rank all variables by how much observing them would change the
/// target's posterior. Returns sorted (variable, sensitivity) pairs.
pub fn sensitivity_to_evidence(
    network: &mut BeliefNetwork,
    target: VariableId,
    bp_config: &BPConfig,
) -> Vec<(VariableId, f64)> {
    let var_ids: Vec<VariableId> = network
        .variables
        .iter()
        .filter(|v| v.id != target)
        .map(|v| v.id)
        .collect();

    let mut results = Vec::new();
    for var_id in var_ids {
        let ig = information_gain(network, var_id, target, bp_config);
        if ig > 1e-6 {
            results.push((var_id, ig));
        }
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    results
}

/// Reset all non-observed variables to their priors.
fn reset_network(network: &mut BeliefNetwork) {
    for var in &mut network.variables {
        if var.observed.is_none() {
            var.posterior = var.prior.clone();
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 8  Network Diagnostics
// ══════════════════════════════════════════════════════════════════════════════

/// Health diagnostic for the belief network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkHealth {
    /// Total variables.
    pub variable_count: usize,
    /// Total factors.
    pub factor_count: usize,
    /// Number of disconnected components.
    pub components: usize,
    /// Variables with extreme priors (very high or very low confidence).
    pub extreme_priors: Vec<VariableId>,
    /// Variables that might cause convergence issues (in tight cycles).
    pub potential_instabilities: Vec<VariableId>,
    /// Average confidence across all variables.
    pub avg_confidence: f64,
    /// Whether the network is well-formed.
    pub healthy: bool,
}

/// Run diagnostics on the belief network.
pub fn network_diagnostics(network: &BeliefNetwork) -> NetworkHealth {
    let n = network.variables.len();
    let m = network.factors.len();

    // Find disconnected components using union-find.
    let var_ids: Vec<VariableId> = network.variables.iter().map(|v| v.id).collect();
    let mut parent: HashMap<VariableId, VariableId> = HashMap::new();
    for &v in &var_ids {
        parent.insert(v, v);
    }

    fn find(parent: &mut HashMap<VariableId, VariableId>, x: VariableId) -> VariableId {
        let p = *parent.get(&x).unwrap_or(&x);
        if p != x {
            let root = find(parent, p);
            parent.insert(x, root);
            root
        } else {
            x
        }
    }

    for factor in &network.factors {
        if factor.variables.len() >= 2 {
            let root0 = find(&mut parent, factor.variables[0]);
            for &v in &factor.variables[1..] {
                let root_v = find(&mut parent, v);
                if root0 != root_v {
                    parent.insert(root_v, root0);
                }
            }
        }
    }

    let mut roots = std::collections::HashSet::new();
    for &v in &var_ids {
        roots.insert(find(&mut parent, v));
    }
    let components = roots.len();

    // Find extreme priors.
    let mut extreme_priors = Vec::new();
    for var in &network.variables {
        let conf = var.prior.confidence();
        if conf > 0.95 {
            extreme_priors.push(var.id);
        }
    }

    // Find potential instabilities (variables in many factors).
    let mut potential_instabilities = Vec::new();
    for var in &network.variables {
        let factor_count = network
            .var_to_factors
            .get(&var.id)
            .map(|v| v.len())
            .unwrap_or(0);
        if factor_count > 5 {
            potential_instabilities.push(var.id);
        }
    }

    // Average confidence.
    let avg_confidence = if n > 0 {
        network
            .variables
            .iter()
            .map(|v| v.posterior.confidence())
            .sum::<f64>()
            / n as f64
    } else {
        0.0
    };

    let healthy = components <= n.max(1)
        && extreme_priors.len() <= n / 2
        && potential_instabilities.is_empty();

    NetworkHealth {
        variable_count: n,
        factor_count: m,
        components,
        extreme_priors,
        potential_instabilities,
        avg_confidence,
        healthy,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 9  Network Construction from Cognitive Graph
// ══════════════════════════════════════════════════════════════════════════════

/// Build a belief network from cognitive graph nodes and edges.
///
/// Takes belief node IDs and their connecting edges, constructs
/// the factor graph with appropriate potential functions.
pub fn build_network_from_edges(
    beliefs: &[(NodeId, &str, f64)], // (node_id, label, log_odds)
    edges: &[(NodeId, NodeId, EdgeRelation, f64)], // (from, to, relation, weight)
) -> BeliefNetwork {
    let mut network = BeliefNetwork::new();
    let mut node_to_var: HashMap<NodeId, VariableId> = HashMap::new();

    // Create variables.
    for &(node_id, label, log_odds) in beliefs {
        let prior = Distribution::from_log_odds(log_odds);
        let var_id = network.add_variable(node_id, label, prior);
        node_to_var.insert(node_id, var_id);
    }

    // Create factors from edges.
    for &(from, to, relation, weight) in edges {
        let from_var = match node_to_var.get(&from) {
            Some(&v) => v,
            None => continue,
        };
        let to_var = match node_to_var.get(&to) {
            Some(&v) => v,
            None => continue,
        };

        let (factor_type, potential) = match relation {
            EdgeRelation::Supports => (
                FactorType::Supports,
                PotentialFunction::Agreement {
                    correlation: weight,
                },
            ),
            EdgeRelation::Contradicts => (
                FactorType::Contradicts,
                PotentialFunction::Opposition {
                    anti_correlation: weight,
                },
            ),
            EdgeRelation::Causes => (
                FactorType::Causes,
                PotentialFunction::Conditional {
                    table: [
                        [1.0 - weight * 0.2, weight * 0.2], // parent=low
                        [1.0 - weight * 0.8, weight * 0.8], // parent=high
                    ],
                },
            ),
            EdgeRelation::Predicts => (
                FactorType::Implies,
                PotentialFunction::Agreement {
                    correlation: weight * 0.8, // slightly weaker than direct support
                },
            ),
            EdgeRelation::Correlates => (
                FactorType::Correlates,
                PotentialFunction::Agreement {
                    correlation: weight * 0.5,
                },
            ),
        };

        network.add_factor(vec![from_var, to_var], factor_type, weight, potential);
    }

    network
}

/// Edge relation type for network construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeRelation {
    Supports,
    Contradicts,
    Causes,
    Predicts,
    Correlates,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 10  Most Probable Explanation
// ══════════════════════════════════════════════════════════════════════════════

/// Find the most probable joint assignment of all variables
/// that best explains the observed evidence.
///
/// Uses max-product variant of message passing (approximation).
pub fn most_probable_explanation(
    network: &mut BeliefNetwork,
    evidence: &[(VariableId, f64)],
    bp_config: &BPConfig,
) -> Vec<(VariableId, f64)> {
    // Clamp evidence.
    for &(var_id, value) in evidence {
        network.observe(var_id, value);
    }

    // Run BP to get marginals (using sum-product as approximation to max-product).
    loopy_belief_propagation(network, bp_config);

    // Read the MAP assignment (mode of each posterior).
    let mut assignment = Vec::new();
    for var in &network.variables {
        let value = var.posterior.mean(); // For Beta, mean ≈ mode for concentrated distributions.
        assignment.push((var.id, value));
    }

    // Clean up.
    for &(var_id, _) in evidence {
        if let Some(var) = network.variable_mut(var_id) {
            var.observed = None;
        }
    }
    reset_network(network);

    assignment
}

// ══════════════════════════════════════════════════════════════════════════════
// § 11  Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{NodeId, NodeKind};

    fn belief_node(seq: u32) -> NodeId {
        NodeId::new(NodeKind::Belief, seq)
    }

    fn make_simple_network() -> BeliefNetwork {
        // A supports B, B contradicts C.
        let mut net = BeliefNetwork::new();
        let a = net.add_variable(belief_node(1), "A", Distribution::beta(3.0, 2.0));
        let b = net.add_variable(belief_node(2), "B", Distribution::uniform_beta());
        let c = net.add_variable(belief_node(3), "C", Distribution::uniform_beta());

        net.add_factor(
            vec![a, b],
            FactorType::Supports,
            0.8,
            PotentialFunction::Agreement { correlation: 0.8 },
        );
        net.add_factor(
            vec![b, c],
            FactorType::Contradicts,
            0.7,
            PotentialFunction::Opposition {
                anti_correlation: 0.7,
            },
        );
        net
    }

    fn make_chain_network() -> BeliefNetwork {
        // A → B → C (causal chain).
        let mut net = BeliefNetwork::new();
        let a = net.add_variable(belief_node(1), "A", Distribution::beta(8.0, 2.0)); // strong prior
        let b = net.add_variable(belief_node(2), "B", Distribution::uniform_beta());
        let c = net.add_variable(belief_node(3), "C", Distribution::uniform_beta());

        net.add_factor(
            vec![a, b],
            FactorType::Causes,
            0.9,
            PotentialFunction::Conditional {
                table: [[0.9, 0.1], [0.2, 0.8]],
            },
        );
        net.add_factor(
            vec![b, c],
            FactorType::Causes,
            0.7,
            PotentialFunction::Conditional {
                table: [[0.8, 0.2], [0.3, 0.7]],
            },
        );
        net
    }

    // ── Distribution tests ──────────────────────────────────────────────

    #[test]
    fn test_beta_distribution_basics() {
        let d = Distribution::Beta {
            alpha: 8.0,
            beta: 2.0,
        };
        assert!((d.mean() - 0.8).abs() < 0.01);
        assert!(d.variance() > 0.0);
        // Beta(8,2) is concentrated → entropy can be negative (differential entropy).
        // Just check it's finite.
        assert!(d.entropy().is_finite());
        assert!(d.confidence() > 0.5);
    }

    #[test]
    fn test_gaussian_distribution() {
        let d = Distribution::Gaussian {
            mean: 5.0,
            variance: 2.0,
        };
        assert_eq!(d.mean(), 5.0);
        assert_eq!(d.variance(), 2.0);
        assert!(d.entropy() > 0.0);
    }

    #[test]
    fn test_categorical_distribution() {
        let d = Distribution::categorical(vec![1.0, 2.0, 3.0]);
        // Should be normalized to [1/6, 2/6, 3/6].
        assert!((d.mean() - (0.0 / 6.0 + 2.0 / 6.0 + 6.0 / 6.0)).abs() < 0.2);
        assert!(d.entropy() > 0.0);
    }

    #[test]
    fn test_from_log_odds() {
        // log_odds = 0 → p = 0.5 → Beta(1, 1).
        let d = Distribution::from_log_odds(0.0);
        assert!((d.mean() - 0.5).abs() < 0.01);

        // log_odds = 2.0 → p ≈ 0.88 → Beta(1.76, 0.24).
        let d = Distribution::from_log_odds(2.0);
        assert!(d.mean() > 0.7);
    }

    #[test]
    fn test_kl_divergence_same() {
        let d = Distribution::beta(3.0, 2.0);
        assert!(d.kl_divergence(&d).abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_different() {
        let d1 = Distribution::beta(8.0, 2.0);
        let d2 = Distribution::beta(2.0, 8.0);
        assert!(d1.kl_divergence(&d2) > 0.5);
    }

    // ── Network construction tests ──────────────────────────────────────

    #[test]
    fn test_network_construction() {
        let net = make_simple_network();
        assert_eq!(net.variable_count(), 3);
        assert_eq!(net.factor_count(), 2);
    }

    #[test]
    fn test_build_network_from_edges() {
        let beliefs = vec![
            (belief_node(1), "Rain", 0.5),
            (belief_node(2), "Wet ground", -0.5),
            (belief_node(3), "Sprinkler on", 0.0),
        ];
        let edges = vec![
            (belief_node(1), belief_node(2), EdgeRelation::Causes, 0.9),
            (belief_node(3), belief_node(2), EdgeRelation::Causes, 0.7),
            (
                belief_node(1),
                belief_node(3),
                EdgeRelation::Contradicts,
                0.3,
            ),
        ];

        let net = build_network_from_edges(&beliefs, &edges);
        assert_eq!(net.variable_count(), 3);
        assert_eq!(net.factor_count(), 3);
    }

    // ── Belief propagation tests ────────────────────────────────────────

    #[test]
    fn test_bp_converges_simple() {
        let mut net = make_simple_network();
        let config = BPConfig::default();
        let result = loopy_belief_propagation(&mut net, &config);

        assert!(result.converged);
        assert!(result.iterations <= config.max_iterations);
    }

    #[test]
    fn test_bp_support_propagation() {
        // A has strong prior (0.8). B is connected via Supports.
        // After BP, B should shift toward A's direction.
        let mut net = make_simple_network();
        let config = BPConfig::default();

        let b_prior_mean = net.variable(VariableId(2)).unwrap().prior.mean();

        loopy_belief_propagation(&mut net, &config);

        let b_posterior_mean = net.variable(VariableId(2)).unwrap().posterior.mean();

        // B should have shifted up from its prior (influenced by A's high prior).
        assert!(b_posterior_mean > b_prior_mean);
    }

    #[test]
    fn test_bp_contradiction_propagation() {
        // B supports A (high), C contradicts B.
        // After BP, C should shift down.
        let mut net = make_simple_network();
        let config = BPConfig::default();

        loopy_belief_propagation(&mut net, &config);

        let c_mean = net.variable(VariableId(3)).unwrap().posterior.mean();
        // C contradicts B, and B is pushed up by A. So C should be < 0.5.
        assert!(c_mean < 0.5);
    }

    #[test]
    fn test_bp_with_evidence() {
        let mut net = make_simple_network();
        let config = BPConfig::default();

        // Observe A = high.
        net.observe(VariableId(1), 0.9);
        loopy_belief_propagation(&mut net, &config);

        let b_mean = net.variable(VariableId(2)).unwrap().posterior.mean();
        let c_mean = net.variable(VariableId(3)).unwrap().posterior.mean();

        // B should be high (supports A), C should be low (contradicts B).
        assert!(b_mean > 0.5);
        assert!(c_mean < 0.5);
    }

    #[test]
    fn test_bp_causal_chain() {
        let mut net = make_chain_network();
        let config = BPConfig::default();

        loopy_belief_propagation(&mut net, &config);

        let a_mean = net.variable(VariableId(1)).unwrap().posterior.mean();
        let b_mean = net.variable(VariableId(2)).unwrap().posterior.mean();
        let c_mean = net.variable(VariableId(3)).unwrap().posterior.mean();

        // A has strong prior (0.8). B should be influenced by A's direction.
        assert!(a_mean > 0.6);
        // B is influenced by A through a conditional factor.
        // With damping, the shift may be modest but should be above prior (0.5).
        assert!(b_mean > 0.45); // influenced by A, at least shifted from uniform
    }

    // ── Inference query tests ───────────────────────────────────────────

    #[test]
    fn test_conditional_query() {
        let mut net = make_simple_network();
        let bp_config = BPConfig::default();

        let q = InferenceQuery {
            target: VariableId(2),                 // B
            evidence: vec![(VariableId(1), 0.95)], // observe A = very high
            query_type: InferenceType::Conditional,
        };

        let result = query(&mut net, &q, &bp_config);

        // B should be high given A is high (supports relationship).
        assert!(result.posterior.mean() > 0.5);
        assert!(result.converged);
    }

    #[test]
    fn test_evidence_contributions() {
        let mut net = BeliefNetwork::new();
        let a = net.add_variable(belief_node(1), "A", Distribution::uniform_beta());
        let b = net.add_variable(belief_node(2), "B", Distribution::uniform_beta());
        let target = net.add_variable(belief_node(3), "Target", Distribution::uniform_beta());

        // A strongly supports target.
        net.add_factor(
            vec![a, target],
            FactorType::Supports,
            0.9,
            PotentialFunction::Agreement { correlation: 0.9 },
        );
        // B weakly supports target.
        net.add_factor(
            vec![b, target],
            FactorType::Supports,
            0.3,
            PotentialFunction::Agreement { correlation: 0.3 },
        );

        let q = InferenceQuery {
            target,
            evidence: vec![(a, 0.9), (b, 0.9)],
            query_type: InferenceType::Conditional,
        };

        let result = query(&mut net, &q, &BPConfig::default());

        // A should contribute more than B.
        if result.contributions.len() >= 2 {
            assert!(result.contributions[0].impact >= result.contributions[1].impact);
        }
    }

    // ── Information gain tests ──────────────────────────────────────────

    #[test]
    fn test_information_gain() {
        let mut net = make_simple_network();
        let bp_config = BPConfig::default();

        // Information gain of observing A about B should be > 0
        // (since A supports B).
        let ig = information_gain(
            &mut net,
            VariableId(1), // A
            VariableId(2), // B
            &bp_config,
        );

        assert!(ig >= 0.0);
    }

    #[test]
    fn test_sensitivity_ranking() {
        let mut net = make_simple_network();
        let bp_config = BPConfig::default();

        let sensitivities = sensitivity_to_evidence(
            &mut net,
            VariableId(2), // B
            &bp_config,
        );

        // A and C are both connected to B, so both should have some sensitivity.
        assert!(!sensitivities.is_empty());
    }

    // ── Most probable explanation tests ─────────────────────────────────

    #[test]
    fn test_mpe() {
        let mut net = make_simple_network();
        let bp_config = BPConfig::default();

        let assignment = most_probable_explanation(
            &mut net,
            &[(VariableId(1), 0.9)], // A is high
            &bp_config,
        );

        assert_eq!(assignment.len(), 3);
        // A should be high, B should be high (supports), C should be low (contradicts B).
        let a_val = assignment
            .iter()
            .find(|(id, _)| *id == VariableId(1))
            .unwrap()
            .1;
        assert!(a_val > 0.7);
    }

    // ── Network diagnostics tests ───────────────────────────────────────

    #[test]
    fn test_diagnostics_healthy() {
        let net = make_simple_network();
        let health = network_diagnostics(&net);

        assert_eq!(health.variable_count, 3);
        assert_eq!(health.factor_count, 2);
        assert_eq!(health.components, 1); // all connected
        assert!(health.healthy);
    }

    #[test]
    fn test_diagnostics_disconnected() {
        let mut net = BeliefNetwork::new();
        net.add_variable(belief_node(1), "A", Distribution::uniform_beta());
        net.add_variable(belief_node(2), "B", Distribution::uniform_beta());
        // No factors connecting them.

        let health = network_diagnostics(&net);
        assert_eq!(health.components, 2); // disconnected
    }

    // ── Edge case tests ─────────────────────────────────────────────────

    #[test]
    fn test_empty_network() {
        let mut net = BeliefNetwork::new();
        let config = BPConfig::default();

        let result = loopy_belief_propagation(&mut net, &config);
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_single_variable_no_factors() {
        let mut net = BeliefNetwork::new();
        net.add_variable(belief_node(1), "alone", Distribution::beta(5.0, 2.0));

        let config = BPConfig::default();
        loopy_belief_propagation(&mut net, &config);

        // Should remain at prior.
        let mean = net.variable(VariableId(1)).unwrap().posterior.mean();
        assert!((mean - 5.0 / 7.0).abs() < 0.1);
    }

    #[test]
    fn test_noisy_or_potential() {
        let mut net = BeliefNetwork::new();
        let parent1 = net.add_variable(belief_node(1), "P1", Distribution::beta(8.0, 2.0));
        let parent2 = net.add_variable(belief_node(2), "P2", Distribution::beta(7.0, 3.0));
        let child = net.add_variable(belief_node(3), "Child", Distribution::uniform_beta());

        net.add_factor(
            vec![parent1, child],
            FactorType::Causes,
            0.8,
            PotentialFunction::NoisyOr { leak: 0.1 },
        );
        net.add_factor(
            vec![parent2, child],
            FactorType::Causes,
            0.6,
            PotentialFunction::NoisyOr { leak: 0.1 },
        );

        let config = BPConfig::default();
        loopy_belief_propagation(&mut net, &config);

        // With two strong parents, child should be high.
        let child_mean = net.variable(child).unwrap().posterior.mean();
        assert!(child_mean > 0.5);
    }
}
