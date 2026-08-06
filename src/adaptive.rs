use crate::policy::{Decision, Policy};
use rand::Rng;
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};

const MIN_OBSERVATIONS: usize = 5;
const FIT_STEPS: usize = 100;
const FIT_RATE: f32 = 0.5;
const BOOTSTRAPS: usize = 20;

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

// Fit P(correct | s) with a two-parameter logistic model.
pub fn fit(observations: &[(f32, bool)]) -> Option<(f32, f32)> {
    if observations.len() < MIN_OBSERVATIONS
        || observations.iter().all(|(_, correct)| *correct)
        || observations.iter().all(|(_, correct)| !*correct)
    {
        return None;
    }

    let mut t = observations.iter().map(|(sim, _)| *sim).sum::<f32>() / observations.len() as f32;
    let mut gamma = 1.0f32;
    let count = observations.len() as f32;

    for _ in 0..FIT_STEPS {
        let mut dgamma = 0.0f32;
        let mut dt = 0.0f32;
        for &(similarity, correct) in observations {
            let probability = sigmoid(gamma * (similarity - t));
            let label = if correct { 1.0 } else { 0.0 };
            let error = label - probability;
            dgamma += error * (similarity - t);
            dt += error * -gamma;
        }
        gamma += FIT_RATE * dgamma / count;
        t += FIT_RATE * dt / count;
    }

    if t.is_finite() && gamma.is_finite() {
        Some((t, gamma))
    } else {
        None
    }
}

// Keep a good single-label entry usable after cold start.
pub fn fit_or_confident(observations: &[(f32, bool)]) -> Option<(f32, f32)> {
    if let Some(fitted) = fit(observations) {
        return Some(fitted);
    }
    if observations.len() < MIN_OBSERVATIONS || !observations.iter().all(|(_, correct)| *correct) {
        return None;
    }
    let minimum = observations
        .iter()
        .map(|(similarity, _)| *similarity)
        .fold(f32::INFINITY, f32::min);
    if minimum.is_finite() {
        Some((minimum - 0.05, 50.0))
    } else {
        None
    }
}

fn clip_probability(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn tau(delta: f32, alpha: f32) -> f32 {
    let denominator = 1.0 - alpha;
    if denominator <= 0.0 {
        return 1.0;
    }
    clip_probability(1.0 - delta / denominator)
}

fn laplace_cap(alpha: f32, observations: usize) -> f32 {
    alpha.min((observations + 1) as f32 / (observations + 2) as f32)
}

fn draw_decision(rng: &mut ChaCha8Rng, explore_probability: f32) -> Decision {
    if rng.gen::<f32>() <= explore_probability {
        Decision::Miss
    } else {
        Decision::Hit
    }
}

fn reset_last_observation(last_correct: &mut Option<bool>) {
    *last_correct = None;
}

fn should_insert(last_correct: Option<bool>) -> bool {
    last_correct.map(|correct| !correct).unwrap_or(true)
}

#[derive(Default)]
struct EntryState {
    observations: Vec<Vec<(f32, bool)>>,
    fits: Vec<Option<(f32, f32)>>,
}

impl EntryState {
    fn fit_for(&self, entry: usize) -> Option<(f32, f32)> {
        self.fits.get(entry).copied().flatten()
    }

    fn observe(&mut self, entry: usize, similarity: f32, correct: bool) {
        if self.observations.len() <= entry {
            self.observations.resize_with(entry + 1, Vec::new);
        }
        if self.fits.len() <= entry {
            self.fits.resize(entry + 1, None);
        }
        self.observations[entry].push((similarity, correct));
        self.fits[entry] = fit_or_confident(&self.observations[entry]);
    }
}

// Use one sigmoid fit for observations from all entries.
pub struct GdPolicy {
    pub delta: f32,
    observations: Vec<(f32, bool)>,
    fitted: Option<(f32, f32)>,
    pending_refits: usize,
    rng: ChaCha8Rng,
    last_correct: Option<bool>,
}

impl GdPolicy {
    pub fn new(delta: f32) -> Self {
        Self {
            delta,
            observations: Vec::new(),
            fitted: None,
            pending_refits: 0,
            rng: ChaCha8Rng::seed_from_u64(42),
            last_correct: None,
        }
    }
}

impl Policy for GdPolicy {
    fn decide(&mut self, neighbor: Option<(usize, f32)>) -> Decision {
        reset_last_observation(&mut self.last_correct);
        let Some((_, similarity)) = neighbor else {
            return Decision::Miss;
        };
        let Some((t, gamma)) = self.fitted else {
            return Decision::Miss;
        };
        let alpha = laplace_cap(sigmoid(gamma * (similarity - t)), self.observations.len());
        draw_decision(&mut self.rng, tau(self.delta, alpha))
    }

    fn should_insert(&self) -> bool {
        should_insert(self.last_correct)
    }

    fn observe(&mut self, _entry: usize, similarity: f32, correct: bool) {
        self.last_correct = Some(correct);
        self.observations.push((similarity, correct));
        self.pending_refits += 1;
        if self.pending_refits >= 64 || self.fitted.is_none() {
            self.fitted = fit_or_confident(&self.observations);
            self.pending_refits = 0;
        }
    }
}

// Fit one sigmoid for each cached entry.
pub struct LdPolicy {
    pub delta: f32,
    state: EntryState,
    rng: ChaCha8Rng,
    last_correct: Option<bool>,
}

impl LdPolicy {
    pub fn new(delta: f32) -> Self {
        Self {
            delta,
            state: EntryState::default(),
            rng: ChaCha8Rng::seed_from_u64(42),
            last_correct: None,
        }
    }
}

impl Policy for LdPolicy {
    fn decide(&mut self, neighbor: Option<(usize, f32)>) -> Decision {
        reset_last_observation(&mut self.last_correct);
        let Some((entry, similarity)) = neighbor else {
            return Decision::Miss;
        };
        let Some((t, gamma)) = self.state.fit_for(entry) else {
            return Decision::Miss;
        };
        let alpha = laplace_cap(
            sigmoid(gamma * (similarity - t)),
            self.state.observations[entry].len(),
        );
        draw_decision(&mut self.rng, tau(self.delta, alpha))
    }

    fn should_insert(&self) -> bool {
        should_insert(self.last_correct)
    }

    fn observe(&mut self, entry: usize, similarity: f32, correct: bool) {
        self.last_correct = Some(correct);
        self.state.observe(entry, similarity, correct);
    }
}

// Fit one sigmoid for each entry and bound its threshold with bootstrap samples.
pub struct Ld3Policy {
    pub delta: f32,
    state: EntryState,
    bounds: Vec<Vec<f32>>,
    rng: ChaCha8Rng,
    last_correct: Option<bool>,
}

impl Ld3Policy {
    pub fn new(delta: f32) -> Self {
        Self {
            delta,
            state: EntryState::default(),
            bounds: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(42),
            last_correct: None,
        }
    }

    fn refit_bound(&mut self, entry: usize) {
        if self.bounds.len() <= entry {
            self.bounds.resize_with(entry + 1, Vec::new);
        }
        let observations = &self.state.observations[entry];
        let mut values = Vec::with_capacity(BOOTSTRAPS);
        for _ in 0..BOOTSTRAPS {
            let sample: Vec<_> = (0..observations.len())
                .map(|_| observations[self.rng.gen_range(0..observations.len())])
                .collect();
            if let Some((threshold, _)) = fit(&sample) {
                values.push(threshold);
            }
        }
        values.sort_by(|a, b| a.total_cmp(b));
        if values.is_empty() {
            if let Some((threshold, _)) = self.state.fit_for(entry) {
                values.push(threshold);
            }
        }
        self.bounds[entry] = values;
    }
}

impl Policy for Ld3Policy {
    fn decide(&mut self, neighbor: Option<(usize, f32)>) -> Decision {
        reset_last_observation(&mut self.last_correct);
        let Some((entry, similarity)) = neighbor else {
            return Decision::Miss;
        };
        let Some((_, gamma)) = self.state.fit_for(entry) else {
            return Decision::Miss;
        };
        let Some(bounds) = self.bounds.get(entry).filter(|values| !values.is_empty()) else {
            return Decision::Miss;
        };

        let mut tau_hat = 1.0f32;
        for step in 0..20 {
            let epsilon = 0.01 + step as f32 * 0.98 / 19.0;
            let index = ((bounds.len() - 1) as f32 * epsilon) as usize;
            let threshold = bounds[index];
            let alpha = laplace_cap(
                (1.0 - epsilon) * sigmoid(gamma * (similarity - threshold)),
                self.state.observations[entry].len(),
            );
            tau_hat = tau_hat.min(tau(self.delta, alpha));
        }
        draw_decision(&mut self.rng, tau_hat)
    }

    fn should_insert(&self) -> bool {
        should_insert(self.last_correct)
    }

    fn observe(&mut self, entry: usize, similarity: f32, correct: bool) {
        self.last_correct = Some(correct);
        self.state.observe(entry, similarity, correct);
        self.refit_bound(entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{Decision, Policy};

    fn separable_observations() -> Vec<(f32, bool)> {
        (0..20)
            .map(|index| {
                let similarity = 0.61 + index as f32 * 0.02;
                (similarity, similarity >= 0.8)
            })
            .collect()
    }

    #[test]
    fn fit_finds_separable_boundary() {
        let (threshold, gamma) = fit(&separable_observations()).unwrap();
        assert!((0.7..=0.9).contains(&threshold), "t={threshold}");
        assert!(gamma > 0.0, "gamma={gamma}");
    }

    #[test]
    fn fit_rejects_short_and_single_label_logs() {
        assert!(fit(&[(0.1, false); 4]).is_none());
        assert!(fit(&[(0.1, true); 5]).is_none());
        assert!(fit(&[(0.1, false); 5]).is_none());
    }

    #[test]
    fn confident_fit_keeps_good_entries_usable() {
        let observations = vec![(0.8, true); 6];
        let (threshold, gamma) = fit_or_confident(&observations).unwrap();
        assert!((threshold - 0.75).abs() < 1e-6);
        assert_eq!(gamma, 50.0);

        let mut policy = LdPolicy::new(0.05);
        for &(similarity, correct) in &observations {
            policy.observe(0, similarity, correct);
        }
        let hits = (0..1000)
            .filter(|_| policy.decide(Some((0, 0.95))) == Decision::Hit)
            .count();
        assert!(hits > 0, "hits={hits}");

        let mut mature = LdPolicy::new(0.05);
        for _ in 0..200 {
            mature.observe(0, 0.8, true);
        }
        let mature_hits = (0..1000)
            .filter(|_| mature.decide(Some((0, 0.95))) == Decision::Hit)
            .count();
        assert!(mature_hits > 950, "mature_hits={mature_hits}");
    }

    #[test]
    fn all_false_logs_still_explore() {
        let observations = vec![(0.8, false); 6];
        assert!(fit_or_confident(&observations).is_none());
        let mut policy = LdPolicy::new(0.05);
        for &(similarity, correct) in &observations {
            policy.observe(0, similarity, correct);
        }
        for _ in 0..100 {
            assert_eq!(policy.decide(Some((0, 0.95))), Decision::Miss);
        }
    }

    #[test]
    fn ld3_uses_confident_fit_for_good_entries() {
        let mut policy = Ld3Policy::new(0.05);
        for _ in 0..6 {
            policy.observe(0, 0.8, true);
        }
        let hits = (0..1000)
            .filter(|_| policy.decide(Some((0, 0.95))) == Decision::Hit)
            .count();
        assert!(hits > 0, "hits={hits}");

        let mut mature = Ld3Policy::new(0.05);
        for _ in 0..200 {
            mature.observe(0, 0.8, true);
        }
        let mature_hits = (0..1000)
            .filter(|_| mature.decide(Some((0, 0.95))) == Decision::Hit)
            .count();
        assert!(mature_hits > 950, "mature_hits={mature_hits}");
    }

    #[test]
    fn ld_policy_explores_below_and_exploits_above_boundary() {
        let mut policy = LdPolicy::new(0.05);
        for (similarity, correct) in separable_observations() {
            policy.observe(0, similarity, correct);
        }
        policy.state.fits[0] = Some((0.8, 20.0));
        let mut low_misses = 0;
        let mut high_misses = 0;
        for _ in 0..100 {
            if policy.decide(Some((0, 0.6))) == Decision::Miss {
                low_misses += 1;
            }
            if policy.decide(Some((0, 0.95))) == Decision::Miss {
                high_misses += 1;
            }
        }
        assert!(low_misses > 90, "low misses={low_misses}");
        assert!(high_misses < 10, "high misses={high_misses}");
    }

    #[test]
    fn ld3_policy_is_deterministic() {
        let observations = separable_observations();
        let mut first = Ld3Policy::new(0.05);
        let mut second = Ld3Policy::new(0.05);
        for (similarity, correct) in observations {
            first.observe(0, similarity, correct);
            second.observe(0, similarity, correct);
        }
        for similarity in [0.6, 0.75, 0.85, 0.95] {
            assert_eq!(
                first.decide(Some((0, similarity))),
                second.decide(Some((0, similarity)))
            );
        }
    }

    #[test]
    fn insert_rule_follows_last_observation() {
        let mut policy = LdPolicy::new(0.05);
        assert!(policy.should_insert());
        policy.observe(0, 0.9, true);
        assert!(!policy.should_insert());
        policy.observe(0, 0.2, false);
        assert!(policy.should_insert());
    }

    #[test]
    fn adaptive_replay_completes_with_consistent_counts() {
        let classes = [0, 0, 1, 2, 1, 2, 0, 1];
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.99, 0.14],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, 0.99],
            vec![-0.99, 0.14],
            vec![0.98, 0.2],
            vec![0.1, 0.99],
        ];
        let embed_us = [10; 8];
        let mut policy = LdPolicy::new(0.05);
        let result = crate::replay::replay(&classes, &embeddings, &embed_us, &mut policy, &[1.0]);
        assert_eq!(result.queries, 8);
        assert!(result.hits <= result.queries);
        assert!(result.false_hits <= result.hits);
        assert!(result
            .events
            .iter()
            .all(|event| { event.decision == Decision::Miss || event.correct.is_some() }));
    }

    #[test]
    fn observations_keep_out_of_order_entries() {
        let mut state = EntryState::default();
        state.observe(5, 0.9, true);
        state.observe(5, 0.8, true);
        state.observe(2, 0.7, false);
        state.observe(5, 0.95, true);
        assert_eq!(state.observations[5].len(), 3);
        assert_eq!(state.observations[2].len(), 1);
    }
}
