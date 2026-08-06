use crate::policy::{Decision, Policy};
use hdrhistogram::Histogram;
use hnsw_rs::prelude::*;
use std::time::Instant;

// Store one replay decision for the reference policy log.
#[derive(Debug, Clone)]
pub struct ReplayEvent {
    pub id: usize,
    pub similarity: Option<f32>,
    pub decision: Decision,
    pub correct: Option<bool>,
    pub lookup_us: u64,
}

// Store replay counts and latency distributions.
pub struct ReplayResult {
    pub queries: usize,
    pub hits: usize,
    pub false_hits: usize,
    pub false_misses: usize,
    pub lookup_us: Histogram<u64>,
    pub total_us: Histogram<u64>,
    pub events: Vec<ReplayEvent>,
}

fn micros(millis: f64) -> u64 {
    if millis.is_finite() && millis > 0.0 {
        (millis * 1_000.0).round() as u64
    } else {
        0
    }
}

// Replay one ordered trace against a fresh in-memory HNSW index.
pub fn replay(
    classes: &[i64],
    embeddings: &[Vec<f32>],
    embed_us: &[u64],
    policy: &mut dyn Policy,
    miss_latencies_ms: &[f64],
) -> ReplayResult {
    assert_eq!(
        classes.len(),
        embeddings.len(),
        "class and embedding counts differ"
    );
    assert_eq!(
        classes.len(),
        embed_us.len(),
        "class and timing counts differ"
    );
    assert!(!embeddings.is_empty(), "replay needs at least one query");

    let capacity = embeddings.len().max(100);
    let index = Hnsw::new(16, capacity, 16, 100, DistCosine {});
    let mut entry_classes: Vec<i64> = Vec::with_capacity(embeddings.len());
    let mut hits = 0;
    let mut false_hits = 0;
    let mut false_misses = 0;
    let mut miss_count = 0;
    let mut lookup_histogram = Histogram::new(3).expect("valid lookup histogram");
    let mut total_histogram = Histogram::new(3).expect("valid total histogram");
    let mut events = Vec::with_capacity(embeddings.len());

    for (query_index, embedding) in embeddings.iter().enumerate() {
        let search_started = Instant::now();
        let neighbor = if entry_classes.is_empty() {
            None
        } else {
            index
                .search(embedding, 1, 32)
                .into_iter()
                .next()
                .map(|item| (item.d_id, 1.0 - item.distance))
        };
        let decision = policy.decide(neighbor);
        let lookup_us =
            embed_us[query_index].saturating_add(search_started.elapsed().as_micros() as u64);
        lookup_histogram
            .record(lookup_us)
            .expect("lookup latency fits histogram");

        let (correct, total_us) = match (decision, neighbor) {
            (Decision::Hit, Some((entry, _similarity))) => {
                let is_correct = entry_classes[entry] == classes[query_index];
                hits += 1;
                if !is_correct {
                    false_hits += 1;
                }
                (Some(is_correct), lookup_us)
            }
            (Decision::Hit, None) => (Some(false), lookup_us),
            (Decision::Miss, neighbor) => {
                if entry_classes
                    .iter()
                    .any(|&class| class == classes[query_index])
                {
                    false_misses += 1;
                }
                if let Some((entry, similarity)) = neighbor {
                    let is_correct = entry_classes[entry] == classes[query_index];
                    policy.observe(entry, similarity, is_correct);
                }
                let llm_us = miss_latencies_ms
                    .get(miss_count % miss_latencies_ms.len().max(1))
                    .copied()
                    .map(micros)
                    .unwrap_or(0);
                miss_count += 1;
                if policy.should_insert() {
                    let entry_id = entry_classes.len();
                    index.insert((embedding, entry_id));
                    entry_classes.push(classes[query_index]);
                }
                (None, lookup_us.saturating_add(llm_us))
            }
        };

        total_histogram
            .record(total_us)
            .expect("total latency fits histogram");
        events.push(ReplayEvent {
            id: query_index,
            similarity: neighbor.map(|(_, similarity)| similarity),
            decision,
            correct,
            lookup_us,
        });
    }

    ReplayResult {
        queries: classes.len(),
        hits,
        false_hits,
        false_misses,
        lookup_us: lookup_histogram,
        total_us: total_histogram,
        events,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::StaticPolicy;

    #[derive(Default)]
    struct ObservationCounter {
        observations: usize,
    }

    impl Policy for ObservationCounter {
        fn decide(&mut self, _neighbor: Option<(usize, f32)>) -> Decision {
            Decision::Miss
        }

        fn should_insert(&self) -> bool {
            true
        }

        fn observe(&mut self, _entry: usize, _sim: f32, _correct: bool) {
            self.observations += 1;
        }
    }

    #[test]
    fn replay_observes_only_misses_with_neighbors() {
        let classes = [0, 0, 1];
        let embeddings = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let embed_us = [10, 10, 10];
        let mut policy = ObservationCounter::default();
        replay(&classes, &embeddings, &embed_us, &mut policy, &[1.0]);
        assert_eq!(policy.observations, 2);
    }

    #[test]
    fn replay_counts_hits_false_hits_false_misses_and_inserts_misses() {
        let classes = [0, 0, 1, 2];
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.4358899],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let embed_us = [10, 10, 10, 10];
        let mut policy = StaticPolicy { threshold: 0.995 };
        let result = replay(&classes, &embeddings, &embed_us, &mut policy, &[1.0]);

        assert_eq!(result.queries, 4);
        assert_eq!(result.hits, 1);
        assert_eq!(result.false_hits, 1);
        assert_eq!(result.false_misses, 1);
        assert_eq!(result.events.len(), 4);
        assert_eq!(result.events[0].decision, Decision::Miss);
        assert_eq!(result.events[1].decision, Decision::Miss);
        assert_eq!(result.events[2].decision, Decision::Hit);
        assert_eq!(result.events[3].decision, Decision::Miss);
        assert_eq!(result.total_us.len(), 4);
    }
}
