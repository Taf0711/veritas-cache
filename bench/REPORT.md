# Trace Benchmark Report

Date: 2026-08-03. Status: Phase 2 baseline measurement.

## Summary

This report measures the benchmark trace for veritas-cache. The trace comes from
Quora Question Pairs. The measurements show that one static similarity threshold
cannot give a high hit rate and a low error rate at the same time. This result
matches the central claim of the vCache paper (arXiv 2502.03771). The trace is a
valid test bed for per-entry adaptive thresholds in Phase 3.

## Data

- Source: GLUE QQP release of the Quora Question Pairs dataset, train and dev splits.
- 404,276 labeled pairs.
- Union-find over the duplicate pairs gives 60,397 equivalence classes with 2 or
  more members.
- The stream keeps 8,101 whole classes and 20,000 prompts. Seed 42.
- Same class means same correct response. Quora labels are noisy. This noise
  affects both false hits and false misses.
- The rebuild is deterministic. The digest is anchored in scripts/test_trace.py.

## Method

- Embedding model: all-MiniLM-L6-v2 (ONNX, 384 dimensions, CPU). Mean pooling and
  L2 normalization.
- Measurement 1: random pairs. 300 same-class pairs and 300 cross-class pairs,
  sampled with a fixed stride.
- Measurement 2: nearest neighbor. Leave-one-out over all 20,000 entries. 2,000
  queries sampled by stride 10. For each query, the best same-class similarity and
  the best cross-class similarity.
- All numbers come from the committed tests in src/main.rs. See the reproduction
  section.

## Results

### Random pairs

Figure: charts/separation_hist.png

| group | mean | p5 | p50 | p95 |
|---|---|---|---|---|
| same class (n 300) | 0.8366 | 0.6577 | 0.8498 | 0.9832 |
| cross class (n 300) | 0.0514 | -0.0637 | 0.0358 | 0.2096 |

### Nearest neighbor

Figure: charts/nn_hist.png

| statistic | mean | p5 | p50 | p95 |
|---|---|---|---|---|
| best same class (n 2000) | 0.9041 | 0.7527 | 0.9238 | 0.9935 |
| best cross class (n 2000) | 0.6447 | 0.4462 | 0.6349 | 0.8660 |

The mean cross-class similarity rises from 0.0514 for a random entry to 0.6447 for
the nearest entry. The nearest wrong entry is the error source for a semantic
cache.

Confusable share of queries:

- 13.20% have a cross-class neighbor at 0.80 or higher.
- 6.70% have a cross-class neighbor at 0.85 or higher.
- 2.10% have a cross-class neighbor at 0.90 or higher.

### Static threshold curve

Figure: charts/static_curve.png

Leave-one-out over the full cache. A query is a hit when its nearest neighbor
clears the threshold. The hit is false when the neighbor has a different class.

| threshold | hit rate | false-hit rate | misses (of 2000) |
|---|---|---|---|
| 0.30 | 1.0000 | 0.0335 | 0 |
| 0.50 | 1.0000 | 0.0335 | 0 |
| 0.70 | 0.9800 | 0.0290 | 40 |
| 0.80 | 0.8950 | 0.0180 | 210 |
| 0.85 | 0.7910 | 0.0145 | 418 |
| 0.90 | 0.6230 | 0.0085 | 754 |
| 0.95 | 0.3585 | 0.0040 | 1283 |
| 0.99 | 0.0780 | 0.0005 | 1844 |

A threshold of 0.85 gives a hit rate of 79.1% and serves 1.45% wrong answers. A
threshold of 0.95 cuts the error to 0.40% but the hit rate falls to 35.9%. No
single threshold gives a high hit rate at low error.

## Comparison with vCache

- vCache reports 1.7% error at a static threshold of 0.99 with 150,000 samples.
  Our cache holds 20,000 entries and shows 0.05% at 0.99. The paper shows that
  error grows with cache size. Our smaller cache shows the same direction at a
  smaller magnitude.
- vCache reports a 57% hit rate at under 0.5% error with adaptive thresholds. On
  this trace, a static threshold near 0.5% error gives about 36% hit rate. This
  gap is the Phase 3 target.
- The paper uses E5-large-v2, GTE-large, and text-embedding-3-small. Absolute
  similarity values do not transfer across models. The curve shapes do.

## Findings

1. The trace separates classes. Random same-class pairs embed at 0.84 mean
   similarity.
2. The trace has the failure mode that matters. The nearest wrong entry is often
   close.
3. A static global threshold cannot hold a high hit rate and a low error rate on
   this trace.
4. Per-entry thresholds have a measurable gap to exploit in Phase 3.

## Limits

- Quora labels are noisy. Some same-class pairs do not share one correct response.
- The nearest-neighbor run is a full-cache snapshot. The streaming harness in
  Phase 2 will give different operating points.
- MiniLM is weaker than the paper models. The absolute numbers are not comparable
  to the paper.
- 2,000 queries give a false-hit resolution of 0.05%.

## Reproduce

Run these commands from the repository root.

```bash
python3 scripts/build_trace.py
python3 scripts/test_trace.py
cargo test --release -- --ignored trace_similarity_separation --nocapture
cargo test --release -- --ignored trace_nearest_neighbor_difficulty --nocapture
python3 scripts/make_charts.py
```

The nearest-neighbor test takes about 23 minutes on an Apple Silicon laptop.
