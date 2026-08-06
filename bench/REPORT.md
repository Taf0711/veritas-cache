# Benchmark Report: Static and Adaptive Thresholds for a Semantic Cache

Date: 2026-08-04. Status: Phase 3 measurement complete.

## Abstract

We replay 200,000 LLM-style queries through a semantic cache under four decision
policies. One policy uses a single static similarity threshold. Three policies
learn the threshold online. The per-entry adaptive policy with a confidence
bound holds its user-set error budget at every operating point. The measured
false-hit rate stays 10 to 20 times below the budget. At matched error, the
per-entry policy beats the global adaptive policy by about 20 points of hit
rate. A tuned static threshold still reaches a higher raw hit rate on this
trace. The adaptive policies need no labeled tuning data. The static threshold
does.

## 1. Objective

Measure whether per-entry adaptive similarity thresholds give a semantic cache a
bounded, controllable false-hit rate. Compare against a static global threshold
on identical traffic. The reference claim comes from vCache (arXiv 2502.03771):
static thresholds give no error guarantee, and per-entry adaptation reduces
error at matched hit rate.

## 2. Hypotheses

- H1. The false-hit rate of each adaptive policy stays below its budget delta.
- H2. At matched false-hit rate, the per-entry policy with bounds (ld3) beats
  the global adaptive policy (gd) on hit rate.
- H3. At matched false-hit rate, adaptive policies beat the best static
  threshold on hit rate.

H1 and H2 hold. H3 does not hold on this trace. Section 5 discusses why.

## 3. Method

### 3.1 Dataset

The trace comes from the GLUE QQP release of Quora Question Pairs, train and dev
splits. The build takes 404,276 labeled pairs. Union-find over the duplicate
pairs gives 60,397 equivalence classes with 2 or more members. The stream keeps
8,101 whole classes and 20,000 prompts. Same class means same correct response.
The build is deterministic with seed 42. A SHA-256 digest anchors the artifact
in scripts/test_trace.py.

### 3.2 Traffic model

The harness repeats the 20,000-prompt stream 10 times in whole-stream order.
The replay has 200,000 queries. Repetition models the recurrence of real LLM
traffic. Quora classes average 2.5 members. Without repetition, no cache entry
can accumulate the 5 observations that per-entry fitting needs.

### 3.3 Apparatus

- Embedding model: all-MiniLM-L6-v2, ONNX, 384 dimensions, CPU. Mean pooling and
  L2 normalization.
- Approximate nearest neighbor: hnsw_rs, M 16, ef_construction 100, ef_search
  32, k 1, cosine distance.
- Hardware: Apple Silicon laptop, release build.
- Decision draws: ChaCha8 random generator, seed 42.

### 3.4 Policies under test

- gs: one global static cosine threshold. A hit serves when the nearest
  neighbor clears the threshold. Every miss inserts a new entry.
- gd: one global sigmoid fit over all observations. Explore with probability
  tau = 1 - delta / (1 - alpha), where alpha estimates correctness at the
  measured similarity.
- ld: one sigmoid fit per cache entry. Same exploration rule as gd.
- ld3: like ld, plus a bootstrap confidence bound on the fitted boundary. The
  bound uses 20 resamples and a 20-point epsilon scan.

Shared rules follow the vCache design. Observations come only from explores. A
hit produces no observation because a served answer has no ground truth. A miss
inserts a new entry only when the cached answer was wrong. A fit needs at least
5 observations. A Laplace cap, (n+1)/(n+2), bounds the correctness estimate at
small n. An all-correct log fits a steep confident sigmoid. An all-wrong log
always explores.

### 3.5 Latency model

Lookup latency is measured per query. It contains embedding, search, and
decision time. Miss latency adds a sample from a seeded lognormal model with
median 800 ms and sigma 0.6. The model is not a measurement. Replace
bench/miss_latencies.txt with real recordings to change it.

### 3.6 Metrics

- Hit rate: hits over queries.
- False-hit rate: wrong answers served over queries. This is the error rate.
- False-miss rate: misses with a same-class entry already cached, over queries.
- Latency: p50 and p99 from HDR histograms.

## 4. Results

### 4.1 Trace difficulty

Leave-one-out over the full 20,000-entry cache, 2,000 queries. Figure:
charts/nn_hist.png.

| statistic | mean | p5 | p50 | p95 |
|---|---|---|---|---|
| best same class | 0.9041 | 0.7527 | 0.9238 | 0.9935 |
| best cross class | 0.6447 | 0.4462 | 0.6349 | 0.8660 |

The nearest wrong entry is close. 6.70% of queries have a cross-class neighbor
at 0.85 or higher. Random wrong pairs embed at 0.05 mean similarity. The
nearest wrong entry embeds at 0.64. The nearest neighbor is the error source.
Figure: charts/separation_hist.png.

### 4.2 Static threshold sweep

Streaming replay, 200,000 queries. Figure: charts/stream_curve.png.

| threshold | hit rate | false-hit rate | p99 total |
|---|---|---|---|
| 0.30 | 0.9958 | 0.8955 | 41 ms |
| 0.50 | 0.9777 | 0.4709 | 888 ms |
| 0.70 | 0.9587 | 0.1143 | 1229 ms |
| 0.80 | 0.9486 | 0.0386 | 1347 ms |
| 0.85 | 0.9394 | 0.0207 | 1455 ms |
| 0.90 | 0.9272 | 0.0084 | 1568 ms |
| 0.95 | 0.9100 | 0.0030 | 1683 ms |
| 0.99 | 0.8908 | 0.0003 | 1791 ms |

### 4.3 Adaptive ladder

Streaming replay, 200,000 queries. Figure: charts/adaptive_curve.png.

| policy | delta | hit rate | false-hit rate |
|---|---|---|---|
| gd | 0.01 | 0.1255 | 0.0007 |
| gd | 0.02 | 0.2298 | 0.0013 |
| gd | 0.05 | 0.4652 | 0.0037 |
| gd | 0.10 | 0.6867 | 0.0080 |
| gd | 0.20 | 0.8376 | 0.0191 |
| ld | 0.01 | 0.1022 | 0.0001 |
| ld | 0.02 | 0.1879 | 0.0001 |
| ld | 0.05 | 0.3869 | 0.0004 |
| ld | 0.10 | 0.5905 | 0.0008 |
| ld | 0.20 | 0.7294 | 0.0024 |
| ld3 | 0.01 | 0.1192 | 0.0001 |
| ld3 | 0.02 | 0.2203 | 0.0004 |
| ld3 | 0.05 | 0.4436 | 0.0011 |
| ld3 | 0.10 | 0.6580 | 0.0031 |
| ld3 | 0.20 | 0.7695 | 0.0102 |

Every adaptive point holds its budget. The measured false-hit rate stays 10 to
20 times below delta.

### 4.4 Head-to-head at matched error

At a false-hit rate near 0.003, the operating points are:

| policy | operating point | hit rate |
|---|---|---|
| ld3 | delta 0.10, measured error 0.0031 | 0.6580 |
| gd | interpolated to error 0.0031 | about 0.44 |
| gs | threshold 0.95, measured error 0.0030 | 0.9100 |

H2 holds. ld3 beats gd by about 20 points of hit rate at matched error. This
reproduces the per-entry advantage that the vCache paper reports.

H3 does not hold. The static threshold at 0.95 reaches 0.9100 hit rate at the
same error. Exact repeats dominate this trace. They clear any high threshold at
no risk. The adaptive policies pay an exploration tax on every entry, and 10
loops do not amortize it.

The lookup p50 is about 18.6 ms for every policy. Embedding dominates it. A hit
costs the lookup only. Under the latency model, a miss adds a median of 800 ms.
A cache hit is about 43 times faster than a miss at the median.

## 5. Discussion

The adaptive mechanism works as designed. The error budget holds everywhere,
with a wide margin. The per-entry fit with a confidence bound converts that
budget into more hits than a global fit at the same error.

The static threshold wins the raw operating point on this trace. Two properties
of the trace drive this. Exact repeats are safe at any threshold. The
confusable share above 0.95 is small. The static threshold has a hidden cost
instead: nothing tells the operator which threshold is safe. Its error must be
measured on labeled data, and the paper shows it grows with cache size. The
adaptive policies trade hit rate for a guarantee that needs no labels.

The exploration tax deserves attention. Every new entry explores until its log
matures. Traffic with more recurrence per entry amortizes the tax further. The
10-loop horizon is the conservative end of real workloads.

## 6. Threats to validity

- Quora duplicate labels are noisy. The noise inflates both false hits and
  false misses.
- The loop model repeats each prompt exactly 10 times. Real recurrence is
  uneven. Exact repeats favor high static thresholds.
- Miss latency is a parametric model, not a measurement. Hit and miss rates do
  not depend on it. Total latency does.
- MiniLM is weaker than the embedding models in the paper. Absolute similarity
  values do not transfer. The curve shapes do.
- hnsw_rs builds its index with an unseeded random generator. Results vary
  slightly between runs.
- One dataset, one embedding model, one harness. 200,000 queries give a
  false-hit resolution of 0.005%.
- The gd policy refits its global sigmoid every 64 observations. This
  approximation can shift its operating points slightly.

## 7. Reproduction

Run these commands from the repository root.

```bash
python3 scripts/build_trace.py
python3 scripts/test_trace.py
cargo test
cargo test --release -- --ignored trace_similarity_separation --nocapture
cargo test --release -- --ignored trace_nearest_neighbor_difficulty --nocapture
cargo run --release --bin bench
python3 scripts/make_charts.py
```

The nearest-neighbor test takes about 23 minutes. The full replay takes about
40 minutes. Both cache their embeddings for reruns.

## References

- vCache: Semantic Caching with Verified Error Bounds. arXiv 2502.03771, ICLR
  2026. Prior art for the adaptive threshold design. This project implements
  the idea from the paper text only. No vCache code was read or used.
- Quora Question Pairs, GLUE QQP release.
- all-MiniLM-L6-v2, Xenova ONNX export.
