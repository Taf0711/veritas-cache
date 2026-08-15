#!/usr/bin/env python3
"""Prototype: does a sigmoid fit stage-pooled agent observations?

Read the baseline shadow logs of the splice experiment runs. Judge each
shadow row with the same rule the proxy uses. Fit a two-parameter logistic
per stage, mirroring src/adaptive.rs. Simulate a stage-pooled policy in
request order and measure hit rate and false-hit rate against the judged
ground truth at delta 0.05.

Usage: pooled_fit.py [run_dir ...]
Default: the three dev-binary baseline runs.
"""
import json
import math
import os
import sqlite3
import sys

MIN_OBSERVATIONS = 5
FIT_STEPS = 100
FIT_RATE = 0.5
DELTA = 0.05

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RUNS = [
    "bench/splice/runs/20260814-161810",
    "bench/splice/runs/20260815-121628",
    "bench/splice/runs/20260815-125316",
]


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def response_content(response_json):
    """Mirror response_content in src/lib.rs. None when content is not a string."""
    try:
        value = json.loads(response_json)
        content = value["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    if not isinstance(content, str):
        return None
    return " ".join(content.split())


def fit(observations):
    """Mirror fit in src/adaptive.rs. Return (t, gamma) or None."""
    if len(observations) < MIN_OBSERVATIONS:
        return None
    labels = {c for _, c in observations}
    if len(labels) == 1:
        return None
    t = sum(s for s, _ in observations) / len(observations)
    gamma = 1.0
    count = len(observations)
    for _ in range(FIT_STEPS):
        dgamma = 0.0
        dt = 0.0
        for s, c in observations:
            probability = sigmoid(gamma * (s - t))
            error = (1.0 if c else 0.0) - probability
            dgamma += error * (s - t)
            dt += error * -gamma
        gamma += FIT_RATE * dgamma / count
        t += FIT_RATE * dt / count
    if math.isfinite(t) and math.isfinite(gamma):
        return (t, gamma)
    return None


def fit_or_confident(observations):
    """Mirror fit_or_confident in src/adaptive.rs."""
    fitted = fit(observations)
    if fitted:
        return fitted
    if len(observations) < MIN_OBSERVATIONS or not all(c for _, c in observations):
        return None
    minimum = min(s for s, _ in observations)
    return (minimum - 0.05, 50.0)


def load_rows(run_dir):
    """Judged shadow rows for one baseline arm: (seq, stage, similarity, correct)."""
    db_path = os.path.join(run_dir, "baseline", "cache.db")
    conn = sqlite3.connect(db_path)
    rows = []
    skipped = 0
    for rowid, key_hash, sim, would, fresh in conn.execute(
        "SELECT id, key_hash, similarity, would_serve_json, fresh_json"
        " FROM shadow_log WHERE similarity IS NOT NULL AND fresh_json IS NOT NULL"
        " ORDER BY id"
    ):
        cached = response_content(would or "")
        live = response_content(fresh)
        if cached is None or live is None:
            skipped += 1
            continue
        stage_row = conn.execute(
            "SELECT json_extract(request_json, '$.prompt_cache_key')"
            " FROM entries WHERE key_hash = ?1",
            (key_hash,),
        ).fetchone()
        stage = "unknown"
        if stage_row and stage_row[0] and ":" in stage_row[0]:
            stage = stage_row[0].split(":", 1)[1]
        rows.append((rowid, stage, sim, cached == live))
    conn.close()
    return rows, skipped


def simulate(stage_rows):
    """Run the stage-pooled policy in request order. Return per-stage tallies.

    Serve with the paper rule: explore with probability tau = clip(1 - delta/(1-alpha)).
    Report expected serves and a deterministic sample with a fixed seed.
    """
    import random

    rng = random.Random(42)
    observations = []
    expected_serves = 0.0
    sampled_hits = 0
    sampled_false = 0
    serves = []
    for rowid, stage, sim, correct in stage_rows:
        fitted = fit_or_confident(observations)
        serve_probability = 0.0
        if fitted:
            t, gamma = fitted
            alpha = sigmoid(gamma * (sim - t))
            explore = max(0.0, min(1.0, 1.0 - DELTA / (1.0 - alpha))) if alpha < 1.0 else 0.0
            serve_probability = 1.0 - explore
        expected_serves += serve_probability
        if rng.random() < serve_probability:
            sampled_hits += 1
            if not correct:
                sampled_false += 1
            serves.append((rowid, sim, correct))
        else:
            observations.append((sim, correct))
    return observations, expected_serves, sampled_hits, sampled_false, serves


def main():
    run_dirs = sys.argv[1:] or [os.path.join(REPO, d) for d in DEFAULT_RUNS]
    all_rows = []
    total_skipped = 0
    for run_dir in run_dirs:
        rows, skipped = load_rows(run_dir)
        all_rows.extend(rows)
        total_skipped += skipped
        print(f"{os.path.basename(run_dir)}: {len(rows)} judged rows, {skipped} unjudgeable")

    by_stage = {}
    for row in all_rows:
        by_stage.setdefault(row[1], []).append(row)

    print(f"\ntotal judged: {len(all_rows)}, unjudgeable (null content): {total_skipped}")
    for stage, rows in sorted(by_stage.items()):
        rows.sort(key=lambda r: r[0])
        obs = [(sim, correct) for _, _, sim, correct in rows]
        fitted = fit_or_confident(obs)
        print(f"\nstage {stage}: {len(rows)} observations")
        curve = " ".join(f"{sim:.2f}{'+' if correct else '-'}" for _, _, sim, correct in rows)
        print(f"  curve (sim, +correct/-wrong): {curve}")
        if fitted:
            t, gamma = fitted
            print(f"  fit: t={t:.3f} gamma={gamma:.2f}")
        else:
            print("  fit: none (too few or single-label)")
        remaining, expected, hits, false_hits, serves = simulate(rows)
        total = len(rows)
        print(f"  simulated: expected serves {expected:.1f}/{total}, sampled {hits}, false hits {false_hits}")
        for rowid, sim, correct in serves:
            mark = "WRONG" if not correct else "ok"
            print(f"    served row {rowid} sim={sim:.3f} {mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
