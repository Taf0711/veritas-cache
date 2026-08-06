#!/usr/bin/env python3
"""Create benchmark charts for veritas-cache.

Read the result CSV files from the bench tests.
Write PNG charts into bench/charts/.
Use csv and matplotlib only.
"""

import csv
import os

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "bench", "results")
CHARTS_DIR = os.path.join(BASE_DIR, "bench", "charts")
SEPARATION_PATH = os.path.join(RESULTS_DIR, "separation.csv")
NN_DIFFICULTY_PATH = os.path.join(RESULTS_DIR, "nn_difficulty.csv")
STREAM_STATIC_PATH = os.path.join(RESULTS_DIR, "stream_static.csv")
STREAM_ADAPTIVE_PATH = os.path.join(RESULTS_DIR, "stream_adaptive.csv")


def read_separation(path):
    """Read the separation CSV. Return two lists of similarities."""
    same = []
    cross = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            similarity = float(row["sim"])
            if row["group"] == "same":
                same.append(similarity)
            else:
                cross.append(similarity)
    return same, cross


def read_nn_difficulty(path):
    """Read the nearest-neighbor CSV.

    Return three lists: best_same, best_cross, and a list of
    (nn_sim, nn_same_class) tuples.
    """
    best_same = []
    best_cross = []
    neighbor = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            best_same.append(float(row["best_same"]))
            best_cross.append(float(row["best_cross"]))
            neighbor.append(
                (float(row["nn_sim"]), row["nn_same_class"].strip().lower() == "true")
            )
    return best_same, best_cross, neighbor


def histogram_chart(pairs, path, title):
    """Draw an overlaid histogram of two similarity lists."""
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(
        pairs[0],
        bins=50,
        alpha=0.6,
        label=pairs[2][0],
        color="#2a6f97",
    )
    axis.hist(
        pairs[1],
        bins=50,
        alpha=0.6,
        label=pairs[2][1],
        color="#d95f02",
    )
    axis.set_title(title)
    axis.set_xlabel("Cosine similarity")
    axis.set_ylabel("Number of pairs")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Wrote {}".format(os.path.relpath(path, BASE_DIR)))


def static_curve_chart(neighbor, path, title):
    """Draw the static-threshold tradeoff curve.

    A query is a hit when its nearest-neighbor similarity reaches
    the threshold. The hit is false when the nearest neighbor is in
    a different class.
    """
    grid = [index / 100.0 for index in range(101)]
    hit_rates = []
    false_rates = []
    total = len(neighbor)
    for threshold in grid:
        hits = 0
        false_hits = 0
        for nn_sim, nn_same_class in neighbor:
            if nn_sim >= threshold:
                hits += 1
                if not nn_same_class:
                    false_hits += 1
        hit_rates.append(hits / total)
        false_rates.append(false_hits / total)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(hit_rates, false_rates, marker=".", markersize=3)
    for threshold in (0.85, 0.90, 0.95):
        index = int(round(threshold * 100))
        axis.annotate(
            "t={:.2f}".format(threshold),
            (hit_rates[index], false_rates[index]),
            textcoords="offset points",
            xytext=(8, 8),
        )
    axis.set_title(title)
    axis.set_xlabel("Hit rate")
    axis.set_ylabel("False-hit rate")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Wrote {}".format(os.path.relpath(path, BASE_DIR)))


def read_stream_static(path):
    """Read the streaming static CSV. Return the threshold rows."""
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                (
                    float(row["threshold"]),
                    float(row["hit_rate"]),
                    float(row["false_hit_rate"]),
                )
            )
    return rows


def stream_curve_chart(rows, path, title):
    """Draw the streaming static-threshold tradeoff curve.

    Points are connected in threshold order. Each point is labeled
    with its threshold value.
    """
    hit_rates = [row[1] for row in rows]
    false_rates = [row[2] for row in rows]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(hit_rates, false_rates, marker="o", markersize=5)
    for threshold, hit_rate, false_rate in rows:
        axis.annotate(
            "{:.2f}".format(threshold),
            (hit_rate, false_rate),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    axis.set_title(title)
    axis.set_xlabel("Hit rate")
    axis.set_ylabel("False-hit rate")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Wrote {}".format(os.path.relpath(path, BASE_DIR)))


def read_stream_adaptive(path):
    """Read the adaptive CSV. Return (policy, delta, hit, false) rows."""
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                (
                    row["policy"],
                    float(row["delta"]),
                    float(row["hit_rate"]),
                    float(row["false_hit_rate"]),
                )
            )
    return rows


def adaptive_curve_chart(static_rows, adaptive_rows, path, title):
    """Draw static and adaptive policies on one tradeoff chart.

    The y-axis uses a log scale because false-hit rates span
    several orders of magnitude. Each adaptive delta is labeled.
    """
    fig, axis = plt.subplots(figsize=(8, 5))
    static_hit = [row[1] for row in static_rows]
    static_false = [row[2] for row in static_rows]
    axis.plot(
        static_hit,
        static_false,
        marker=".",
        markersize=6,
        label="static",
    )
    for policy in ("gd", "ld", "ld3"):
        points = sorted(
            [row for row in adaptive_rows if row[0] == policy],
            key=lambda row: row[1],
        )
        hit = [row[2] for row in points]
        false = [row[3] for row in points]
        axis.plot(hit, false, marker="o", markersize=5, label=policy)
        for _, delta, hit_rate, false_rate in points:
            axis.annotate(
                "{:.2f}".format(delta),
                (hit_rate, false_rate),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )
    axis.set_title(title)
    axis.set_xlabel("Hit rate")
    axis.set_ylabel("False-hit rate")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Wrote {}".format(os.path.relpath(path, BASE_DIR)))


def main():
    """Run the chart pipeline. Skip each chart with missing input."""
    os.makedirs(CHARTS_DIR, exist_ok=True)

    same, cross = read_separation(SEPARATION_PATH)
    histogram_chart(
        (same, cross, ("same-class", "cross-class")),
        os.path.join(CHARTS_DIR, "separation_hist.png"),
        "Random pairs: same class vs cross class",
    )

    if os.path.exists(NN_DIFFICULTY_PATH):
        best_same, best_cross, neighbor = read_nn_difficulty(NN_DIFFICULTY_PATH)
        histogram_chart(
            (best_same, best_cross, ("best same-class", "best cross-class")),
            os.path.join(CHARTS_DIR, "nn_hist.png"),
            "Nearest neighbor: best same-class vs best cross-class",
        )
        static_curve_chart(
            neighbor,
            os.path.join(CHARTS_DIR, "static_curve.png"),
            "Static threshold tradeoff (leave-one-out, 20000-entry cache)",
        )
    else:
        print(
            "Skipped nn_hist.png and static_curve.png. "
            "Missing {}.".format(NN_DIFFICULTY_PATH)
        )

    if os.path.exists(STREAM_STATIC_PATH):
        rows = read_stream_static(STREAM_STATIC_PATH)
        stream_curve_chart(
            rows,
            os.path.join(CHARTS_DIR, "stream_curve.png"),
            "Streaming static threshold tradeoff (20000 prompts)",
        )
    else:
        print(
            "Skipped stream_curve.png. Missing {}.".format(STREAM_STATIC_PATH)
        )

    if os.path.exists(STREAM_STATIC_PATH) and os.path.exists(STREAM_ADAPTIVE_PATH):
        static_rows = read_stream_static(STREAM_STATIC_PATH)
        adaptive_rows = read_stream_adaptive(STREAM_ADAPTIVE_PATH)
        adaptive_curve_chart(
            static_rows,
            adaptive_rows,
            os.path.join(CHARTS_DIR, "adaptive_curve.png"),
            "Static vs adaptive policies (200000 queries)",
        )
    else:
        print(
            "Skipped adaptive_curve.png. Missing {} or {}.".format(
                STREAM_STATIC_PATH, STREAM_ADAPTIVE_PATH
            )
        )


if __name__ == "__main__":
    main()
