#!/usr/bin/env python3
"""Build the Phase 2 benchmark trace for veritas-cache.

The trace is an ordered stream of prompts with equivalence-class labels.
Same class means the same correct response.
The source is the Quora Question Pairs dataset.
"""

import csv
import io
import json
import os
import random
import ssl
import sys
import urllib.request
import zipfile

try:
    import certifi
except ImportError:
    certifi = None

# Constants.
SEED = 42
MAX_PROMPTS = 20000
MIN_CLASS_SIZE = 2
QUORA_PARQUET_URL = "https://datasets-server.huggingface.co/parquet?dataset=quora"
QQP_ZIP_URL = "https://dl.fbaipublicfiles.com/glue/data/QQP.zip"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(BASE_DIR, "bench")
CACHE_DIR = os.path.join(BENCH_DIR, "cache")
CACHE_ZIP = os.path.join(CACHE_DIR, "QQP.zip")
OUTPUT_PATH = os.path.join(BENCH_DIR, "trace.jsonl")
README_PATH = os.path.join(BENCH_DIR, "README.md")


class SourceError(RuntimeError):
    """Raised when a data source cannot be fetched."""


def http_get(url, timeout=60):
    """Fetch a URL. Return the response body as bytes."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "veritas-cache-trace-builder/0.1"}
    )
    if certifi is not None:
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            return response.read()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def try_quora_parquet():
    """Try the Hugging Face datasets-server parquet export for the quora dataset.

    Return the list of parquet file URLs or raise SourceError.
    """
    try:
        body = http_get(QUORA_PARQUET_URL)
        data = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise SourceError("quora parquet endpoint failed: {}".format(exc))
    if "parquet_files" not in data:
        raise SourceError(
            "quora parquet endpoint has no parquet_files: {}".format(data)
        )
    return [f["url"] for f in data["parquet_files"]]


def download_qqp_zip():
    """Download the GLUE QQP zip archive. Cache it on disk. Return its path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CACHE_ZIP):
        return CACHE_ZIP
    print("Downloading QQP.zip from GLUE. This is about 42 MB.")
    try:
        body = http_get(QQP_ZIP_URL, timeout=300)
    except Exception as exc:
        raise SourceError("QQP.zip download failed: {}".format(exc))
    with open(CACHE_ZIP, "wb") as handle:
        handle.write(body)
    return CACHE_ZIP


def read_pairs_from_zip(zip_path):
    """Read labeled question pairs from the GLUE QQP train and dev splits.

    Return a list of (question1, question2, is_duplicate) tuples.
    """
    pairs = []
    with zipfile.ZipFile(zip_path) as archive:
        by_name = {os.path.basename(name): name for name in archive.namelist()}
        for member in ["train.tsv", "dev.tsv"]:
            with archive.open(by_name[member]) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                reader = csv.DictReader(text, delimiter="\t")
                for row in reader:
                    pairs.append(
                        (
                            row["question1"].strip(),
                            row["question2"].strip(),
                            row["is_duplicate"].strip(),
                        )
                    )
    return pairs


def read_pairs():
    """Fetch question pairs. Use the quora parquet export first.

    Fall back to the GLUE QQP zip archive when the parquet export fails.
    Return a tuple of (pairs, source_name, source_note).
    """
    try:
        urls = try_quora_parquet()
        raise SourceError(
            "parquet export exists but no parquet reader is installed: {}".format(urls)
        )
    except SourceError as exc:
        print("Parquet source unavailable: {}".format(exc))
        print("Falling back to the GLUE QQP zip archive.")
    zip_path = download_qqp_zip()
    pairs = read_pairs_from_zip(zip_path)
    return (
        pairs,
        "GLUE QQP zip archive",
        "The Hugging Face datasets-server endpoint for dataset quora returns a "
        "renamed-dataset error. The GLUE QQP zip archive contains the same "
        "Quora Question Pairs data.",
    )


class UnionFind:
    """Disjoint-set data structure over question texts."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, item):
        """Return the root of the set that contains item."""
        parent = self.parent.setdefault(item, item)
        self.rank.setdefault(item, 0)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a, b):
        """Merge the sets that contain a and b."""
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def build_classes(pairs):
    """Group duplicate questions into equivalence classes.

    Return a dict from class key to the sorted list of member texts.
    """
    union_find = UnionFind()
    for q1, q2, is_duplicate in pairs:
        if is_duplicate == "1":
            union_find.union(q1, q2)
    classes = {}
    for q1, q2, _ in pairs:
        for text in (q1, q2):
            if text:
                classes.setdefault(union_find.find(text), set()).add(text)
    classes = _clean_members(classes)
    # Drop classes with fewer than MIN_CLASS_SIZE members.
    kept = {}
    for key, members in classes.items():
        if len(members) >= MIN_CLASS_SIZE:
            kept[key] = sorted(members)
    return kept


def _clean_members(classes):
    """Remove empty texts from every class."""
    cleaned = {}
    for key, members in classes.items():
        cleaned[key] = [text for text in members if text]
    return cleaned


def build_stream(classes, seed=SEED, max_prompts=MAX_PROMPTS):
    """Build the ordered prompt stream from whole classes.

    Assign stable class ids from the full sorted class list.
    Shuffle the class order with the seed.
    Take whole classes until the member total reaches the cap.
    Shuffle the emitted members with the same seed.
    Return the list of (prompt, class_id) tuples.
    """
    # Sort classes by their smallest member text. This gives stable class ids.
    ordered_keys = sorted(classes, key=lambda key: (classes[key][0], key))
    class_ids = {key: index for index, key in enumerate(ordered_keys)}

    # Shuffle the class order. Take whole classes only.
    rng = random.Random(seed)
    shuffled_keys = list(ordered_keys)
    rng.shuffle(shuffled_keys)

    kept_keys = []
    total = 0
    for key in shuffled_keys:
        # Skip a class that does not fit. The cap is never exceeded.
        if total + len(classes[key]) > max_prompts:
            continue
        kept_keys.append(key)
        total += len(classes[key])

    # Emit all members of the kept classes, then shuffle the stream.
    stream = []
    for key in kept_keys:
        for text in classes[key]:
            stream.append((text, class_ids[key]))
    rng.shuffle(stream)
    return stream


def write_trace(stream, path):
    """Write the trace as one JSON object per line."""
    with open(path, "w", encoding="utf-8") as handle:
        for index, (prompt, class_id) in enumerate(stream):
            handle.write(
                json.dumps(
                    {"id": index, "prompt": prompt, "class_id": class_id},
                    ensure_ascii=False,
                )
            )
            handle.write("\n")


def write_readme(pairs, classes, stream, source_name, source_note):
    """Write the short benchmark README."""
    class_count = len(classes)
    prompt_count = len(stream)
    with open(README_PATH, "w", encoding="utf-8") as handle:
        handle.write("# Trace: Quora Question Pairs\n\n")
        handle.write("## Source\n\n")
        handle.write(
            "The trace comes from the Quora Question Pairs dataset.\n"
            "The data is the GLUE QQP release.\n"
        )
        handle.write("\n{}\n".format(source_note))
        handle.write("\n## Contents\n\n")
        handle.write("- {} labeled pairs (train and dev)\n".format(len(pairs)))
        handle.write("- {} equivalence classes with 2 or more members\n".format(class_count))
        handle.write("- {} classes in the stream\n".format(len(set(class_id for _, class_id in stream))))
        handle.write("- {} prompts in the stream\n".format(prompt_count))
        handle.write(
            "- stream seed {}\n".format(SEED)
        )
        handle.write("\n## Label noise\n\n")
        handle.write(
            "Quora duplicate labels are noisy.\n"
            "A duplicate label does not always mean the same correct response.\n"
            "The harness treats same class as same correct response.\n"
        )
        handle.write("\n## Reproduce\n\n")
        handle.write("Run this command from the repository root.\n\n")
        handle.write("```bash\npython3 scripts/build_trace.py\n```\n")


def self_check(stream, classes, path):
    """Run assert-based checks on the trace."""
    # The line count matches the stream length.
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    assert len(lines) == len(stream), "line count mismatch"

    # Class ids are stable. Re-derive the sorted class order and compare.
    ordered_keys = sorted(classes, key=lambda key: (classes[key][0], key))
    class_ids = {key: index for index, key in enumerate(ordered_keys)}
    # Build one reverse map from prompt to its class id. Use it for both checks.
    prompt_to_class = {}
    for key in ordered_keys:
        for text in classes[key]:
            prompt_to_class[text] = class_ids[key]

    # Every prompt maps to the class that produced it.
    for index, (line, (prompt, class_id)) in enumerate(zip(lines, stream)):
        record = json.loads(line)
        assert record["prompt"] == prompt, "prompt mismatch at id {}".format(record["id"])
        assert record["class_id"] == class_id, "class id mismatch at id {}".format(record["id"])
        assert record["id"] == index, "id sequence mismatch at line {}".format(index)
        assert prompt in prompt_to_class, "prompt {} belongs to no kept class".format(prompt)
        assert prompt_to_class[prompt] == class_id, "unstable class id for {}".format(prompt)

    # Every class present in the trace has at least MIN_CLASS_SIZE members.
    # This catches the split-class bug: no class is cut by the cap.
    present_sizes = {}
    for _, class_id in stream:
        present_sizes[class_id] = present_sizes.get(class_id, 0) + 1
    split = [cid for cid, size in present_sizes.items() if size < MIN_CLASS_SIZE]
    assert not split, "split classes in trace: {}".format(split[:5])

    # The stream length is capped as configured.
    assert len(stream) <= MAX_PROMPTS, "stream exceeds the cap"


def main():
    """Run the whole build. Exit non-zero on failure."""
    pairs, source_name, source_note = read_pairs()
    classes = build_classes(pairs)
    stream = build_stream(classes)
    os.makedirs(BENCH_DIR, exist_ok=True)
    write_trace(stream, OUTPUT_PATH)
    write_readme(pairs, classes, stream, source_name, source_note)
    self_check(stream, classes, OUTPUT_PATH)
    print("Source: {}".format(source_name))
    print("Labeled pairs: {}".format(len(pairs)))
    print("Classes with >= {} members: {}".format(MIN_CLASS_SIZE, len(classes)))
    print("Prompts in stream: {}".format(len(stream)))
    print("Classes in stream: {}".format(len(set(class_id for _, class_id in stream))))
    print("Trace written to {}".format(OUTPUT_PATH))
    print("README written to {}".format(README_PATH))
    print("Self-check passed.")


if __name__ == "__main__":
    try:
        main()
    except SourceError as exc:
        print("FATAL: {}".format(exc))
        sys.exit(1)
