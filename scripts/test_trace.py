#!/usr/bin/env python3
"""Regression tests for the Phase 2 benchmark trace.

Run the checks against the committed bench/trace.jsonl.
The script reads the trace and prints PASS or FAIL per check.
Add --rebuild to regenerate the trace and compare the result.
Exit code is non-zero when any check fails.
"""

import hashlib
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TRACE_PATH = os.path.join(REPO_ROOT, "bench", "trace.jsonl")
BUILDER_PATH = os.path.join(SCRIPT_DIR, "build_trace.py")

EXPECTED_LINES = 20000
EXPECTED_CLASSES = 8101
GOLDEN_SHA256 = "091a12477656aa85c772416769cd69b96d65461dfb98305fc79dcc519cea4353"

# One entry per check. Each entry is (name, ok, detail).
CHECKS = []


def record(name, ok, detail=""):
    """Add one check result to the list."""
    CHECKS.append((name, ok, detail))


def file_sha256(path):
    """Return the SHA-256 digest of a file as a hex string."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace():
    """Load the trace file. Return the list of records."""
    with open(TRACE_PATH, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_checks():
    """Run every regression check and collect the results."""
    entries = load_trace()

    # Every line is valid JSON with exactly the three expected keys.
    schema_ok = all(
        set(entry) == {"id", "prompt", "class_id"}
        and isinstance(entry["id"], int)
        and isinstance(entry["prompt"], str)
        and isinstance(entry["class_id"], int)
        for entry in entries
    )
    record("schema", schema_ok, "each line has id, prompt, class_id with correct types")

    # The line count is correct and the ids are sequential from zero.
    ids = [entry["id"] for entry in entries]
    count_ok = len(entries) == EXPECTED_LINES
    sequential_ok = ids == list(range(EXPECTED_LINES))
    record("count and ids", count_ok and sequential_ok,
           "{} lines, ids 0..{}".format(len(entries), EXPECTED_LINES - 1))

    # No prompt is empty or repeated.
    prompts = [entry["prompt"] for entry in entries]
    empty_ok = all(prompt.strip() for prompt in prompts)
    unique_ok = len(prompts) == len(set(prompts))
    record("prompt text", empty_ok and unique_ok,
           "no empty prompts, {} duplicate texts".format(len(prompts) - len(set(prompts))))

    # No prompt appears in two different classes.
    prompt_to_class = {}
    conflict = False
    for entry in entries:
        previous = prompt_to_class.get(entry["prompt"])
        if previous is not None and previous != entry["class_id"]:
            conflict = True
            break
        prompt_to_class[entry["prompt"]] = entry["class_id"]
    record("class consistency", not conflict, "no prompt belongs to two classes")

    # Every class has at least two members and the class count is correct.
    class_sizes = {}
    for entry in entries:
        class_sizes[entry["class_id"]] = class_sizes.get(entry["class_id"], 0) + 1
    size_ok = all(size >= 2 for size in class_sizes.values())
    count_ok = len(class_sizes) == EXPECTED_CLASSES
    record("class sizes", size_ok and count_ok,
           "{} classes, min size {}".format(len(class_sizes), min(class_sizes.values())))

    # The file content matches the golden digest.
    digest = file_sha256(TRACE_PATH)
    record("golden digest", digest == GOLDEN_SHA256,
           "update the hash only if the trace was intentionally rebuilt")


def run_rebuild_check():
    """Rebuild the trace and compare the result with the committed file."""
    before = file_sha256(TRACE_PATH)
    result = subprocess.run(
        [sys.executable, BUILDER_PATH],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        record("rebuild", False, "builder failed: {}".format(result.stderr.strip()[-300:]))
        return
    after = file_sha256(TRACE_PATH)
    record("rebuild", before == after,
           "regenerated digest {} {} committed digest".format(
               after, "equals" if before == after else "differs from"))


def print_report():
    """Print the check results and return the exit code."""
    failed = 0
    for name, ok, detail in CHECKS:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print("{} {} - {}".format(status, name, detail))
    print("{} checks, {} failed".format(len(CHECKS), failed))
    return 1 if failed else 0


def main():
    """Run the checks and exit."""
    if not os.path.exists(TRACE_PATH):
        print("FAIL missing trace file {}".format(TRACE_PATH))
        sys.exit(1)
    run_checks()
    if "--rebuild" in sys.argv:
        run_rebuild_check()
    sys.exit(print_report())


if __name__ == "__main__":
    main()
