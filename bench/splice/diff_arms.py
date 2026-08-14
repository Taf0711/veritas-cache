#!/usr/bin/env python3
"""Compare the arms of the Phase 6.3 control-loop experiment.

Read each arm report and cache database. Print a compact table per task.
Signal: iteration counts, escalation events, abort status and reason.
Cost signal: splice-reported tokens and proxy-observed upstream tokens.
Functionality signal: pass and fail counts across the tasks.

Usage: diff_arms.py [run_dir]
Default run_dir: bench/splice/runs/latest
"""
import json
import os
import re
import sqlite3
import sys

ARMS = ["baseline", "exact", "static", "ld3"]


def load_arm(run_dir, arm):
    """Load the report and cache stats for one arm. Return None when missing."""
    arm_dir = os.path.join(run_dir, arm)
    report_path = os.path.join(arm_dir, "agent-eval-report.json")
    if not os.path.exists(report_path):
        return None
    with open(report_path) as handle:
        report = json.load(handle)
    db_path = os.path.join(arm_dir, "cache.db")
    db_stats = {}
    upstream_prompt = 0
    upstream_completion = 0
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        shadow_rows = 0
        if "shadow_log" in tables:
            db_stats["shadow_decisions"] = dict(
                conn.execute(
                    "SELECT decision, COUNT(*) FROM shadow_log GROUP BY decision"
                ).fetchall()
            )
            shadow_rows = sum(db_stats["shadow_decisions"].values())
        if "entries" in tables:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM entries"
            ).fetchone()
            db_stats["entries"] = row[0]
            db_stats["hit_count_sum"] = row[1]
        # Upstream tokens seen by the proxy. Sum each upstream response once.
        # Shadow mode stores every fresh response in shadow_log and in entries.
        # Count shadow_log only there. Live arms store miss responses in entries.
        if shadow_rows:
            source = conn.execute(
                "SELECT fresh_json FROM shadow_log WHERE fresh_json IS NOT NULL"
            )
        elif "entries" in tables:
            source = conn.execute("SELECT response_json FROM entries")
        else:
            source = []
        for (body,) in source:
            prompt, completion = usage_tokens(body)
            upstream_prompt += prompt
            upstream_completion += completion
        conn.close()
    db_stats["upstream_prompt_tokens"] = upstream_prompt
    db_stats["upstream_completion_tokens"] = upstream_completion
    return report, db_stats


def usage_tokens(body):
    """Read prompt and completion tokens from one stored response body."""
    try:
        usage = json.loads(body).get("usage") or {}
        return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    except (ValueError, TypeError, AttributeError):
        return 0, 0


def abort_reason(report):
    """Pull the abort reason from the failure record or the stream-json stdout."""
    failures = report.get("failures") or []
    text = "\n".join(f.get("message", "") for f in failures)
    benchmark = report.get("benchmark") or {}
    for task in benchmark.get("tasks") or []:
        text += "\n" + (task.get("agent") or {}).get("stdout", "")
    match = re.search(r"abort_\w+: [^\n.]*", text)
    return match.group(0) if match else ""


def task_rows(report):
    """Extract the flat signal columns for each task in one arm report."""
    benchmark = report.get("benchmark") or {}
    rows = []
    for task in benchmark.get("tasks") or []:
        agent = task.get("agent", {})
        scoring = task.get("report", {})
        stdout = agent.get("stdout", "")
        stages = {
            stage["name"]: stage.get("iteration", 0)
            for stage in agent.get("stages", [])
            if stage.get("name")
        }
        checks = [
            (r.get("id", "?"), r.get("status", "?"))
            for r in scoring.get("results", [])
            if r.get("kind") == "command"
        ]
        rows.append(
            {
                "task": task.get("taskId", "?"),
                "status": scoring.get("status", "?"),
                "error": scoring.get("error", ""),
                "exit": agent.get("exitCode", "?"),
                "escalations": stdout.count("[escalation]"),
                "latency_ms": task.get("latencyMs", agent.get("latencyMs", "?")),
                "stage_iterations": stages,
                "checks": checks,
            }
        )
    return rows


def splice_tokens(report):
    """Sum the usage samples that splice reported across all tasks."""
    benchmark = report.get("benchmark") or {}
    prompt = 0
    completion = 0
    for task in benchmark.get("tasks") or []:
        agent = task.get("agent") or {}
        for sample in agent.get("usageSamples") or []:
            prompt += sample.get("inputTokens") or 0
            completion += sample.get("outputTokens") or 0
    return prompt, completion


def status_counts(rows):
    """Count the report status values across tasks."""
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "runs", "latest"
    )
    run_dir = os.path.realpath(run_dir)
    if not os.path.isdir(run_dir):
        print(f"FAIL run dir not found: {run_dir}")
        return 1

    print(f"run dir: {run_dir}")
    for arm in ARMS:
        loaded = load_arm(run_dir, arm)
        if loaded is None:
            print(f"\narm {arm}: no report")
            continue
        report, db_stats = loaded
        print(f"\narm {arm}")
        reason = abort_reason(report)
        if reason:
            print(f"  abort: {reason}")
        if db_stats:
            print(f"  db: {db_stats}")
        rows = task_rows(report)
        if not rows:
            print("  no tasks in report")
        print(f"  tasks: {status_counts(rows)}")
        in_tok, out_tok = splice_tokens(report)
        print(f"  splice tokens: input={in_tok} output={out_tok}")
        print(
            "  upstream tokens: prompt={} completion={}".format(
                db_stats.get("upstream_prompt_tokens", 0),
                db_stats.get("upstream_completion_tokens", 0),
            )
        )
        if arm == "ld3":
            # ld3 declines to store some misses. Those tokens do not appear here.
            print("  note: ld3 upstream tokens count stored responses only")
        for row in rows:
            print(
                "  task {task}: status={status} exit={exit} "
                "escalations={escalations} latencyMs={latency_ms}".format(**row)
            )
            if row["error"]:
                print(f"    error: {row['error']}")
            for name, iteration in sorted(row["stage_iterations"].items()):
                print(f"    stage {name}: iterations={iteration}")
            for check_id, check_status in row["checks"]:
                print(f"    check {check_id}: {check_status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
