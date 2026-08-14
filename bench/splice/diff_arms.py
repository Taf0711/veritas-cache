#!/usr/bin/env python3
"""Compare the three arms of the Phase 6.3 control-loop experiment.

Read each arm report and cache database. Print a compact table per task.
Signal: iteration counts, escalation events, abort status and reason.

Usage: diff_arms.py [run_dir]
Default run_dir: bench/splice/runs/latest
"""
import json
import os
import sqlite3
import sys

ARMS = ["baseline", "static", "ld3"]


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
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "shadow_log" in tables:
            db_stats["shadow_decisions"] = dict(
                conn.execute(
                    "SELECT decision, COUNT(*) FROM shadow_log GROUP BY decision"
                ).fetchall()
            )
        if "entries" in tables:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM entries"
            ).fetchone()
            db_stats["entries"] = row[0]
            db_stats["hit_count_sum"] = row[1]
        conn.close()
    return report, db_stats


def task_rows(report):
    """Extract the flat signal columns for each task in one arm report."""
    rows = []
    for task in report.get("tasks", []):
        agent = task.get("agent", {})
        scoring = task.get("report", {})
        stdout = agent.get("stdout", "")
        stages = {
            stage["name"]: stage.get("iteration", 0)
            for stage in agent.get("stages", [])
            if stage.get("name")
        }
        rows.append(
            {
                "task": task.get("taskId", "?"),
                "status": scoring.get("status", "?"),
                "error": scoring.get("error", ""),
                "exit": agent.get("exitCode", "?"),
                "escalations": stdout.count("[escalation]"),
                "latency_ms": task.get("latencyMs", agent.get("latencyMs", "?")),
                "stage_iterations": stages,
            }
        )
    return rows


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
        if db_stats:
            print(f"  db: {db_stats}")
        rows = task_rows(report)
        if not rows:
            print("  no tasks in report")
        for row in rows:
            print(
                "  task {task}: status={status} exit={exit} "
                "escalations={escalations} latencyMs={latency_ms}".format(**row)
            )
            if row["error"]:
                print(f"    error: {row['error']}")
            for name, iteration in sorted(row["stage_iterations"].items()):
                print(f"    stage {name}: iterations={iteration}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
