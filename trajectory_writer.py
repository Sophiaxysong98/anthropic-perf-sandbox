"""Trajectory writer for benchmark results (GCS-only).

Writes benchmark results (cycle counts, speedups, kernel code, trust scores)
to GCS trajectory files. This is the standalone version for Docker/GitHub
without google3 dependencies.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

BASELINE_CYCLES = 147734

GCS_TRAJECTORY_BUCKET = (
    "gs://gemini-frontier-evals-anthropic-perf-inspect-logs/trajectories"
)


def _build_run_id(score_metadata: dict[str, Any]) -> str:
  model = os.environ.get(
      "MODEL_SHORT",
      os.environ.get("INSPECT_EVAL_MODEL", os.environ.get("MODEL", "unknown")),
  )
  model_short = model.rsplit("/", 1)[-1].lower().replace(".", "-")[:25]
  sandbox_mode = os.environ.get("SANDBOX_MODE", "unknown")
  round_id = os.environ.get("ROUND", "unknown")
  timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S")
  cycles = score_metadata.get("cycles", 0)
  speedup = score_metadata.get("speedup", 0.0)
  return (
      f"{round_id}/{model_short}/{sandbox_mode}"
      f"/{timestamp}_{cycles}cyc_{speedup:.1f}x"
  )


def _format_trajectory_summary(
    score_metadata: dict[str, Any],
    run_id: str,
) -> str:
  baseline = score_metadata.get("baseline_cycles", BASELINE_CYCLES)
  lines = [
      "=== Trajectory Summary ===",
      f"Run: {run_id}",
      f"Baseline: {baseline:,} cycles",
      "",
  ]

  trajectory = score_metadata.get("improvement_trajectory", [])
  if trajectory:
    for i, step in enumerate(trajectory, 1):
      if not isinstance(step, dict):
        continue
      cycles = step.get("cycles", 0)
      rate = step.get("improvement_rate", 0)
      action = step.get("action_summary", "")
      action_str = f" — {action}" if action else ""
      lines.append(
          f"Step {i}: {cycles:>8,} cycles ({rate:.1f}x improvement){action_str}"
      )
  else:
    lines.append("(No intermediate improvement steps recorded)")

  lines.append("")
  final_cycles = score_metadata.get("cycles", 0)
  final_speedup = score_metadata.get("speedup", 0.0)
  verification = score_metadata.get("verification_status", "N/A")
  lines.append(f"Final: {final_cycles:,} cycles ({final_speedup:.2f}x speedup)")
  lines.append(f"Verification: {verification}")

  best_intermediate = score_metadata.get("best_intermediate_cycles")
  if best_intermediate and best_intermediate != final_cycles:
    lines.append(f"Best intermediate: {best_intermediate:,} cycles")

  gaming_events = score_metadata.get("gaming_events", [])
  if gaming_events:
    lines.append(f"\nGaming events detected: {len(gaming_events)}")
    for evt in gaming_events[:5]:
      lines.append(
          f"  - {evt.get('type', '?')} at iteration {evt.get('iteration', '?')}"
      )

  return "\n".join(lines)


def _gcs_write(content: str, gcs_path: str) -> None:
  proc = subprocess.run(
      ["gcloud", "storage", "cp", "-", gcs_path],
      input=content,
      capture_output=True,
      text=True,
      timeout=30,
      check=False,
  )
  if proc.returncode != 0:
    logger.warning(
        "[TRAJECTORY] gcloud storage write failed for %s: %s",
        gcs_path,
        proc.stderr[:500],
    )


def _write_gcs_trajectory(
    score_metadata: dict[str, Any],
    run_id: str,
) -> dict[str, str]:
  gcs_dir = f"{GCS_TRAJECTORY_BUCKET}/{run_id}"

  trajectory = score_metadata.get("improvement_trajectory", [])
  jsonl_lines = []
  for step in trajectory:
    if isinstance(step, dict):
      jsonl_lines.append(json.dumps(step, default=str))

  summary_record = {
      "type": "final_score",
      "run_id": run_id,
      "cycles": score_metadata.get("cycles", 0),
      "speedup": score_metadata.get("speedup", 0.0),
      "best_intermediate_cycles": score_metadata.get(
          "best_intermediate_cycles", 0
      ),
      "verification_status": score_metadata.get("verification_status", ""),
      "gaming_events_count": len(score_metadata.get("gaming_events", [])),
      "all_observed_cycles": score_metadata.get("all_observed_cycles", []),
      "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
  }
  jsonl_lines.append(json.dumps(summary_record, default=str))
  jsonl_content = "\n".join(jsonl_lines)

  summary_text = _format_trajectory_summary(score_metadata, run_id)
  metadata_json = json.dumps(score_metadata, indent=2, default=str)

  urls = {}
  try:
    jsonl_path = f"{gcs_dir}/trajectory.jsonl"
    _gcs_write(jsonl_content, jsonl_path)
    urls["gcs_trajectory_jsonl"] = jsonl_path
    logger.info("[TRAJECTORY] JSONL written to %s", jsonl_path)

    summary_path = f"{gcs_dir}/trajectory_summary.txt"
    _gcs_write(summary_text, summary_path)
    urls["gcs_trajectory_summary"] = summary_path
    logger.info("[TRAJECTORY] Summary written to %s", summary_path)

    metadata_path = f"{gcs_dir}/score_metadata.json"
    _gcs_write(metadata_json, metadata_path)
    urls["gcs_score_metadata"] = metadata_path
    logger.info("[TRAJECTORY] Metadata written to %s", metadata_path)

  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.error("[TRAJECTORY] Failed to write GCS trajectory: %s", e)

  return urls


def write_trajectory(
    score_metadata: dict[str, Any],
    run_id: str | None = None,
) -> dict[str, str]:
  """Write benchmark trajectory data to GCS.

  Args:
    score_metadata: The metadata dict from the Inspect AI Score object.
    run_id: Optional identifier for this run. Auto-generated if not provided.

  Returns:
    Dict with GCS URLs/paths, or empty if writes failed.
  """
  if run_id is None:
    run_id = _build_run_id(score_metadata)

  return _write_gcs_trajectory(score_metadata, run_id)
