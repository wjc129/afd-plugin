#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

FATAL_MARKERS = (
    "AFD NPU FFN worker loop failed",
    "error code is 507015",
    "HCCL timeout",
    "Exception in thread",
)
U2_STAGE_ZERO_MARKER = "key=((0,"
U2_STAGE_ONE_MARKER = "), (1,"


def _is_healthy(endpoint: str) -> bool:
    try:
        with urllib.request.urlopen(endpoint, timeout=5) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _fatal_markers(text: str) -> list[str]:
    return [marker for marker in FATAL_MARKERS if marker in text]


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--health-endpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runtime_path = args.log_dir / "runtime.env"
    attention_path = args.log_dir / "attention.log"
    ffn_path = args.log_dir / "ffn.log"
    runtime_text = _read_text(runtime_path)
    attention_text = _read_text(attention_path)
    ffn_text = _read_text(ffn_path)

    stage_lines = [
        line
        for line in attention_text.splitlines()
        if U2_STAGE_ZERO_MARKER in line and U2_STAGE_ONE_MARKER in line
    ]
    role_fatals = {
        "attention": _fatal_markers(attention_text),
        "ffn": _fatal_markers(ffn_text),
    }
    checks = {
        "runtime_manifest_present": runtime_path.is_file(),
        "attention_log_present": attention_path.is_file(),
        "ffn_log_present": ffn_path.is_file(),
        "runtime_declares_eager_u2": (
            "execution_mode=eager" in runtime_text and "u_batches=2" in runtime_text
        ),
        "runtime_declares_a8f8": (
            "attention_ranks=8" in runtime_text and "ffn_ranks=8" in runtime_text
        ),
        "decode_health": _is_healthy(args.health_endpoint),
        "two_stage_runtime_evidence": bool(stage_lines),
        "no_attention_fatal": not role_fatals["attention"],
        "no_ffn_fatal": not role_fatals["ffn"],
        "ffn_has_no_mooncake": "Mooncake" not in ffn_text,
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "health_endpoint": args.health_endpoint,
        "log_dir": str(args.log_dir),
        "runtime": runtime_text.splitlines(),
        "two_stage_evidence_lines": stage_lines[-8:],
        "fatal_markers": role_fatals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
