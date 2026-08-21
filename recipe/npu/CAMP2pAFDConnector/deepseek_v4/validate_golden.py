#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _request(
    endpoint: str,
    model: str,
    prompts: str | list[str],
) -> list[dict[str, Any]]:
    payload = {
        "model": model,
        "prompt": prompts,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 16,
        "seed": 1024,
        "stream": False,
        "return_token_ids": True,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {details}") from error
    choices = sorted(body["choices"], key=lambda choice: choice["index"])
    return [
        {
            "prompt_token_ids": choice["prompt_token_ids"],
            "token_ids": choice["token_ids"],
            "text": choice["text"],
            "finish_reason": choice["finish_reason"],
        }
        for choice in choices
    ]


def _batch_result_valid(
    result: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    return (
        result["prompt_token_ids"] == expected["prompt_token_ids"]
        and len(result["token_ids"]) == len(expected["token_ids"])
        and all(isinstance(token_id, int) for token_id in result["token_ids"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="dsv4-afd")
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 8, 32])
    parser.add_argument("--prompt-indices", type=int, nargs="*")
    parser.add_argument(
        "--require-batch-token-exact",
        action="store_true",
        help="Fail when any batched output differs from its golden token IDs.",
    )
    args = parser.parse_args()

    golden_report = json.loads(args.golden.read_text(encoding="utf-8"))
    golden = golden_report["golden"]
    prompt_indices = (
        args.prompt_indices
        if args.prompt_indices is not None
        else list(range(len(golden)))
    )
    if not prompt_indices:
        raise ValueError("at least one prompt index is required")
    invalid_indices = [index for index in prompt_indices if str(index) not in golden]
    if invalid_indices:
        raise ValueError(f"invalid golden prompt indices: {invalid_indices}")
    prompts = [golden[str(index)]["prompt"] for index in prompt_indices]
    records = []
    mismatches = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    for round_idx in range(args.rounds):
        for selected_idx, prompt in enumerate(prompts):
            prompt_idx = prompt_indices[selected_idx]
            result = _request(args.endpoint, args.model, prompt)[0]
            expected = golden[str(prompt_idx)]
            matched = (
                result["prompt_token_ids"] == expected["prompt_token_ids"]
                and result["token_ids"] == expected["token_ids"]
            )
            records.append(
                {
                    "kind": "golden",
                    "round": round_idx + 1,
                    "prompt_index": prompt_idx,
                    "matched": matched,
                    **result,
                }
            )
            if not matched:
                mismatches.append(
                    {"round": round_idx + 1, "prompt_index": prompt_idx}
                )
            print(
                f"round={round_idx + 1} prompt={prompt_idx:02d} "
                f"matched={matched} tokens={result['token_ids']}"
            )

    batch_records = []
    for batch_size in args.batch_sizes:
        batch_prompts = [prompts[index % len(prompts)] for index in range(batch_size)]
        results = _request(args.endpoint, args.model, batch_prompts)
        if len(results) != batch_size:
            raise AssertionError(
                f"batch {batch_size}: expected {batch_size} choices, got {len(results)}"
            )
        valid = []
        token_exact = []
        for index, result in enumerate(results):
            prompt_idx = prompt_indices[index % len(prompts)]
            expected = golden[str(prompt_idx)]
            item_valid = _batch_result_valid(result, expected)
            valid.append(item_valid)
            token_exact.append(result["token_ids"] == expected["token_ids"])
            if not item_valid:
                mismatches.append(
                    {"batch_size": batch_size, "choice": index, "kind": "invalid"}
                )
            elif args.require_batch_token_exact and not token_exact[-1]:
                mismatches.append(
                    {
                        "batch_size": batch_size,
                        "choice": index,
                        "kind": "token_mismatch",
                    }
                )
        batch_records.append(
            {
                "batch_size": batch_size,
                "valid": all(valid),
                "token_exact_count": sum(token_exact),
                "choice_count": len(results),
            }
        )
        print(
            f"batch={batch_size} valid={all(valid)} "
            f"token_exact={sum(token_exact)}/{batch_size}"
        )

    report = {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "endpoint": args.endpoint,
        "model": args.model,
        "golden_source": str(args.golden),
        "rounds": args.rounds,
        "require_batch_token_exact": args.require_batch_token_exact,
        "prompt_count": len(prompts),
        "prompt_indices": prompt_indices,
        "request_count": args.rounds * len(prompts),
        "batch_records": batch_records,
        "mismatches": mismatches,
        "passed": not mismatches,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"passed={report['passed']} output={args.output}")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
