"""`archi eval`: run bundle smoke evaluations against instance MCP endpoint.

    archi eval [INSTANCE] [--bundle BUNDLE] [--model MODEL] [--output REPORT] [--json]

Executes the bundle's smoke question set (`eval/smoke.jsonl`) against the instance's
MCP server, assesses retrieval and synthesis completeness, and generates `report.md`.
Zero MCP tools fails immediately with status `execution_failed` (exit code 2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from archi.evaluation.runner import evaluate_instance
from archi.stack.instance import Instance, InstanceError


def cmd_eval(args: argparse.Namespace) -> int:
    try:
        instance = Instance.load(args.instance)
    except InstanceError as exc:
        if args.json:
            print(json.dumps({"ok": False, "instance": str(args.instance), "error": str(exc), "status": "execution_failed"}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2

    pack_roots = getattr(args, "pack_root", [])
    bundle_roots = getattr(args, "bundle_root", [])

    result = evaluate_instance(
        instance,
        bundle_name=args.bundle,
        model=args.model,
        output_path=args.output,
        pack_roots=pack_roots,
        bundle_roots=bundle_roots,
    )

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(f"evaluating bundle '{result.bundle}' against MCP on {instance.dir}...")
        if result.status == "execution_failed":
            print(f"FAIL execution_failed: {result.error}", file=sys.stderr)
            print(f"wrote report to {result.report_path}")
            return 2

        for item in result.items:
            mark = "ok  " if item["status"] == "pass" else "FAIL"
            print(f"{mark}  {item['id']}: {item['question']}")
            if item["status"] != "pass":
                print(f"      reason: {item['details']}")
        print(f"\n{result.passed}/{result.total} questions passed. Report written to {result.report_path}")

    return 0 if result.ok else 1


def add_parser(sub, roots) -> None:
    p = sub.add_parser("eval", help="run bundle smoke evaluations against the instance MCP endpoint")
    p.add_argument("instance", nargs="?", type=Path, default=Path("."), help="instance directory (default: .)")
    p.add_argument("--bundle", default=None, help="bundle to evaluate (default: instance's primary bundle)")
    p.add_argument("--model", default=None, help="model identifier to record in report")
    p.add_argument("--output", type=Path, default=None, help="path to write report.md (default: <instance>/report.md)")
    p.add_argument("--json", action="store_true", help="output JSON result object for CI/agents")
    roots(p)
    p.set_defaults(func=cmd_eval)
