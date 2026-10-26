"""Smoke evaluation runner for archi bundles and instance MCP endpoints.

Executes `eval/smoke.jsonl` against an instance's local MCP endpoint, checks
citation retrieval and answer completeness, and outputs `report.md`.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import yaml

from archi.bundles import discovery
from archi.stack.instance import Instance, InstanceError, read_env


@dataclass(frozen=True)
class SmokeItem:
    """One smoke question from eval/smoke.jsonl."""

    id: str
    question: str
    expected_sources: list[str] = field(default_factory=list)
    expected_entities: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class EvalResult:
    """Outcome of evaluating an instance against a bundle's smoke set."""

    ok: bool
    status: str  # "ok", "execution_failed", "evaluation_failed"
    instance: str
    bundle: str
    model: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    tools: list[str] = field(default_factory=list)
    report_path: Path = field(default_factory=lambda: Path("report.md"))
    items: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["report_path"] = str(self.report_path)
        return data


def load_smoke_set(path: Path) -> list[SmokeItem]:
    """Read a bundle's smoke.jsonl question set."""
    if not path.is_file():
        raise FileNotFoundError(f"Smoke evaluation set not found: {path}")

    items: list[SmokeItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc

            items.append(
                SmokeItem(
                    id=str(data.get("id", f"q{line_no}")),
                    question=str(data["question"]),
                    expected_sources=list(data.get("expected_sources", [])),
                    expected_entities=list(data.get("expected_entities", [])),
                    description=str(data.get("description", "")),
                )
            )

    if not items:
        raise ValueError(f"{path}: holds no evaluation questions")
    return items


def _default_mcp_session(url: str, token: str) -> tuple[list[str], Callable[[str], dict[str, Any]]]:
    """Connect to live MCP endpoint using official MCP client session."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def _inspect() -> tuple[list[str], dict[str, Any]]:
        headers = {"Authorization": f"Bearer {token}"}
        async with streamablehttp_client(url, headers=headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = [t.name for t in (await session.list_tools()).tools]
                return tools, {}

    tools, _ = asyncio.run(_inspect())

    def _query(question: str) -> dict[str, Any]:
        async def _run() -> dict[str, Any]:
            headers = {"Authorization": f"Bearer {token}"}
            async with streamablehttp_client(url, headers=headers) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    # Query search or graph inspection tool
                    call_res = await session.call_tool("search", {"query": question})
                    text = "".join(getattr(c, "text", "") for c in call_res.content)
                    return {"text": text, "citations": []}

        return asyncio.run(_run())

    return tools, _query


def format_report_md(result: EvalResult) -> str:
    """Generate Markdown evaluation report."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        f"# Smoke Evaluation Report: {result.bundle}",
        "",
        f"- **Instance**: `{result.instance}`",
        f"- **Bundle**: `{result.bundle}`",
        f"- **Model**: `{result.model}`",
        f"- **Date**: {timestamp}",
        f"- **Status**: `{result.status}`",
        f"- **MCP Tools**: `{', '.join(result.tools) if result.tools else 'none'}`",
        f"- **Summary**: {result.passed}/{result.total} passed ({result.passed * 100 // max(1, result.total)}%)",
        "",
    ]

    if result.error:
        lines.extend([
            f"> [!CAUTION]",
            f"> Execution Failed: {result.error}",
            "",
        ])

    lines.extend([
        "## Results Table",
        "",
        "| ID | Question | Status | Citations Matched | Entities Matched | Details |",
        "|---|---|---|---|---|---|",
    ])

    for item in result.items:
        qid = item.get("id", "")
        q = item.get("question", "").replace("|", "\\|")
        status = item.get("status", "fail")
        badge = "PASS" if status == "pass" else "FAIL"
        c_matched = f"{len(item.get('matched_sources', []))}/{len(item.get('expected_sources', []))}"
        e_matched = f"{len(item.get('matched_entities', []))}/{len(item.get('expected_entities', []))}"
        details = item.get("details", "").replace("|", "\\|")
        lines.append(f"| `{qid}` | {q} | **{badge}** | {c_matched} | {e_matched} | {details} |")

    lines.append("")
    return "\n".join(lines)


def evaluate_instance(
    instance: Instance,
    *,
    bundle_name: Optional[str] = None,
    bundle_dir: Optional[Path] = None,
    model: Optional[str] = None,
    output_path: Optional[Path] = None,
    mcp_runner: Optional[Callable[[str, str], tuple[list[str], Callable[[str], dict[str, Any]]]]] = None,
    pack_roots: Sequence[Path] = (),
    bundle_roots: Sequence[Path] = (),
) -> EvalResult:
    """Run smoke evaluations against an instance's MCP endpoint."""
    # 1. Resolve bundle and bundle directory
    chosen_bundle = bundle_name or (instance.bundles[0] if instance.bundles else "cern-team")
    catalog = discovery.discover(pack_roots, bundle_roots)

    b_dir = bundle_dir
    if b_dir is None:
        if chosen_bundle in catalog.bundles:
            b_dir = catalog.bundles[chosen_bundle].dir
        else:
            candidates = [Path("bundles") / chosen_bundle]
            for root in bundle_roots:
                candidates.append(root / chosen_bundle)
                if root.is_dir():
                    for sub in root.glob("*"):
                        if sub.is_dir():
                            candidates.append(sub)
            for c in candidates:
                if c.is_dir() and (c / "eval" / "smoke.jsonl").is_file():
                    b_dir = c
                    break
            if b_dir is None:
                b_dir = instance.dir

    smoke_file = b_dir / "eval" / "smoke.jsonl"
    report_file = output_path or (instance.dir / "report.md")

    # 2. Resolve default model from bundle manifest or fallback
    eval_model = model
    if not eval_model:
        bundle_yaml_path = b_dir / "bundle.yaml"
        if bundle_yaml_path.is_file():
            try:
                b_meta = yaml.safe_load(bundle_yaml_path.read_text(encoding="utf-8")) or {}
                eval_model = (b_meta.get("chat") or {}).get("default_model")
            except Exception:
                pass
    if not eval_model:
        eval_model = "llama3.1:8b"

    # 3. Read secrets and locate MCP endpoint
    try:
        secrets = read_env(instance.env_path)
    except InstanceError:
        secrets = {}
    token = secrets.get("OKG_CHAT_MCP_TOKEN", "")
    url = f"http://127.0.0.1:{instance.ports.mcp}/mcp"

    # 4. Connect to MCP endpoint
    runner_fn = mcp_runner or _default_mcp_session
    try:
        tools, query_fn = runner_fn(url, token)
    except Exception as exc:
        result = EvalResult(
            ok=False,
            status="execution_failed",
            instance=str(instance.dir),
            bundle=chosen_bundle,
            model=eval_model,
            report_path=report_file,
            error=f"could not connect to MCP at {url}: {exc}",
        )
        report_file.write_text(format_report_md(result), encoding="utf-8")
        return result

    # Check zero MCP tools rule: zero MCP tools fails as execution_failed
    if not tools:
        result = EvalResult(
            ok=False,
            status="execution_failed",
            instance=str(instance.dir),
            bundle=chosen_bundle,
            model=eval_model,
            tools=[],
            report_path=report_file,
            error="zero MCP tools served; instance MCP is not operational",
        )
        report_file.write_text(format_report_md(result), encoding="utf-8")
        return result

    # 5. Load smoke questions
    try:
        items = load_smoke_set(smoke_file)
    except Exception as exc:
        result = EvalResult(
            ok=False,
            status="execution_failed",
            instance=str(instance.dir),
            bundle=chosen_bundle,
            model=eval_model,
            tools=tools,
            report_path=report_file,
            error=f"failed to load smoke set from {smoke_file}: {exc}",
        )
        report_file.write_text(format_report_md(result), encoding="utf-8")
        return result

    # 6. Execute smoke questions
    item_results: list[dict[str, Any]] = []
    passed_count = 0

    for item in items:
        try:
            resp = query_fn(item.question)
            text = str(resp.get("text", ""))
            citations = [str(c) for c in resp.get("citations", [])]

            matched_sources = [
                s for s in item.expected_sources
                if any(s.lower() in c.lower() for c in citations) or s.lower() in text.lower()
            ]
            matched_entities = [
                e for e in item.expected_entities
                if e.lower() in text.lower()
            ]

            sources_ok = not item.expected_sources or bool(matched_sources)
            entities_ok = not item.expected_entities or (len(matched_entities) >= len(item.expected_entities) // 2 + 1)
            is_pass = sources_ok and entities_ok

            if is_pass:
                passed_count += 1
                details = "criteria satisfied"
            else:
                missing = []
                if not sources_ok:
                    missing.append(f"sources {item.expected_sources} not cited")
                if not entities_ok:
                    missing.append(f"entities {item.expected_entities} missing")
                details = "; ".join(missing)

            item_results.append({
                "id": item.id,
                "question": item.question,
                "status": "pass" if is_pass else "fail",
                "expected_sources": item.expected_sources,
                "matched_sources": matched_sources,
                "expected_entities": item.expected_entities,
                "matched_entities": matched_entities,
                "details": details,
            })
        except Exception as exc:
            item_results.append({
                "id": item.id,
                "question": item.question,
                "status": "fail",
                "expected_sources": item.expected_sources,
                "matched_sources": [],
                "expected_entities": item.expected_entities,
                "matched_entities": [],
                "details": f"query failed: {exc}",
            })

    all_passed = (passed_count == len(items))
    result = EvalResult(
        ok=all_passed,
        status="ok" if all_passed else "evaluation_failed",
        instance=str(instance.dir),
        bundle=chosen_bundle,
        model=eval_model,
        total=len(items),
        passed=passed_count,
        failed=len(items) - passed_count,
        tools=tools,
        report_path=report_file,
        items=item_results,
    )

    report_content = format_report_md(result)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report_content, encoding="utf-8")
    return result
