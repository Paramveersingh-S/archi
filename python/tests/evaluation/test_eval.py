"""Tests for archi eval and evaluation runner (PACT r3-bundle-evals)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from archi.bundles.cli import build_parser
from archi.cli.eval import cmd_eval
from archi.evaluation.runner import (
    EvalResult,
    SmokeItem,
    evaluate_instance,
    format_report_md,
    load_smoke_set,
)
from archi.stack.instance import Instance, Ports


def test_load_smoke_set_valid(tmp_path):
    smoke_file = tmp_path / "smoke.jsonl"
    smoke_file.write_text(
        '{"id": "q1", "question": "What is Archi?", "expected_sources": ["docsite"], "expected_entities": ["Archi"]}\n'
        '{"id": "q2", "question": "Where is the meeting?", "expected_sources": ["indico"]}\n',
        encoding="utf-8",
    )
    items = load_smoke_set(smoke_file)
    assert len(items) == 2
    assert items[0].id == "q1"
    assert items[0].question == "What is Archi?"
    assert items[0].expected_sources == ["docsite"]
    assert items[0].expected_entities == ["Archi"]
    assert items[1].id == "q2"


def test_load_smoke_set_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_smoke_set(tmp_path / "nonexistent.jsonl")


def test_load_smoke_set_invalid_json(tmp_path):
    smoke_file = tmp_path / "smoke.jsonl"
    smoke_file.write_text('{"id": "q1", "question": "ok"}\n{invalid json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_smoke_set(smoke_file)


def test_load_smoke_set_empty(tmp_path):
    smoke_file = tmp_path / "smoke.jsonl"
    smoke_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="holds no evaluation questions"):
        load_smoke_set(smoke_file)


def _create_scratch_instance(tmp_path: Path) -> Instance:
    inst_dir = tmp_path / "scratch-inst"
    inst_dir.mkdir(parents=True, exist_ok=True)
    inst = Instance(
        dir=inst_dir,
        name="scratch-inst",
        bundles=["cern-team"],
        data_root=tmp_path / "data",
        runtime="docker",
        ports=Ports(graph_db=5432, mcp=8765, webui=8080),
    )
    inst.save()
    inst.env_path.write_text("OKG_CHAT_MCP_TOKEN=test-token\n", encoding="utf-8")
    return inst


def test_eval_run_on_scratch_instance_produces_report_md(tmp_path):
    """Proves acc.r3-bundle-evals.run: eval run on a scratch instance produces report.md."""
    instance = _create_scratch_instance(tmp_path)
    b_dir = tmp_path / "bundle"
    eval_dir = b_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (b_dir / "bundle.yaml").write_text(
        "name: test-bundle\nchat:\n  default_model: llama3.1:8b\n", encoding="utf-8"
    )
    (eval_dir / "smoke.jsonl").write_text(
        '{"id": "test-1", "question": "What does docsite say?", "expected_sources": ["docsite"], "expected_entities": ["documentation"]}\n'
        '{"id": "test-2", "question": "List meetings", "expected_sources": ["indico"], "expected_entities": ["meeting"]}\n',
        encoding="utf-8",
    )

    def fake_mcp(url, token):
        assert "8765" in url
        assert token == "test-token"
        tools = ["search", "inspect"]

        def query(question: str):
            if "docsite" in question:
                return {
                    "text": "The documentation covers installation and usage.",
                    "citations": ["docsite/getting-started.md"],
                }
            return {
                "text": "Upcoming team meeting on Indico schedule.",
                "citations": ["indico:12345"],
            }

        return tools, query

    report_path = instance.dir / "report.md"
    result = evaluate_instance(
        instance,
        bundle_dir=b_dir,
        bundle_name="test-bundle",
        output_path=report_path,
        mcp_runner=fake_mcp,
    )

    assert result.ok is True
    assert result.status == "ok"
    assert result.passed == 2
    assert result.total == 2
    assert report_path.is_file()

    content = report_path.read_text(encoding="utf-8")
    assert "# Smoke Evaluation Report: test-bundle" in content
    assert "llama3.1:8b" in content
    assert "search, inspect" in content
    assert "test-1" in content
    assert "PASS" in content


def test_zero_mcp_tools_fails_as_execution_failed(tmp_path):
    """Requirement: zero MCP tools fails as execution_failed."""
    instance = _create_scratch_instance(tmp_path)
    b_dir = tmp_path / "bundle"
    eval_dir = b_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "smoke.jsonl").write_text(
        '{"id": "q1", "question": "test", "expected_sources": []}\n', encoding="utf-8"
    )

    def zero_tools_mcp(url, token):
        return [], lambda q: {}

    report_path = instance.dir / "report.md"
    result = evaluate_instance(
        instance,
        bundle_dir=b_dir,
        output_path=report_path,
        mcp_runner=zero_tools_mcp,
    )

    assert result.ok is False
    assert result.status == "execution_failed"
    assert "zero MCP tools" in result.error
    assert report_path.is_file()
    assert "Execution Failed" in report_path.read_text(encoding="utf-8")


def test_cli_eval_command(tmp_path, monkeypatch, capsys):
    instance = _create_scratch_instance(tmp_path)
    b_dir = tmp_path / "bundle"
    eval_dir = b_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (b_dir / "bundle.yaml").write_text("name: test-bundle\n", encoding="utf-8")
    (eval_dir / "smoke.jsonl").write_text(
        '{"id": "q1", "question": "hi", "expected_sources": [], "expected_entities": ["hello"]}\n',
        encoding="utf-8",
    )

    def fake_mcp(url, token):
        return ["search"], lambda q: {"text": "hello world", "citations": []}

    monkeypatch.setattr("archi.evaluation.runner._default_mcp_session", fake_mcp)

    parser = build_parser(load_ops=False)
    report_file = tmp_path / "custom_report.md"
    args = parser.parse_args([
        "eval", str(instance.dir),
        "--bundle-root", str(tmp_path),
        "--bundle", "test-bundle",
        "--output", str(report_file),
        "--json",
    ])
    code = cmd_eval(args)
    assert code == 0
    assert report_file.is_file()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["ok"] is True
    assert data["status"] == "ok"
    assert data["total"] == 1
    assert data["passed"] == 1


def test_packaged_bundle_smoke_sets():
    """Verify packaged bundles have smoke sets with at least 10 questions."""
    cern_team_smoke = Path("bundles/cern-team/eval/smoke.jsonl")
    cms_physics_smoke = Path("bundles/cms-physics/eval/smoke.jsonl")

    assert cern_team_smoke.is_file()
    assert cms_physics_smoke.is_file()

    team_items = load_smoke_set(cern_team_smoke)
    physics_items = load_smoke_set(cms_physics_smoke)

    assert len(team_items) >= 10
    assert len(physics_items) >= 10

