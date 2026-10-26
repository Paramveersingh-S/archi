"""Tests for generic tickets pack (PACT r3-bundle-cluster-manager)."""
from __future__ import annotations

from pathlib import Path
import yaml

from archi.bundles import discovery
from archi.bundles.manifest import validate_pack


PACKS_DIR = Path("bundles/packs")
TICKETS_DIR = PACKS_DIR / "tickets"


def test_tickets_pack_valid():
    """Verify tickets pack passes manifest validation."""
    catalog = discovery.from_roots([PACKS_DIR], [])
    problems = validate_pack(TICKETS_DIR, packs=catalog.packs)
    assert problems == [], [str(p) for p in problems]
    assert "tickets" in catalog.packs


def test_tickets_pack_manifest_properties():
    catalog = discovery.from_roots([PACKS_DIR], [])
    pack = catalog.packs["tickets"]

    assert pack.name == "tickets"
    assert pack.kind == "context"
    assert pack.version == "0.1.0"
    assert pack.needs_credentials is False
    assert "CERN_JIRA_TOKEN" in pack.credentials
    assert "jira_issue" in pack.search_subtypes
    assert "document_chunk" in pack.search_subtypes


def test_tickets_template_configuration():
    template_path = TICKETS_DIR / "sources" / "tickets.yaml"
    assert template_path.is_file()

    data = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    assert "tickets" in data
    entry = data["tickets"]

    assert entry["module"] == "archi.sources.jira"
    assert entry["class"] == "JiraIssueAdapter"
    assert entry["source_class"] == "mutable_api"

    policy = entry["admission_policy"]
    assert policy["authority_scope"]["source_family"] == "tickets"
    assert policy["authority_scope"]["source_name"] == "tickets"

    nodes = [n["subtype"] for n in policy["output_signature"]["nodes"]]
    assert "jira_issue" in nodes
    assert "person" in nodes
    assert "document_chunk" in nodes

    params = entry["params"]
    assert "records_path" in params
    assert "meta_path" in params
    assert "base_url" in params


def test_tickets_invariants_structure():
    inv_path = TICKETS_DIR / "invariants.yaml"
    assert inv_path.is_file()

    data = yaml.safe_load(inv_path.read_text(encoding="utf-8"))
    invariants = data.get("invariants", [])
    assert len(invariants) >= 1

    cov = invariants[0]
    assert cov["name"] == "tickets_coverage"
    assert cov["severity"] == "error"
    assert "jira_issue" in cov["sql"]


def test_tickets_prompt_fragment():
    prompt_path = TICKETS_DIR / "prompt.md"
    assert prompt_path.is_file()

    text = prompt_path.read_text(encoding="utf-8")
    assert "Tickets" in text
    assert "jira_issue" in text
