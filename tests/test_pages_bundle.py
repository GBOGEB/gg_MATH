"""Contract tests for the GitHub Pages source and its two publishers.

The live site (https://gbogeb.github.io/gg_MATH/) is the github-pages Jekyll
build of the repository root: README.md renders as the entry page and the
title comes from the github-pages defaults (repository name + description).
These tests pin the source that build consumes and the deploy-contract shape
of both Pages workflows. They need only pytest and PyYAML.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
JEKYLL = WORKFLOWS / "jekyll-gh-pages.yml"
RELEASE = WORKFLOWS / "release-infrastructure-v1.yml"
SHA_PIN = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
HUB_DEPLOY = "GBOGEB/pipeline-automation-hub/.github/workflows/jade-reusable-deploy.yml@"
JEKYLL_IMAGE = "ghcr.io/actions/jekyll-build-pages@sha256:"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _uses(payload: dict) -> list[str]:
    found = []
    for job in payload["jobs"].values():
        if "uses" in job:
            found.append(job["uses"])
        for step in job.get("steps") or []:
            if "uses" in step:
                found.append(step["uses"])
    return found


def test_entry_page_source_is_the_readme() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("# gg_MATH\n")
    # No index.* at the root and no Jekyll config: README.md stays the entry
    # page and the github-pages defaults keep the live <title>.
    assert not list(ROOT.glob("index.*"))
    for name in ("_config.yml", "_config.yaml", "Gemfile"):
        assert not (ROOT / name).exists(), name


def test_pages_linked_from_the_entry_page_exist() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    local = {
        target
        for target in re.findall(r"\]\(([^)#\s]+)\)", text)
        if not re.match(r"^[a-z]+:", target)
    }
    assert {"LICENSE", "release/README.md"} <= local
    missing = sorted(target for target in local if not (ROOT / target).exists())
    assert missing == []


def test_jekyll_publisher_builds_once_and_promotes_by_digest() -> None:
    payload = _load(JEKYLL)
    jobs = payload["jobs"]
    assert set(jobs) == {"tests", "build", "deploy"}
    build_steps = jobs["build"]["steps"]
    script = "\n".join(step.get("run", "") for step in build_steps)
    assert JEKYLL_IMAGE in script and "INPUT_SOURCE=./" in script
    assert "jade.devops.artifact_manifest.v1" in script
    assert any(step.get("env", {}).get("INPUT_TOKEN") == "${{ github.token }}" for step in build_steps)
    deploy = jobs["deploy"]
    assert deploy["uses"].startswith(HUB_DEPLOY)
    assert deploy["needs"] == ["tests", "build"]
    assert "pull_request" in deploy["if"] and "refs/heads/main" in deploy["if"]
    assert deploy["with"]["environment"] == "production"
    assert deploy["with"]["manifest-sha256"] == "${{ needs.build.outputs.manifest-sha256 }}"
    assert "concurrency" not in payload


def test_release_deploy_keeps_its_id_dependants_and_dispatch_gate() -> None:
    jobs = _load(RELEASE)["jobs"]
    deploy = jobs["deploy-pages"]
    assert deploy["uses"].startswith(HUB_DEPLOY)
    assert deploy["if"] == (
        "github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'"
        " && needs.release-gate.outputs.publish_pages == 'true'"
    )
    assert "deploy-pages" in jobs["pages-readback"]["needs"]
    assert "deploy-pages" in jobs["github-release"]["needs"]
    gate = jobs["release-gate"]["outputs"]
    assert {"pages_artifact", "manifest_sha256", "tests_ran"} <= set(gate)
    assert deploy["with"]["artifact-name"] == "${{ needs.release-gate.outputs.pages_artifact }}"
    assert deploy["with"]["manifest-sha256"] == "${{ needs.release-gate.outputs.manifest_sha256 }}"


def test_every_action_in_both_publishers_is_sha_pinned() -> None:
    for path in (JEKYLL, RELEASE):
        uses = _uses(_load(path))
        assert uses, path.name
        unpinned = [item for item in uses if not SHA_PIN.match(item)]
        assert unpinned == [], f"{path.name}: {unpinned}"
