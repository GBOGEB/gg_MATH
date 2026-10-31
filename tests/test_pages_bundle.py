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
    assert set(jobs) == {"ci", "deploy"}
    ci = jobs["ci"]["with"]
    assert ci["artifact-path"] == "_site"
    assert int(ci["min-tests"]) >= 3
    build = ci["build-command"]
    assert JEKYLL_IMAGE in build and "INPUT_SOURCE=./" in build
    deploy = jobs["deploy"]
    assert deploy["uses"].startswith(HUB_DEPLOY)
    assert deploy["needs"] == "ci"
    assert "pull_request" in deploy["if"] and "refs/heads/main" in deploy["if"]
    assert deploy["with"]["environment"] == "production"
    assert "concurrency" not in payload


def test_every_action_in_the_jekyll_publisher_is_sha_pinned() -> None:
    for path in (JEKYLL,):
        uses = _uses(_load(path))
        assert uses, path.name
        unpinned = [item for item in uses if not SHA_PIN.match(item)]
        assert unpinned == [], f"{path.name}: {unpinned}"
