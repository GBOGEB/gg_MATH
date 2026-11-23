"""Contract test for the Semgrep code-scanning workflow (review r4367330797).

The SARIF upload must be attributed to the commit that was scanned. For pull
requests from forks, github/codeql-action/upload-sarif ignores its ``ref`` and
``sha`` inputs and binds results to the event's merge ref/SHA, so the job has
to scan that merge commit in the fork case. This test evaluates the
workflow's ``${{ }}`` expressions for push, same-repository PR and fork PR
events and models that upload-sarif behaviour. Needs pytest and PyYAML.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEMGREP = ROOT / ".github" / "workflows" / "semgrep.yml"
REPO = "GBOGEB/gg_MATH"
HEAD = "1" * 40
MERGE = "2" * 40
MAIN = "3" * 40

TOKEN = re.compile(r"\s*(?:(?P<str>'(?:[^']|'')*')|(?P<op>==|!=|&&|\|\||[(),])|(?P<name>[A-Za-z_][\w.-]*))")


def _tokens(text: str) -> list[tuple[str, str]]:
    out, pos = [], 0
    text = text.strip()
    while pos < len(text):
        match = TOKEN.match(text, pos)
        if not match or match.end() == pos:
            raise ValueError(f"unsupported expression syntax at {text[pos:]!r}")
        kind = match.lastgroup
        out.append((kind, match.group(kind)))
        pos = match.end()
    return out


def evaluate(expression: str, context: dict):
    """Evaluate the GitHub Actions expression subset this workflow uses."""
    tokens = _tokens(expression)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else (None, None)

    def take(value=None):
        nonlocal pos
        token = tokens[pos]
        if value is not None and token[1] != value:
            raise ValueError(f"expected {value!r}, got {token[1]!r}")
        pos += 1
        return token

    def primary():
        kind, value = take()
        if kind == "str":
            return value[1:-1].replace("''", "'")
        if value == "(":
            result = or_expr()
            take(")")
            return result
        if kind == "name" and peek()[1] == "(":
            take("(")
            args = [or_expr()]
            while peek()[1] == ",":
                take(",")
                args.append(or_expr())
            take(")")
            if value != "format":
                raise ValueError(f"unsupported function {value}")
            return re.sub(r"\{(\d+)\}", lambda m: "" if args[int(m.group(1)) + 1] is None else str(args[int(m.group(1)) + 1]), args[0])
        node = context
        for part in value.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        return node

    def cmp_expr():
        left = primary()
        while peek()[1] in ("==", "!="):
            op = take()[1]
            right = primary()
            same = (str(left).lower() == str(right).lower()) if left is not None and right is not None else left is right
            left = same if op == "==" else not same
        return left

    def and_expr():
        left = cmp_expr()
        while peek()[1] == "&&":
            take()
            right = cmp_expr()
            left = right if left else left
        return left

    def or_expr():
        left = and_expr()
        while peek()[1] == "||":
            take()
            right = and_expr()
            left = left if left else right
        return left

    result = or_expr()
    if pos != len(tokens):
        raise ValueError(f"trailing tokens in {expression!r}")
    return result


def _expr(value: str) -> str:
    match = re.fullmatch(r"\$\{\{(.*)\}\}", value.strip(), re.S)
    assert match, f"not a single expression: {value!r}"
    return match.group(1)


def _context(event: str) -> dict:
    if event == "push":
        return {"github": {"event_name": "push", "repository": REPO, "ref": "refs/heads/main", "sha": MAIN,
                           "event": {}}}
    head_repo = REPO if event == "same_repo_pr" else "someone/gg_MATH"
    return {"github": {"event_name": "pull_request", "repository": REPO, "ref": "refs/pull/7/merge", "sha": MERGE,
                       "event": {"number": 7, "pull_request": {"head": {"sha": HEAD, "repo": {"full_name": head_repo}}}}}}


def _steps() -> tuple[dict, dict]:
    job = yaml.safe_load(SEMGREP.read_text(encoding="utf-8"))["jobs"]["semgrep"]
    checkout = next(s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout@"))
    upload = next(s for s in job["steps"] if str(s.get("uses", "")).startswith("github/codeql-action/upload-sarif@"))
    return checkout, upload


@pytest.mark.parametrize("event, scanned_sha, bound_ref", [
    ("push", MAIN, "refs/heads/main"),
    ("same_repo_pr", HEAD, "refs/pull/7/head"),
    ("fork_pr", MERGE, "refs/pull/7/merge"),
])
def test_sarif_is_attributed_to_the_scanned_commit(event, scanned_sha, bound_ref):
    checkout, upload = _steps()
    context = _context(event)
    scanned = evaluate(_expr(checkout["with"]["ref"]), context)
    ref = evaluate(_expr(upload["with"]["ref"]), context)
    sha = evaluate(_expr(upload["with"]["sha"]), context)
    if event == "fork_pr":
        # upload-sarif ignores ref/sha for fork PRs and uses the merge ref/SHA.
        ref, sha = context["github"]["ref"], context["github"]["sha"]
    assert scanned == sha == scanned_sha
    assert ref == bound_ref


def test_expression_evaluator_semantics():
    context = _context("fork_pr")
    assert evaluate("github.event.pull_request.head.sha || github.sha", context) == HEAD
    assert evaluate("github.event.pull_request.head.sha || github.sha", _context("push")) == MAIN
    assert evaluate("github.event_name == 'pull_request' && format('refs/pull/{0}/head', github.event.number) || github.ref", context) == "refs/pull/7/head"
    assert evaluate("github.event.pull_request.head.repo.full_name == github.repository", context) is False
