"""Execute and account for every gg_MATH mission proof (release gate R3).

``release/MISSION_PROOFS_v1.yaml`` declares the proofs. ``run`` executes them
on the exact source SHA and writes ``release/mission-proofs.json``; ``check``
only performs the workflow accounting. Both fail closed.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CENSUS = Path("release/MISSION_PROOFS_v1.yaml")
WORKFLOWS = Path(".github/workflows")
RELEASE_GATE_WORKFLOW = "release-infrastructure-v1.yml"
OUT_ROOT = Path("artifacts/mission_proofs")
SCHEMA = "gg-math-mission-proof-run/v1"
LOCAL_MODULE_ROOTS = ("kernels", "mission", "release", "research", "scripts", "tests", "tools")
PYTHON_CALL = re.compile(r"(?:^|[\s;&|(])python3?(?=\s)")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
STEP_TIMEOUT_SECONDS = 900


def entrypoint(args: list[str]) -> dict | None:
    """Classify the repository entrypoint of one ``python <args>`` invocation.

    Returns ``{"target": path-or-glob, "kind": "script"|"module"|"unittest"}``,
    ``{"kind": "inline"}`` for ``python -`` heredocs, or ``None`` when the
    invocation runs no repository code (pip, json.tool, coverage json, ...).
    """
    if not args:
        return None
    if args[0] == "-":
        return {"kind": "inline"}
    index = 0
    while index < len(args):
        token = args[index]
        if token == "-m" and index + 1 < len(args):
            module = args[index + 1]
            rest = args[index + 2:]
            if module == "unittest":
                start = rest.index("-s") + 1 if "-s" in rest else None
                pattern = rest[rest.index("-p") + 1] if "-p" in rest and rest.index("-p") + 1 < len(rest) else "test*.py"
                directory = rest[start] if start is not None and start < len(rest) else "."
                return {"target": f"{directory.rstrip('/')}/{pattern}", "kind": "unittest"}
            if module == "coverage":
                if rest[:1] == ["run"]:
                    return entrypoint(rest[1:])
                return None
            if module == "py_compile":
                paths = [item for item in rest if item.endswith(".py")]
                return {"target": paths[0], "kind": "script"} if paths else None
            if module.split(".")[0] in LOCAL_MODULE_ROOTS:
                return {"target": module.replace(".", "/") + ".py", "kind": "module"}
            return None
        if token.startswith("-"):
            index += 1
            continue
        if token.endswith(".py"):
            return {"target": token, "kind": "script"}
        return None
    return None


def _split_invocation(text: str) -> list[str]:
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError:
        tokens = text.split()
    stop = {"|", "||", "&&", ";", ">", ">>", "2>&1", "<", "<<"}
    out = []
    for token in tokens:
        if token in stop or token.startswith(("|", ">", "<<", "2>")):
            break
        out.append(token)
    return out


def workflow_invocations(text: str) -> list[dict]:
    """Every ``python`` invocation in the ``run:`` blocks of one workflow file."""
    payload = yaml.safe_load(text) or {}
    found = []
    for job_name, job in (payload.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            script = step.get("run")
            if not isinstance(script, str):
                continue
            joined = re.sub(r"\\\r?\n\s*", " ", script)
            in_heredoc = None
            for line in joined.splitlines():
                stripped = line.strip()
                if in_heredoc is not None:
                    if stripped == in_heredoc:
                        in_heredoc = None
                    continue
                heredoc = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", stripped)
                for match in PYTHON_CALL.finditer(" " + stripped):
                    args = _split_invocation(stripped[match.end() - 1:])
                    found.append({"job": job_name, "step": step.get("name", ""), "command": stripped, "entry": entrypoint(args)})
                if heredoc:
                    in_heredoc = heredoc.group(1)
    return found


def load_census(root: Path = ROOT) -> dict:
    census = yaml.safe_load((root / CENSUS).read_text(encoding="utf-8"))
    if census.get("schema") != "gg-math-mission-proof-census/v1":
        raise ValueError(f"{CENSUS}: unexpected schema {census.get('schema')!r}")
    ids = [proof["id"] for proof in census.get("proofs", [])]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{CENSUS}: duplicate proof ids")
    return census


def step_target(argv: list[str]) -> str | None:
    entry = entrypoint([item for item in argv[1:]])
    return entry.get("target") if entry else None


def account(root: Path, census: dict) -> dict:
    """Map every workflow entrypoint to its census class; report drift."""
    proof_targets: dict[tuple[str, str], str] = {}
    for proof in census.get("proofs", []):
        for step in proof["steps"]:
            target = step_target(step["argv"])
            if target:
                for workflow in proof["workflows"]:
                    proof_targets.setdefault((workflow, target), proof["id"])
    exclusions = census.get("excluded", [])
    used_proof_keys: set[tuple[str, str]] = set()
    used_exclusions: set[int] = set()

    workflows = []
    unaccounted = []
    for path in sorted((root / WORKFLOWS).glob("*.y*ml")):
        name = path.name
        rows = []
        inline = 0
        for item in workflow_invocations(path.read_text(encoding="utf-8")):
            entry = item["entry"]
            if entry is None:
                continue
            if entry["kind"] == "inline":
                inline += 1
                continue
            target = entry["target"]
            row = {"target": target, "kind": entry["kind"], "job": item["job"]}
            if name == RELEASE_GATE_WORKFLOW:
                row["class"] = "RELEASE_GATE_STEP"
            elif (name, target) in proof_targets:
                used_proof_keys.add((name, target))
                row["class"] = "PROOF_STEP"
                row["proof"] = proof_targets[(name, target)]
            elif entry["kind"] == "unittest" and target.startswith("tests/") and list(root.glob(target)):
                row["class"] = "PYTEST_INVENTORY"
                row["files"] = sorted(p.relative_to(root).as_posix() for p in root.glob(target))
            else:
                for index, rule in enumerate(exclusions):
                    if rule["workflow"] == name and fnmatch.fnmatch(target, rule["target"]):
                        used_exclusions.add(index)
                        row["class"] = "EXCLUDED"
                        row["reason"] = rule["reason"]
                        break
                else:
                    row["class"] = "UNACCOUNTED"
                    unaccounted.append(f"{name}: {target}")
            rows.append(row)
        workflows.append({"workflow": name, "entrypoints": rows, "inline_assertion_blocks": inline})

    gate_targets = {
        row["target"]
        for wf in workflows if wf["workflow"] == RELEASE_GATE_WORKFLOW
        for row in wf["entrypoints"]
    }
    for wf in workflows:
        for row in wf["entrypoints"]:
            if row["class"] == "UNACCOUNTED" and row["target"] in gate_targets:
                row["class"] = "RELEASE_GATE_STEP"
                unaccounted.remove(f"{wf['workflow']}: {row['target']}")

    stale = []
    for (workflow, target), proof_id in sorted(proof_targets.items()):
        if (workflow, target) in used_proof_keys:
            continue
        proof = next(p for p in census["proofs"] if p["id"] == proof_id)
        if not any((wf, target) in used_proof_keys for wf in proof["workflows"]):
            stale.append(f"proof {proof_id}: no listed workflow runs {target}")
    for index, rule in enumerate(exclusions):
        if index not in used_exclusions:
            stale.append(f"exclusion {rule['workflow']}: {rule['target']} matches nothing")
    known = {wf["workflow"] for wf in workflows}
    for proof in census.get("proofs", []):
        for workflow in proof["workflows"]:
            if workflow not in known:
                stale.append(f"proof {proof['id']}: workflow {workflow} does not exist")

    return {
        "workflows": workflows,
        "unaccounted": sorted(set(unaccounted)),
        "stale": sorted(set(stale)),
    }


def _substitute(value: str, mapping: dict[str, str]) -> str:
    for key, replacement in mapping.items():
        value = value.replace("{" + key + "}", replacement)
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_proof(root: Path, proof: dict, source_sha: str, python: str) -> dict:
    out = (OUT_ROOT / proof["id"]).as_posix()
    (root / out).mkdir(parents=True, exist_ok=True)
    mapping = {"python": python, "source_sha": source_sha, "out": out}
    steps = []
    failed = False
    for step in proof["steps"]:
        argv = [_substitute(item, mapping) for item in step["argv"]]
        shown = ["python" if item == python else item for item in argv]
        if failed:
            steps.append({"argv": shown, "status": "NOT_RUN_AFTER_FAILURE"})
            continue
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        # The mission workflows' shared exact-head binding.
        env["SOURCE_SHA"] = source_sha
        env.update({key: _substitute(str(value), mapping) for key, value in (step.get("env") or {}).items()})
        started = time.monotonic()
        try:
            run = subprocess.run(argv, cwd=root, env=env, capture_output=True, timeout=STEP_TIMEOUT_SECONDS)
            returncode, stdout, stderr = run.returncode, run.stdout, run.stderr
        except subprocess.TimeoutExpired as exc:
            returncode, stdout, stderr = -1, exc.stdout or b"", (exc.stderr or b"") + b"\nTIMEOUT"
        record = {
            "argv": shown,
            "returncode": returncode,
            "seconds": round(time.monotonic() - started, 3),
            "stdout_sha256": _sha256_bytes(stdout),
            "stderr_sha256": _sha256_bytes(stderr),
        }
        problems = []
        if returncode != 0:
            problems.append(f"exit {returncode}")
        token = step.get("expect_stdout")
        if token:
            record["expect_stdout"] = token
            if token.encode() not in stdout + stderr:
                problems.append(f"missing {token}")
        products = []
        for raw in step.get("produces") or []:
            rel = _substitute(raw, mapping)
            path = root / rel
            item = {"path": rel}
            if not path.is_file() or path.stat().st_size == 0:
                problems.append(f"missing product {rel}")
                item["status"] = "MISSING"
            else:
                item["sha256"] = _sha256_bytes(path.read_bytes())
                item["bytes"] = path.stat().st_size
                item["status"] = "PRESENT"
                if rel.endswith(".json"):
                    try:
                        json.loads(path.read_text(encoding="utf-8"))
                    except ValueError:
                        problems.append(f"invalid JSON {rel}")
                        item["status"] = "INVALID_JSON"
            products.append(item)
        if products:
            record["produces"] = products
        if problems:
            failed = True
            record["status"] = "FAIL"
            record["problems"] = problems
            record["output_tail"] = (stdout + stderr).decode("utf-8", errors="replace")[-3000:]
        else:
            record["status"] = "PASS"
        steps.append(record)
    return {"id": proof["id"], "workflows": proof["workflows"], "status": "FAIL" if failed else "PASS", "steps": steps}


def run(root: Path, source_sha: str, only: list[str] | None = None, python: str = sys.executable) -> dict:
    census = load_census(root)
    accounting = account(root, census)
    proofs = [p for p in census["proofs"] if not only or p["id"] in only]
    results = [run_proof(root, proof, source_sha, python) for proof in proofs]
    failed = [r["id"] for r in results if r["status"] != "PASS"]
    status = "PASS" if not failed and not accounting["unaccounted"] and not accounting["stale"] and not only else "FAIL"
    classes: dict[str, int] = {}
    for wf in accounting["workflows"]:
        for row in wf["entrypoints"]:
            classes[row["class"]] = classes.get(row["class"], 0) + 1
    return {
        "schema": SCHEMA,
        "status": status,
        "source_sha": source_sha,
        "census": CENSUS.as_posix(),
        "census_sha256": _sha256_bytes((root / CENSUS).read_bytes()),
        "partial_run": bool(only),
        "summary": {
            "proofs": len(results),
            "proofs_passed": len(results) - len(failed),
            "proofs_failed": failed,
            "steps": sum(len(r["steps"]) for r in results),
            "workflows": len(accounting["workflows"]),
            "entrypoint_classes": dict(sorted(classes.items())),
            "inline_assertion_blocks": sum(wf["inline_assertion_blocks"] for wf in accounting["workflows"]),
        },
        "unaccounted": accounting["unaccounted"],
        "stale": accounting["stale"],
        "proofs": results,
        "workflows": accounting["workflows"],
        "authority_transfer": False,
        "formal_credit_delta": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="execute every proof and the workflow accounting")
    run_parser.add_argument("--source-sha", required=True)
    run_parser.add_argument("--out", type=Path, default=Path("release/mission-proofs.json"))
    run_parser.add_argument("--only", action="append", help="debugging: run selected proof ids (result is FAIL)")
    sub.add_parser("check", help="workflow accounting only")
    args = parser.parse_args()

    if args.command == "check":
        accounting = account(ROOT, load_census(ROOT))
        print(json.dumps({"unaccounted": accounting["unaccounted"], "stale": accounting["stale"]}, indent=2))
        if accounting["unaccounted"] or accounting["stale"]:
            raise SystemExit("FAIL_MISSION_PROOF_CENSUS")
        print("PASS_MISSION_PROOF_CENSUS_ACCOUNTING")
        return

    source_sha = args.source_sha.lower()
    if not SHA40.match(source_sha):
        raise SystemExit("--source-sha must be an exact 40-hex SHA")
    result = run(ROOT, source_sha, args.only)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = dict(result["summary"], status=result["status"], unaccounted=result["unaccounted"], stale=result["stale"])
    print(json.dumps(summary, sort_keys=True))
    for proof in result["proofs"]:
        for step in proof["steps"]:
            if step.get("status") == "FAIL":
                print(f"--- {proof['id']}: {' '.join(step['argv'])}: {'; '.join(step['problems'])}", file=sys.stderr)
                print(step.get("output_tail", ""), file=sys.stderr)
    if result["status"] != "PASS":
        raise SystemExit("FAIL_MISSION_PROOF_CENSUS")


if __name__ == "__main__":
    main()
