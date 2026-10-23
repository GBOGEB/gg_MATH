"""Canonical runtime census for the gg_MATH release gate (R1).

The release runtime is declared by three files:

* ``.python-version``                 -- CPython minor version;
* ``requirements-release.txt``        -- direct pins;
* ``release/constraints-release.txt`` -- every transitive pin.

The census compares the running interpreter and installed distributions with
those declarations and fails closed on drift, on a missing pin, or on an
installed transitive dependency that the lock does not cover.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION_FILE = Path(".python-version")
REQUIREMENTS = Path("requirements-release.txt")
CONSTRAINTS = Path("release/constraints-release.txt")
SCHEMA = "gg-math-runtime-census/v1"
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)$")
# Pinned in the lock but only installed on some platforms.
PLATFORM_OPTIONAL = frozenset({"colorama"})


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = PIN.match(line)
        if not match:
            raise ValueError(f"{path.name}: not an exact '==' pin: {raw!r}")
        name = canonical_name(match.group(1))
        if name in pins:
            raise ValueError(f"{path.name}: duplicate pin for {name}")
        pins[name] = match.group(2)
    return pins


def installed_distributions() -> dict[str, metadata.Distribution]:
    found: dict[str, metadata.Distribution] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            found.setdefault(canonical_name(name), dist)
    return found


def dependency_closure(roots: set[str], dists: dict[str, metadata.Distribution]) -> set[str]:
    """Names reachable from ``roots`` through installed, marker-active requirements."""
    from packaging.requirements import Requirement

    seen: set[str] = set()
    pending = sorted(roots)
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        dist = dists.get(name)
        if dist is None:
            continue
        for raw in dist.requires or []:
            requirement = Requirement(raw)
            if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
                continue
            child = canonical_name(requirement.name)
            if child not in seen:
                pending.append(child)
    return seen


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def census(
    root: Path = ROOT,
    installed: dict[str, str] | None = None,
    closure: set[str] | None = None,
    python_version: str | None = None,
) -> dict:
    direct = read_pins(root / REQUIREMENTS)
    transitive = read_pins(root / CONSTRAINTS)
    expected_python = (root / PYTHON_VERSION_FILE).read_text(encoding="utf-8").strip()
    if installed is None or closure is None:
        dists = installed_distributions()
        installed = {name: dist.version for name, dist in dists.items()} if installed is None else installed
        closure = dependency_closure(set(direct), dists) if closure is None else closure
    actual_python = python_version or f"{sys.version_info.major}.{sys.version_info.minor}"

    errors: list[str] = []
    overlap = sorted(set(direct) & set(transitive))
    if overlap:
        errors.append("pinned in both requirements and constraints: " + ", ".join(overlap))
    if actual_python != expected_python:
        errors.append(f"python {actual_python} != .python-version {expected_python}")

    packages = []
    for name, pinned in sorted({**transitive, **direct}.items()):
        have = installed.get(name)
        kind = "direct" if name in direct else "transitive"
        if have is None and kind == "transitive" and name in PLATFORM_OPTIONAL:
            status = "ABSENT_PLATFORM_OPTIONAL"
        elif have is None:
            status = "MISSING"
            errors.append(f"{name}=={pinned} is pinned but not installed")
        elif have != pinned:
            status = "DRIFT"
            errors.append(f"{name} installed {have} != pinned {pinned}")
        else:
            status = "MATCH"
        packages.append({"name": name, "kind": kind, "pinned": pinned, "installed": have, "status": status})

    unlocked = sorted(name for name in closure if name not in direct and name not in transitive)
    for name in unlocked:
        errors.append(f"{name} {installed.get(name)} is an installed dependency missing from the lock")

    return {
        "schema": SCHEMA,
        "status": "FAIL" if errors else "PASS",
        "python": {
            "expected": expected_python,
            "actual": actual_python,
            "implementation": platform.python_implementation(),
            "full_version": platform.python_version(),
        },
        "platform": platform.platform(),
        "declarations": {
            PYTHON_VERSION_FILE.as_posix(): sha256(root / PYTHON_VERSION_FILE),
            REQUIREMENTS.as_posix(): sha256(root / REQUIREMENTS),
            CONSTRAINTS.as_posix(): sha256(root / CONSTRAINTS),
        },
        "packages": packages,
        "unlocked_dependencies": unlocked,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("release/runtime-census.json"))
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="local builds only: record drift as DRIFT_ALLOWED_LOCAL instead of failing",
    )
    args = parser.parse_args()
    result = census()
    if result["status"] == "FAIL" and args.allow_drift:
        result["status"] = "DRIFT_ALLOWED_LOCAL"
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": result["errors"]}, sort_keys=True))
    if result["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
