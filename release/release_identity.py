"""SemVer release identity for the gg_MATH release gate (R0).

``release/VERSION`` is the single canonical next-release identity. Every run of
the release workflow resolves one of three modes:

* ``verify``  -- pull request / push / manual check; nothing is published and
  the identity is ``<VERSION>-dev+sha.<sha12>``;
* ``pages``   -- publish the admitted site to GitHub Pages under that same dev
  identity, as a deploy/readback rehearsal; ``main`` only; no tag, no
  GitHub Release;
* ``release`` -- publish Pages under ``<VERSION>``, then create tag
  ``v<VERSION>`` at the exact source SHA and the GitHub Release.

``release`` fails closed unless the requested version repeats ``VERSION``
exactly, the run is on ``refs/heads/main``, the tag does not exist yet, the
version is newer than every existing ``v*`` tag, ``CHANGELOG.md`` has a section
for it and the repository carries a license file.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = Path("release/VERSION")
CHANGELOG = Path("CHANGELOG.md")
LICENSE_FILES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING")
RELEASE_REF = "refs/heads/main"
MODES = ("verify", "pages", "release")

_ID = r"(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    rf"(?:-({_ID}(?:\.{_ID})*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
SHA40 = re.compile(r"^[0-9a-f]{40}$")


class IdentityError(ValueError):
    pass


def parse_semver(version: str) -> tuple[int, int, int, tuple[str, ...], str]:
    match = SEMVER.match(version)
    if not match:
        raise IdentityError(f"not a SemVer 2.0.0 version: {version!r}")
    major, minor, patch, pre, build = match.groups()
    return int(major), int(minor), int(patch), tuple(pre.split(".")) if pre else (), build or ""


def precedence_key(version: str) -> tuple:
    """Sort key implementing SemVer 2.0.0 section 11 precedence (build ignored)."""
    major, minor, patch, pre, _ = parse_semver(version)
    if not pre:
        return (major, minor, patch, 1, ())
    fields = tuple((0, int(item), "") if item.isdigit() else (1, 0, item) for item in pre)
    return (major, minor, patch, 0, fields)


def read_canonical_version(root: Path = ROOT) -> str:
    version = (root / VERSION_FILE).read_text(encoding="utf-8").strip()
    _, _, _, _, build = parse_semver(version)
    if build:
        raise IdentityError(f"{VERSION_FILE} must not carry build metadata: {version}")
    return version


def changelog_has_section(root: Path, version: str) -> bool:
    path = root / CHANGELOG
    if not path.is_file():
        return False
    heading = re.compile(rf"^## \[{re.escape(version)}\](?:\s|$)", re.M)
    return bool(heading.search(path.read_text(encoding="utf-8")))


def license_file(root: Path) -> str | None:
    for name in LICENSE_FILES:
        if (root / name).is_file():
            return name
    return None


def resolve_identity(
    *,
    mode: str,
    canonical_version: str,
    requested_version: str,
    ref: str,
    source_sha: str,
    existing_tags: list[str],
    changelog_ready: bool,
    license_name: str | None,
) -> dict:
    if mode not in MODES:
        raise IdentityError(f"unknown mode {mode!r}; expected one of {', '.join(MODES)}")
    if not SHA40.match(source_sha):
        raise IdentityError("source_sha must be an exact lowercase 40-hex SHA")
    _, _, _, pre, _ = parse_semver(canonical_version)
    tag = f"v{canonical_version}"

    if mode == "pages" and ref != RELEASE_REF:
        raise IdentityError(f"Pages publication runs from {RELEASE_REF} only, not {ref!r}")
    if mode != "release":
        joiner = "." if pre else "-"
        return {
            "mode": mode,
            "release_version": f"{canonical_version}{joiner}dev+sha.{source_sha[:12]}",
            "canonical_version": canonical_version,
            "tag": "",
            "channel": "development",
            "prerelease": True,
            "publish_pages": mode == "pages",
            "create_release": False,
        }

    errors = []
    if requested_version != canonical_version:
        errors.append(
            f"requested release_version {requested_version!r} does not repeat {VERSION_FILE} {canonical_version!r}"
        )
    if ref != RELEASE_REF:
        errors.append(f"releases are cut from {RELEASE_REF} only, not {ref!r}")
    if tag in existing_tags:
        errors.append(f"tag {tag} already exists")
    prior = []
    for item in existing_tags:
        if not item.startswith("v"):
            continue
        try:
            prior.append((precedence_key(item[1:]), item))
        except IdentityError:
            continue
    if prior:
        newest_key, newest = max(prior)
        if precedence_key(canonical_version) <= newest_key:
            errors.append(f"{canonical_version} is not newer than existing tag {newest}")
    if not changelog_ready:
        errors.append(f"{CHANGELOG} has no '## [{canonical_version}]' section")
    if license_name is None:
        errors.append("no license file (" + ", ".join(LICENSE_FILES) + "); the release licence is an owner decision")
    if errors:
        raise IdentityError("release preconditions failed:\n- " + "\n- ".join(errors))
    return {
        "mode": mode,
        "release_version": canonical_version,
        "canonical_version": canonical_version,
        "tag": tag,
        "channel": "prerelease" if pre else "stable",
        "prerelease": bool(pre),
        "publish_pages": True,
        "create_release": True,
    }


def git_tags(root: Path) -> list[str]:
    output = subprocess.run(
        ["git", "tag", "--list"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    return [line.strip() for line in output.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="verify", choices=MODES)
    parser.add_argument("--requested-version", default="")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--github-output", type=Path, help="append key=value lines for later steps")
    args = parser.parse_args()
    try:
        identity = resolve_identity(
            mode=args.mode,
            canonical_version=read_canonical_version(ROOT),
            requested_version=args.requested_version.strip(),
            ref=args.ref,
            source_sha=args.source_sha.lower(),
            existing_tags=git_tags(ROOT),
            changelog_ready=changelog_has_section(ROOT, read_canonical_version(ROOT)),
            license_name=license_file(ROOT),
        )
    except IdentityError as exc:
        raise SystemExit(f"FAIL_RELEASE_IDENTITY\n{exc}")
    lines = [f"{key}={str(value).lower() if isinstance(value, bool) else value}" for key, value in identity.items()]
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
