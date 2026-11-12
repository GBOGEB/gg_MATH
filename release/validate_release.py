"""Fail-closed validation for the gg_MATH generated release/Pages surface."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit


EXPECTED = (
    "index.html",
    "status/index.html",
    "status/status.json",
    "status/runtime.json",
    "status/mission-proofs.json",
    "methods/index.html",
    "research/consolidation/index.html",
    "research/consolidation/dashboard.html",
    "lm10/w3/index.html",
    "lm10/w3/visual-receipt.html",
    "grandmission-i-b/index.html",
    "grandmission-i-b/compendium.html",
    "receipts/index.html",
    "receipts/index.json",
    "qa/index.html",
    "qa/qa.json",
    "release-manifest.json",
    "SHA256SUMS",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_site_path(site: Path, relative: str) -> Path:
    """Resolve a portable, canonical relative file name inside the site root.

    Validate the spelling before Path normalizes it; Windows paths must also
    be rejected when validation runs on Linux. Resolve symlink ancestors before
    checking containment. The generated site must remain immutable during QA.
    """
    if not isinstance(relative, str) or not relative:
        raise ValueError("empty or non-string site path")
    if any(ord(char) < 32 or ord(char) == 127 for char in relative):
        raise ValueError("control character in site path")
    if any(char in relative for char in '\\:<>"?*|'):
        raise ValueError("non-portable site path")
    if any(part in {"", ".", ".."} or part.endswith((" ", ".")) for part in relative.split("/")):
        raise ValueError("site path must contain canonical relative components")
    if any(re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part, re.IGNORECASE)
           for part in relative.split("/")):
        raise ValueError("reserved Windows device name in site path")
    try:
        root = site.resolve()
        path = (root / relative).resolve()
        path.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("site path escapes root or cannot be resolved") from exc
    return path


class LocalLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        for key, value in attrs:
            if key in {"href", "src"} and value:
                self.links.append(value)


def validate_links(site: Path) -> list[str]:
    errors: list[str] = []
    root = site.resolve()
    for page in sorted(site.rglob("*.html")):
        parser = LocalLinkParser()
        parser.feed(page.read_text(encoding="utf-8", errors="replace"))
        for raw in parser.links:
            split = urlsplit(raw)
            if split.scheme or split.netloc or raw.startswith(("#", "mailto:", "data:", "javascript:")):
                continue
            path_part = split.path
            if not path_part:
                continue
            target = (page.parent / path_part).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                errors.append(f"{page.relative_to(site)} escapes site root: {raw}")
                continue
            if not target.exists():
                errors.append(f"{page.relative_to(site)} broken local link: {raw}")
    return errors


def validate_manifest(site: Path, source_sha: str) -> list[str]:
    errors: list[str] = []
    try:
        manifest_path = safe_site_path(site, "release-manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeError) as exc:
        return [f"invalid release-manifest: {exc}"]
    if not isinstance(manifest, dict):
        return ["invalid release-manifest: expected object"]
    if manifest.get("source_sha") != source_sha:
        errors.append("release-manifest source_sha mismatch")
    for key, expected in (
        ("authority_transfer", False),
        ("formal_credit_delta", 0),
        ("engineering_acceptance", False),
        ("qps_threshold_authority", False),
    ):
        if manifest.get(key) != expected:
            errors.append(f"release-manifest {key} mismatch")
    junit = manifest.get("junit", {})
    if not isinstance(junit, dict):
        errors.append("release-manifest JUnit must be an object")
        junit = {}
    if junit.get("failures") or junit.get("errors"):
        errors.append(f"release-manifest JUnit is not green: {junit}")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        return errors + ["release-manifest files must be a nonempty list"]
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("malformed manifest file entry")
            continue
        rel = entry.get("path")
        try:
            path = safe_site_path(site, rel)
        except ValueError as exc:
            errors.append(f"invalid manifest path {rel!r}: {exc}")
            continue
        if not path.is_file():
            errors.append(f"manifest file missing: {rel}")
        else:
            digest = entry.get("sha256")
            try:
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or sha256(path) != digest:
                    errors.append(f"manifest digest mismatch: {rel}")
            except OSError as exc:
                errors.append(f"manifest file unreadable: {rel}: {exc}")
    return errors


def validate_checksums(site: Path) -> list[str]:
    errors: list[str] = []
    try:
        lines = safe_site_path(site, "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    except (ValueError, OSError, UnicodeError) as exc:
        return [f"invalid SHA256SUMS: {exc}"]
    for line in lines:
        if not line.strip():
            continue
        try:
            digest, rel = line.split("  ", 1)
        except ValueError:
            errors.append(f"malformed SHA256SUMS line: {line}")
            continue
        try:
            path = safe_site_path(site, rel)
        except ValueError as exc:
            errors.append(f"invalid SHA256SUMS path {rel!r}: {exc}")
            continue
        if not path.is_file():
            errors.append(f"SHA256SUMS file missing: {rel}")
        else:
            try:
                if not re.fullmatch(r"[0-9a-f]{64}", digest) or sha256(path) != digest:
                    errors.append(f"SHA256SUMS digest mismatch: {rel}")
            except OSError as exc:
                errors.append(f"SHA256SUMS file unreadable: {rel}: {exc}")
    return errors


def validate_machine_views(site: Path, source_sha: str) -> list[str]:
    errors: list[str] = []
    for rel in ("status/status.json", "receipts/index.json", "qa/qa.json"):
        payload = json.loads((site / rel).read_text(encoding="utf-8"))
        if payload.get("source_sha") != source_sha:
            errors.append(f"{rel} source_sha mismatch")
    registry = json.loads((site / "receipts/index.json").read_text(encoding="utf-8"))
    for item in registry.get("receipts", []):
        if item.get("source_sha") != source_sha and item.get("source_binding") == "RELEASE_MANIFEST_WRAPPER":
            errors.append(f"receipt registry wrapper binding mismatch: {item.get('path')}")
    return errors


def validate_receipts(site: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted((site / "receipts/files").rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(payload, sort_keys=True)
        if "LOCAL_UNBOUND" in text:
            errors.append(f"unbound receipt published: {path.relative_to(site)}")
        if "authority_transfer" in payload and payload["authority_transfer"] is not False:
            errors.append(f"authority transfer promoted: {path.relative_to(site)}")
        if "formal_credit_delta" in payload:
            value = payload["formal_credit_delta"]
            if type(value) is not int or value != 0:
                errors.append(f"formal credit promoted: {path.relative_to(site)}")
    return errors


def validate_admission(site: Path, source_sha: str, allow_local_unadmitted: bool = False) -> list[str]:
    """R1/R2/R3/R0: admitted gate state and a release identity bound to the SHA."""
    errors: list[str] = []
    manifest = json.loads((site / "release-manifest.json").read_text(encoding="utf-8"))
    admission = manifest.get("admission")
    if admission != "ADMITTED" and not (allow_local_unadmitted and admission == "LOCAL_UNADMITTED"):
        errors.append(f"release-manifest admission is {admission!r}, not ADMITTED")
    if admission == "ADMITTED":
        if manifest.get("admission_blockers"):
            errors.append("ADMITTED manifest carries admission blockers")
        if manifest.get("collection_gaps"):
            errors.append(f"collection gaps: {manifest.get('collection_gaps')}")
        if (manifest.get("runtime") or {}).get("status") != "PASS":
            errors.append("runtime census is not PASS")
        if (manifest.get("mission_proofs") or {}).get("status") != "PASS":
            errors.append("mission-proof census is not PASS")
    proofs = json.loads((site / "status/mission-proofs.json").read_text(encoding="utf-8"))
    if proofs.get("source_sha") != source_sha:
        errors.append("status/mission-proofs.json source_sha mismatch")
    version = str(manifest.get("release_version", ""))
    canonical = str(manifest.get("canonical_version", ""))
    tag = str(manifest.get("release_tag", ""))
    if manifest.get("release_mode") == "release":
        if version != canonical or tag != f"v{canonical}" or "+" in version:
            errors.append(f"release identity {version!r}/{tag!r} is not v<release/VERSION>")
        if not manifest.get("license_file"):
            errors.append("release without a licence file")
        if not manifest.get("changelog_section"):
            errors.append("release without a CHANGELOG.md section")
    else:
        if tag:
            errors.append(f"non-release build carries tag {tag!r}")
        if not version.endswith(f"+sha.{source_sha[:12]}"):
            errors.append(f"development identity {version!r} is not bound to +sha.{source_sha[:12]}")
    return errors


def validate_site(site: Path, source_sha: str, allow_local_unadmitted: bool = False) -> list[str]:
    errors: list[str] = []
    for rel in EXPECTED:
        try:
            path = safe_site_path(site, rel)
        except ValueError as exc:
            errors.append(f"invalid expected release path {rel}: {exc}")
            continue
        if not path.is_file():
            errors.append(f"expected release path missing: {rel}")
    if errors:
        return errors
    errors.extend(validate_links(site))
    errors.extend(validate_manifest(site, source_sha))
    errors.extend(validate_checksums(site))
    errors.extend(validate_machine_views(site, source_sha))
    errors.extend(validate_receipts(site))
    errors.extend(validate_admission(site, source_sha, allow_local_unadmitted))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument(
        "--allow-local-unadmitted",
        action="store_true",
        help="local inspection only: accept a LOCAL_UNADMITTED build",
    )
    args = parser.parse_args()
    site = args.site_dir.resolve()
    source_sha = args.source_sha.lower()
    errors = validate_site(site, source_sha, args.allow_local_unadmitted)
    if errors:
        raise SystemExit("FAIL_RELEASE_VALIDATION\n" + "\n".join(f"- {item}" for item in errors))
    print(f"PASS_RELEASE_VALIDATION files={sum(1 for p in site.rglob('*') if p.is_file())} source_sha={source_sha}")


if __name__ == "__main__":
    main()
