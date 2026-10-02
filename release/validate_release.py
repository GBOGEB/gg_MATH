"""Fail-closed validation for the gg_MATH generated release/Pages surface."""
from __future__ import annotations

import argparse
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit


EXPECTED = (
    "index.html",
    "status/index.html",
    "methods/index.html",
    "research/consolidation/index.html",
    "research/consolidation/dashboard.html",
    "lm10/w3/index.html",
    "lm10/w3/visual-receipt.html",
    "grandmission-i-b/index.html",
    "grandmission-i-b/compendium.html",
    "receipts/index.html",
    "qa/index.html",
    "release-manifest.json",
    "SHA256SUMS",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    manifest_path = site / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    if junit.get("failures") or junit.get("errors"):
        errors.append(f"release-manifest JUnit is not green: {junit}")
    for entry in manifest.get("files", []):
        path = site / entry["path"]
        if not path.is_file():
            errors.append(f"manifest file missing: {entry['path']}")
        elif sha256(path) != entry["sha256"]:
            errors.append(f"manifest digest mismatch: {entry['path']}")
    return errors


def validate_checksums(site: Path) -> list[str]:
    errors: list[str] = []
    for line in (site / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, rel = line.split("  ", 1)
        except ValueError:
            errors.append(f"malformed SHA256SUMS line: {line}")
            continue
        path = site / rel
        if not path.is_file():
            errors.append(f"SHA256SUMS file missing: {rel}")
        elif sha256(path) != digest:
            errors.append(f"SHA256SUMS digest mismatch: {rel}")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    site = args.site_dir.resolve()
    source_sha = args.source_sha.lower()
    errors: list[str] = []
    for rel in EXPECTED:
        if not (site / rel).is_file():
            errors.append(f"expected release path missing: {rel}")
    errors.extend(validate_links(site))
    errors.extend(validate_manifest(site, source_sha))
    errors.extend(validate_checksums(site))
    errors.extend(validate_receipts(site))
    if errors:
        raise SystemExit("FAIL_RELEASE_VALIDATION\n" + "\n".join(f"- {item}" for item in errors))
    print(f"PASS_RELEASE_VALIDATION files={sum(1 for p in site.rglob('*') if p.is_file())} source_sha={source_sha}")


if __name__ == "__main__":
    main()
