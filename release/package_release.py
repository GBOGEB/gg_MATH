"""Package the admitted gg_MATH build as GitHub Release assets (R8).

Assets (all bound by ``RELEASE_SHA256SUMS``):

* ``gg_MATH-<version>-site.tar.gz`` -- deterministic bundle of the exact Pages
  artifact (sorted entries, fixed mtime/owner/mode, gzip mtime 0);
* ``release-manifest.json``         -- the site manifest;
* gate evidence: JUnit, coverage, runtime and mission-proof census;
* ``pages-readback.json``           -- the deployed-site readback receipt.

Release notes are the ``CHANGELOG.md`` section for the version plus a
generated identity/evidence block.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import tarfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMS = "RELEASE_SHA256SUMS"
EVIDENCE = ("test-results.xml", "coverage.xml", "runtime-census.json", "mission-proofs.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deterministic_tarball(site: Path, out: Path, prefix: str, mtime: int) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(site.rglob("*"), key=lambda p: p.relative_to(site).as_posix()):
            rel = path.relative_to(site).as_posix()
            info = tarfile.TarInfo(f"{prefix}/{rel}")
            info.mtime = mtime
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            if path.is_dir():
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                tar.addfile(info)
            else:
                data = path.read_bytes()
                info.size = len(data)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))
    with out.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(buffer.getvalue())


def changelog_section(root: Path, version: str) -> str:
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", text, re.M | re.S)
    if not match:
        raise ValueError(f"CHANGELOG.md has no section for {version}")
    return match.group(1).strip()


def release_notes(manifest: dict, section: str, pages_url: str, readback: dict) -> str:
    junit, coverage = manifest["junit"], manifest["coverage"]
    proofs = manifest["mission_proofs"]["summary"]
    runtime = manifest["runtime"]
    return f"""{section}

---

### Release identity and evidence

| | |
|---|---|
| Version | `{manifest['release_version']}` ({manifest['release_channel']}) |
| Tag | `{manifest['release_tag']}` at exact source `{manifest['source_sha']}` |
| Pages | {pages_url} (read back: {readback['files_matched']}/{readback['files_checked']} files byte-identical, browser QA {'executed' if readback.get('browser_qa_executed') else 'not executed'}) |
| Tests | {junit['tests']} collected, {junit['failures']} failed, {junit['errors']} errors, {junit['skipped']} skipped |
| Coverage | {coverage['line_percent']}% line / {coverage['branch_percent']}% branch |
| Mission proofs | {proofs['proofs_passed']}/{proofs['proofs']} proofs, {proofs['steps']} steps |
| Runtime | CPython {runtime['python']['full_version']}, lock `release/constraints-release.txt` |
| Gate run | {manifest['workflow_run_url'] or 'local'} |

Verify the assets with `sha256sum -c {SUMS}`; the site bundle's own `SHA256SUMS`
and `release-manifest.json` bind every published file.

gg_MATH provides mathematical methods, executable diagnostics and evidence
receipts. This release transfers no engineering, QPS threshold or project
authority (`authority_transfer: false`, `formal_credit_delta: 0`).
"""


def package(root: Path, site: Path, evidence_dir: Path, pages_readback: Path, pages_url: str, out_dir: Path) -> dict:
    manifest = json.loads((site / "release-manifest.json").read_text(encoding="utf-8"))
    readback = json.loads(pages_readback.read_text(encoding="utf-8"))
    if manifest.get("admission") != "ADMITTED" or manifest.get("release_mode") != "release":
        raise ValueError("only an ADMITTED release-mode build can be packaged")
    if readback.get("status") != "PASS" or readback.get("source_sha") != manifest["source_sha"]:
        raise ValueError("Pages readback is not PASS for this source SHA")
    if readback.get("manifest_sha256") != sha256(site / "release-manifest.json"):
        raise ValueError("Pages readback was taken against a different manifest")
    version = manifest["release_version"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    mtime = int(datetime.fromisoformat(manifest["build_timestamp"]).timestamp())
    bundle = out_dir / f"gg_MATH-{version}-site.tar.gz"
    deterministic_tarball(site, bundle, f"gg_MATH-{version}-site", mtime)
    shutil.copy2(site / "release-manifest.json", out_dir / "release-manifest.json")
    for name in EVIDENCE:
        shutil.copy2(evidence_dir / name, out_dir / name)
    shutil.copy2(pages_readback, out_dir / "pages-readback.json")

    assets = sorted(p for p in out_dir.iterdir() if p.is_file())
    (out_dir / SUMS).write_text("".join(f"{sha256(p)}  {p.name}\n" for p in assets), encoding="utf-8")
    notes = release_notes(manifest, changelog_section(root, manifest["canonical_version"]), pages_url, readback)
    return {
        "version": version,
        "tag": manifest["release_tag"],
        "assets": [p.name for p in assets] + [SUMS],
        "notes": notes,
        "manifest_sha256": sha256(site / "release-manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True, help="directory holding the gate's JUnit/coverage/census files")
    parser.add_argument("--pages-readback", type=Path, required=True)
    parser.add_argument("--pages-url", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("dist"))
    parser.add_argument("--notes-out", type=Path, default=Path("release-notes.md"))
    args = parser.parse_args()
    result = package(ROOT, args.site_dir.resolve(), args.evidence_dir.resolve(), args.pages_readback.resolve(), args.pages_url, args.out_dir.resolve())
    args.notes_out.write_text(result.pop("notes"), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
