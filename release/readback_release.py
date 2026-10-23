"""Read back a published gg_MATH GitHub Release (R9).

Inputs are the JSON that ``gh`` returns for the release and the tag, the
assets downloaded from the published release, and the locally packaged
assets. The point release is declared ``RELEASE_DEPLOYED`` only when:

* the release is public (not draft), uses the expected tag and prerelease flag;
* the tag resolves (directly or through an annotated tag object) to the exact
  source SHA;
* the published asset set equals the packaged set and every downloaded asset
  matches ``RELEASE_SHA256SUMS`` byte for byte;
* the downloaded site bundle re-validates with ``validate_release.validate_site``;
* the live Pages ``release-manifest.json`` is still the bundled one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from readback_pages import fetch  # noqa: E402
from validate_release import validate_site  # noqa: E402

SUMS = "RELEASE_SHA256SUMS"
SCHEMA = "gg-math-release-readback/v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_sums(text: str) -> dict[str, str]:
    sums = {}
    for line in text.splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            sums[name] = digest
    return sums


def tag_commit(tag_ref: dict, tag_object: dict | None) -> str | None:
    target = tag_ref.get("object") or {}
    if target.get("type") == "commit":
        return target.get("sha")
    if target.get("type") == "tag" and tag_object:
        if tag_object.get("sha") != target.get("sha"):
            return None
        inner = tag_object.get("object") or {}
        return inner.get("sha") if inner.get("type") == "commit" else None
    return None


def safe_extract(bundle: Path, into: Path) -> Path:
    with tarfile.open(bundle, "r:gz") as tar:
        members = tar.getmembers()
        base = into.resolve()
        for member in members:
            target = (into / member.name).resolve()
            if not target.is_relative_to(base) or member.issym() or member.islnk():
                raise ValueError(f"unsafe bundle member {member.name}")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(into, members=members, filter="data")
        else:
            tar.extractall(into, members=members)
    roots = sorted({member.name.split("/", 1)[0] for member in members})
    if len(roots) != 1:
        raise ValueError(f"bundle must have one top-level directory, found {roots}")
    return into / roots[0]


def readback(
    *,
    release: dict,
    tag_ref: dict,
    tag_object: dict | None,
    downloaded: Path,
    packaged: Path,
    source_sha: str,
    tag: str,
    prerelease: bool,
    pages_url: str | None,
) -> dict:
    errors: list[str] = []
    if release.get("tagName") != tag:
        errors.append(f"release tag {release.get('tagName')!r} != {tag!r}")
    if release.get("isDraft"):
        errors.append("release is still a draft")
    if bool(release.get("isPrerelease")) != prerelease:
        errors.append(f"release prerelease flag {release.get('isPrerelease')} != {prerelease}")
    commit = tag_commit(tag_ref, tag_object)
    if commit != source_sha:
        errors.append(f"tag {tag} resolves to {commit!r}, not {source_sha}")

    expected_sums = parse_sums((packaged / SUMS).read_text(encoding="utf-8"))
    packaged_names = {p.name for p in packaged.iterdir() if p.is_file()}
    published = {asset.get("name"): asset for asset in release.get("assets", [])}
    if set(published) != packaged_names:
        errors.append(f"published assets {sorted(published)} != packaged {sorted(packaged_names)}")
    downloaded_sums_path = downloaded / SUMS
    if not downloaded_sums_path.is_file() or downloaded_sums_path.read_bytes() != (packaged / SUMS).read_bytes():
        errors.append(f"downloaded {SUMS} differs from the packaged one")
    assets = []
    for name, digest in sorted(expected_sums.items()):
        path = downloaded / name
        actual = sha256_bytes(path.read_bytes()) if path.is_file() else None
        size_ok = path.is_file() and published.get(name, {}).get("size") in (None, path.stat().st_size)
        ok = actual == digest and size_ok
        assets.append({"name": name, "sha256": actual, "match": ok})
        if not ok:
            errors.append(f"asset {name}: {'missing' if actual is None else 'digest/size mismatch'}")

    bundle_errors: list[str] = []
    bundle_manifest_sha = None
    bundles = sorted(downloaded.glob("gg_MATH-*-site.tar.gz"))
    if len(bundles) != 1:
        errors.append(f"expected one site bundle, found {len(bundles)}")
    else:
        with tempfile.TemporaryDirectory() as raw:
            try:
                site = safe_extract(bundles[0], Path(raw))
                bundle_errors = validate_site(site, source_sha)
                bundle_manifest_sha = sha256_bytes((site / "release-manifest.json").read_bytes())
            except (ValueError, OSError, tarfile.TarError) as exc:
                bundle_errors = [f"bundle unreadable: {exc}"]
        errors.extend(f"bundle: {item}" for item in bundle_errors)

    pages_status = None
    if pages_url and bundle_manifest_sha:
        code, body = fetch(pages_url.rstrip("/") + f"/release-manifest.json?readback={source_sha[:12]}")
        pages_status = code
        if code != 200 or sha256_bytes(body) != bundle_manifest_sha:
            errors.append(f"live Pages manifest HTTP {code} does not match the released bundle")

    return {
        "schema": SCHEMA,
        "status": "FAIL" if errors else "RELEASE_DEPLOYED",
        "tag": tag,
        "source_sha": source_sha,
        "release_url": release.get("url"),
        "prerelease": prerelease,
        "assets": assets,
        "bundle_manifest_sha256": bundle_manifest_sha,
        "pages_url": pages_url,
        "pages_manifest_http_status": pages_status,
        "errors": errors,
        "authority_transfer": False,
        "formal_credit_delta": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-json", type=Path, required=True, help="gh release view --json tagName,isDraft,isPrerelease,assets,url")
    parser.add_argument("--tag-ref-json", type=Path, required=True, help="gh api repos/<repo>/git/ref/tags/<tag>")
    parser.add_argument("--tag-object-json", type=Path, help="gh api repos/<repo>/git/tags/<sha> for an annotated tag")
    parser.add_argument("--downloaded-dir", type=Path, required=True)
    parser.add_argument("--packaged-dir", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--prerelease", choices=("true", "false"), required=True)
    parser.add_argument("--pages-url", default="")
    parser.add_argument("--out", type=Path, default=Path("release-readback.json"))
    args = parser.parse_args()
    tag_object = None
    if args.tag_object_json and args.tag_object_json.is_file():
        tag_object = json.loads(args.tag_object_json.read_text(encoding="utf-8"))
    result = readback(
        release=json.loads(args.release_json.read_text(encoding="utf-8")),
        tag_ref=json.loads(args.tag_ref_json.read_text(encoding="utf-8")),
        tag_object=tag_object,
        downloaded=args.downloaded_dir,
        packaged=args.packaged_dir,
        source_sha=args.source_sha.lower(),
        tag=args.tag,
        prerelease=args.prerelease == "true",
        pages_url=args.pages_url or None,
    )
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "tag", "source_sha", "release_url")}, sort_keys=True))
    for item in result["errors"]:
        print(f"- {item}")
    if result["status"] != "RELEASE_DEPLOYED":
        raise SystemExit("FAIL_RELEASE_READBACK")
    print("PASS_RELEASE_READBACK RELEASE_DEPLOYED")


if __name__ == "__main__":
    main()
