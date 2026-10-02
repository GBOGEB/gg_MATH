"""Build the governed gg_MATH GitHub Pages/release surface."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_ARTIFACTS = {
    "consolidation_dashboard": Path("research/consolidation_output/dashboard.html"),
    "consolidation_receipt": Path("research/consolidation_output/receipt.json"),
    "consolidation_csv": Path("research/consolidation_output/synthetic_features.csv"),
    "lm10_w3_html": Path("artifacts/lm10_w3/LM10_W3_VISUAL_RECEIPT.html"),
    "lm10_w3_receipt": Path("artifacts/lm10_w3/LM10_W3_VISUAL_RECEIPT.json"),
    "grandmission_html": Path(
        "artifacts/grandmission_i_b_temporal_pca/"
        "GRANDMISSION_I_B_TEMPORAL_PCA_COMPENDIUM.html"
    ),
    "grandmission_receipt": Path(
        "artifacts/grandmission_i_b_temporal_pca/"
        "GRANDMISSION_I_B_TEMPORAL_PCA_FEDERATION_RECEIPT.json"
    ),
    "grandmission_readiness": Path(
        "artifacts/grandmission_i_b_temporal_pca/"
        "GM_I_A_MULTI_CLOCK_READINESS_RECEIPT.json"
    ),
}

EXPECTED_PAGES = (
    "index.html",
    "status/index.html",
    "methods/index.html",
    "research/consolidation/index.html",
    "lm10/w3/index.html",
    "grandmission-i-b/index.html",
    "receipts/index.html",
    "qa/index.html",
)

CSS = """:root { color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
body { max-width: 1180px; margin: 0 auto; padding: 1.5rem; line-height: 1.5; }
nav { display: flex; flex-wrap: wrap; gap: .8rem; margin: 0 0 1.5rem; }
nav a { text-decoration: none; font-weight: 650; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0 1.5rem; }
th, td { border: 1px solid #8888; padding: .5rem; text-align: left; vertical-align: top; }
code, pre { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
pre { overflow-x: auto; padding: 1rem; border: 1px solid #8886; }
.guard { border-left: 4px solid #777; padding: .75rem 1rem; background: #8881; }
.good { font-weight: 700; }
.warn { font-weight: 700; }
iframe { width: 100%; min-height: 900px; border: 1px solid #8886; background: white; }
.small { font-size: .9rem; opacity: .85; }
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_required(root: Path) -> dict[str, Path]:
    resolved = {name: root / rel for name, rel in EXPECTED_ARTIFACTS.items()}
    missing = [str(path.relative_to(root)) for path in resolved.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing release artifacts: " + ", ".join(sorted(missing)))
    return resolved


def copy_required(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def nav(prefix: str) -> str:
    links = [
        ("Home", "index.html"),
        ("Status", "status/index.html"),
        ("Methods", "methods/index.html"),
        ("Consolidation", "research/consolidation/index.html"),
        ("LM10 W3", "lm10/w3/index.html"),
        ("Grandmission I-B", "grandmission-i-b/index.html"),
        ("Receipts", "receipts/index.html"),
        ("QA", "qa/index.html"),
    ]
    return "<nav>" + " ".join(
        f'<a href="{html.escape(prefix + href)}">{html.escape(label)}</a>'
        for label, href in links
    ) + "</nav>"


def page(title: str, body: str, prefix: str = "") -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} — gg_MATH</title>
<link rel="stylesheet" href="{html.escape(prefix)}assets/site.css">
</head>
<body>
{nav(prefix)}
<h1>{html.escape(title)}</h1>
{body}
<footer class="small"><p>Generated from exact repository source. gg_MATH is a mathematical provider; outward evidence does not transfer engineering or QPS authority.</p></footer>
</body>
</html>
"""


def parse_junit(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    node = root if root.tag == "testsuite" else next(iter(root.findall("testsuite")), root)
    def number(name: str) -> int:
        raw = node.attrib.get(name, root.attrib.get(name, "0"))
        return int(float(raw))
    return {
        "tests": number("tests"),
        "failures": number("failures"),
        "errors": number("errors"),
        "skipped": number("skipped"),
    }


def parse_coverage(path: Path) -> dict[str, float]:
    root = ET.parse(path).getroot()
    line_rate = float(root.attrib.get("line-rate", "0"))
    branch_rate = float(root.attrib.get("branch-rate", "0"))
    return {
        "line_rate": line_rate,
        "branch_rate": branch_rate,
        "line_percent": round(line_rate * 100.0, 2),
        "branch_percent": round(branch_rate * 100.0, 2),
    }


def load_methods(root: Path) -> tuple[dict, dict]:
    graph = yaml.safe_load((root / "research/MATH_METHOD_DEPENDENCY_GRAPH_v1.yaml").read_text())
    cards = yaml.safe_load((root / "research/MATH_METHOD_INFO_CARDS_v1.yaml").read_text())
    return graph, cards


def authority_value(payload: dict, key: str):
    if key in payload:
        return payload[key]
    provider = payload.get("provider")
    if isinstance(provider, dict) and key in provider:
        return provider[key]
    return None


def workflow_inventory(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / ".github/workflows").glob("*.y*ml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rows.append({
            "name": payload.get("name", path.stem),
            "path": path.relative_to(root).as_posix(),
            "release_gate": path.name == "release-infrastructure-v1.yml",
        })
    return rows


def receipt_rows(root: Path, site: Path, release_source_sha: str) -> list[dict]:
    candidates: list[Path] = []
    artifact_root = root / "artifacts"
    if artifact_root.exists():
        candidates.extend(sorted(artifact_root.rglob("*.json")))
    consolidation = root / "research/consolidation_output/receipt.json"
    if consolidation.is_file():
        candidates.append(consolidation)

    rows = []
    seen = set()
    for source in candidates:
        rel = source.relative_to(root)
        rel_text = rel.as_posix()
        if rel_text in seen:
            continue
        seen.add(rel_text)
        target = site / "receipts/files" / rel
        copy_required(source, target)
        payload = json.loads(source.read_text(encoding="utf-8"))
        provider = payload.get("provider") if isinstance(payload.get("provider"), dict) else {}
        intrinsic_source_sha = payload.get("source_sha", payload.get("sha", provider.get("exact_sha", "")))
        intrinsically_bound = (
            isinstance(intrinsic_source_sha, str)
            and len(intrinsic_source_sha) == 40
            and all(c in "0123456789abcdef" for c in intrinsic_source_sha.lower())
        )
        rows.append({
            "path": rel_text,
            "href": "files/" + rel_text,
            "schema": payload.get("schema", payload.get("schema_version", "UNSPECIFIED")),
            "source_sha": intrinsic_source_sha if intrinsically_bound else release_source_sha,
            "source_binding": "INTRINSIC_RECEIPT" if intrinsically_bound else "RELEASE_MANIFEST_WRAPPER",
            "authority_transfer": authority_value(payload, "authority_transfer"),
            "formal_credit_delta": authority_value(payload, "formal_credit_delta"),
            "sha256": sha256(source),
        })
    return rows


def build_site(
    root: Path,
    site: Path,
    source_sha: str,
    release_version: str,
    run_url: str,
    build_timestamp: str,
    junit: Path,
    coverage: Path,
) -> dict:
    if len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha.lower()):
        raise ValueError("source_sha must be exact 40-hex")
    artifacts = ensure_required(root)
    junit_summary = parse_junit(junit)
    coverage_summary = parse_coverage(coverage)
    if junit_summary["tests"] <= 0:
        raise ValueError("JUnit collected zero tests")
    if junit_summary["failures"] or junit_summary["errors"] or junit_summary["skipped"]:
        raise ValueError(f"JUnit is not release-green: {junit_summary}")

    if site.exists():
        shutil.rmtree(site)
    (site / "assets/data").mkdir(parents=True, exist_ok=True)
    (site / "assets/site.css").write_text(CSS, encoding="utf-8")

    graph, cards = load_methods(root)
    copy_required(root / "research/MATH_METHOD_DEPENDENCY_GRAPH_v1.yaml", site / "assets/data/MATH_METHOD_DEPENDENCY_GRAPH_v1.yaml")
    copy_required(root / "research/MATH_METHOD_INFO_CARDS_v1.yaml", site / "assets/data/MATH_METHOD_INFO_CARDS_v1.yaml")

    copy_required(artifacts["consolidation_dashboard"], site / "research/consolidation/dashboard.html")
    copy_required(artifacts["consolidation_receipt"], site / "research/consolidation/receipt.json")
    copy_required(artifacts["consolidation_csv"], site / "research/consolidation/synthetic_features.csv")
    copy_required(artifacts["lm10_w3_html"], site / "lm10/w3/visual-receipt.html")
    copy_required(artifacts["lm10_w3_receipt"], site / "lm10/w3/receipt.json")
    copy_required(artifacts["grandmission_html"], site / "grandmission-i-b/compendium.html")
    copy_required(artifacts["grandmission_receipt"], site / "grandmission-i-b/federation-receipt.json")
    copy_required(artifacts["grandmission_readiness"], site / "grandmission-i-b/readiness-receipt.json")
    copy_required(junit, site / "qa/test-results.xml")
    copy_required(coverage, site / "qa/coverage.xml")

    workflow_index = workflow_inventory(root)
    receipt_index = receipt_rows(root, site, source_sha)
    receipts_registry = {
        "schema": "gg-math-receipt-registry/v1",
        "repository": "GBOGEB/gg_MATH",
        "source_sha": source_sha,
        "release_version": release_version,
        "receipts": receipt_index,
    }
    (site / "receipts/index.json").write_text(
        json.dumps(receipts_registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    method_rows = []
    node_by_id = {item["id"]: item["topic"] for item in graph.get("nodes", [])}
    for card in cards.get("cards", []):
        threshold = card.get("threshold", {})
        method_rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(card.get('id', '')))}</code></td>"
            f"<td>{html.escape(str(card.get('topic', '')))}</td>"
            f"<td><code>{html.escape(str(card.get('math', '')))}</code></td>"
            f"<td>{html.escape(str(card.get('value', '')))}</td>"
            f"<td>{html.escape(str(threshold.get('kind', '')))}</td>"
            f"<td>{html.escape(str(threshold.get('rule', '')))}</td>"
            "</tr>"
        )
    edge_rows = []
    for edge in graph.get("edges", []):
        if len(edge) == 3:
            left, right, kind = edge
            edge_rows.append(
                f"<tr><td>{html.escape(str(left))} — {html.escape(node_by_id.get(left, ''))}</td>"
                f"<td>{html.escape(str(kind))}</td>"
                f"<td>{html.escape(str(right))} — {html.escape(node_by_id.get(right, ''))}</td></tr>"
            )

    methods_body = f"""
<div class="guard"><b>Method authority:</b> method cards describe mathematical operations, assumptions and threshold kinds. PROJECT_GOVERNED values remain external/read-only to gg_MATH.</div>
<p><a href="../assets/data/MATH_METHOD_DEPENDENCY_GRAPH_v1.yaml">Dependency graph YAML</a> ·
<a href="../assets/data/MATH_METHOD_INFO_CARDS_v1.yaml">Method cards YAML</a></p>
<h2>Method cards</h2>
<table><tr><th>ID</th><th>Topic</th><th>Math</th><th>Value</th><th>Threshold kind</th><th>Rule</th></tr>{''.join(method_rows)}</table>
<h2>Typed dependency edges</h2>
<table><tr><th>From</th><th>Relation</th><th>To</th></tr>{''.join(edge_rows)}</table>
"""
    (site / "methods").mkdir(parents=True, exist_ok=True)
    (site / "methods/index.html").write_text(page("Mathematical method navigator", methods_body, "../"), encoding="utf-8")

    receipt_table = "".join(
        "<tr>"
        f"<td><a href=\"{html.escape(row['href'])}\">{html.escape(row['path'])}</a></td>"
        f"<td>{html.escape(str(row['schema']))}</td>"
        f"<td><code>{html.escape(str(row['source_sha']))}</code></td>"
        f"<td>{html.escape(str(row['authority_transfer']))}</td>"
        f"<td>{html.escape(str(row['formal_credit_delta']))}</td>"
        f"<td><code>{row['sha256']}</code></td>"
        "</tr>"
        for row in receipt_index
    )
    receipts_body = f"""
<p>Generated receipt census for this exact release build.</p>
<p><a href="index.json">Machine-readable receipt registry</a></p>
<table><tr><th>Receipt</th><th>Schema</th><th>Source SHA</th><th>Authority transfer</th><th>Formal credit delta</th><th>SHA-256</th></tr>{receipt_table}</table>
"""
    (site / "receipts").mkdir(parents=True, exist_ok=True)
    (site / "receipts/index.html").write_text(page("Receipt registry", receipts_body, "../"), encoding="utf-8")

    qa_body = f"""
<div class="guard"><b>Release-gate QA:</b> PASS requires the complete pytest invocation to complete before this site is built.</div>
<table>
<tr><th>Tests</th><td>{junit_summary['tests']}</td></tr>
<tr><th>Failures</th><td>{junit_summary['failures']}</td></tr>
<tr><th>Errors</th><td>{junit_summary['errors']}</td></tr>
<tr><th>Skipped</th><td>{junit_summary['skipped']}</td></tr>
<tr><th>Line coverage</th><td>{coverage_summary['line_percent']}%</td></tr>
<tr><th>Branch coverage</th><td>{coverage_summary['branch_percent']}%</td></tr>
</table>
<p><a href="qa.json">Machine QA JSON</a> · <a href="test-results.xml">JUnit XML</a> · <a href="coverage.xml">Coverage XML</a></p>
<p>Coverage is reported as evidence; Release Infrastructure v1 does not invent a minimum threshold that the repository has not yet governed.</p>
"""
    (site / "qa").mkdir(parents=True, exist_ok=True)
    (site / "qa/index.html").write_text(page("Release QA", qa_body, "../"), encoding="utf-8")
    (site / "qa/qa.json").write_text(
        json.dumps({
            "schema": "gg-math-release-qa/v1",
            "source_sha": source_sha,
            "release_version": release_version,
            "junit": junit_summary,
            "coverage": coverage_summary,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    status_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td class=\"good\">PRESENT</td><td><code>{html.escape(str(path.relative_to(root)))}</code></td></tr>"
        for name, path in artifacts.items()
    )
    workflow_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['name'])}</td>"
        f"<td><code>{html.escape(item['path'])}</code></td>"
        f"<td>{'EXECUTED_RELEASE_GATE' if item['release_gate'] else 'INVENTORIED_NOT_AGGREGATED'}</td>"
        "</tr>"
        for item in workflow_index
    )
    status_body = f"""
<table>
<tr><th>Release identity</th><td><code>{html.escape(release_version)}</code></td></tr>
<tr><th>Exact source SHA</th><td><code>{html.escape(source_sha)}</code></td></tr>
<tr><th>Build timestamp</th><td>{html.escape(build_timestamp)}</td></tr>
<tr><th>Workflow run</th><td><a href="{html.escape(run_url)}">{html.escape(run_url or 'local')}</a></td></tr>
<tr><th>Pytest result</th><td class="good">PASS ({junit_summary['tests']} tests; {junit_summary['skipped']} skipped)</td></tr>
<tr><th>Coverage</th><td>{coverage_summary['line_percent']}% line / {coverage_summary['branch_percent']}% branch</td></tr>
</table>
<h2>Expected outward artifact census</h2>
<table><tr><th>Artifact</th><th>Status</th><th>Source path</th></tr>{status_rows}</table>
<h2>Workflow census</h2>
<p>The release gate executes the complete pytest/release build. Other mission workflows are inventoried here but are not falsely aggregated into a green state.</p>
<table><tr><th>Workflow</th><th>Path</th><th>This release run</th></tr>{workflow_rows}</table>
<div class="guard"><b>Fail-closed publication:</b> the builder aborts if any expected release artifact is missing or if JUnit is zero-test, skipped, failed or errored.</div>
<p><a href="status.json">Machine status JSON</a> · <a href="../release-manifest.json">Release manifest</a> · <a href="../SHA256SUMS">SHA256SUMS</a></p>
"""
    (site / "status").mkdir(parents=True, exist_ok=True)
    (site / "status/index.html").write_text(page("Build and release status", status_body, "../"), encoding="utf-8")
    (site / "status/status.json").write_text(
        json.dumps({
            "schema": "gg-math-release-status/v1",
            "repository": "GBOGEB/gg_MATH",
            "source_sha": source_sha,
            "release_version": release_version,
            "build_timestamp": build_timestamp,
            "workflow_run_url": run_url,
            "release_gate": {"junit": junit_summary, "coverage": coverage_summary},
            "workflow_inventory": workflow_index,
            "artifact_census": [
                {"name": name, "status": "PRESENT", "source_path": path.relative_to(root).as_posix()}
                for name, path in artifacts.items()
            ],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    consolidation_body = """
<div class="guard"><b>SYNTHETIC RESEARCH ONLY.</b> No QPS engineering, ranking, acceptance or release credit is created by the consolidation population.</div>
<p><a href="receipt.json">Machine receipt</a> · <a href="synthetic_features.csv">Synthetic CSV</a> · <a href="dashboard.html">Open dashboard directly</a></p>
<iframe src="dashboard.html" title="Math consolidation dashboard"></iframe>
"""
    (site / "research/consolidation/index.html").write_text(page("Consolidation research dashboard", consolidation_body, "../../"), encoding="utf-8")

    lm10_body = """
<div class="guard"><b>Authority cap:</b> LM-10 W3 is synthetic/calibration evidence and does not promote project authority.</div>
<p><a href="receipt.json">Machine receipt</a> · <a href="visual-receipt.html">Open visual receipt directly</a></p>
<iframe src="visual-receipt.html" title="LM10 W3 visual receipt"></iframe>
"""
    (site / "lm10/w3/index.html").write_text(page("LM-10 W3 visual receipt", lm10_body, "../../"), encoding="utf-8")

    grandmission_body = """
<div class="guard"><b>Federation boundary:</b> authority transfer remains false and formal credit remains zero.</div>
<p><a href="federation-receipt.json">Federation receipt</a> · <a href="readiness-receipt.json">Readiness receipt</a> · <a href="compendium.html">Open compendium directly</a></p>
<iframe src="compendium.html" title="Grandmission I-B Temporal PCA compendium"></iframe>
"""
    (site / "grandmission-i-b/index.html").write_text(page("Grandmission I-B Temporal PCA", grandmission_body, "../"), encoding="utf-8")

    home_body = f"""
<p><b>Release:</b> <code>{html.escape(release_version)}</code><br>
<b>Source:</b> <code>{html.escape(source_sha)}</code><br>
<b>Built:</b> {html.escape(build_timestamp)}</p>
<div class="guard"><b>Governance invariant:</b> gg_MATH provides mathematical methods, diagnostics and evidence receipts. Consumer/project authority remains external unless an explicit governed contract states otherwise.</div>
<h2>Release surfaces</h2>
<ul>
<li><a href="status/">Status</a> — exact-SHA build, artifact census and release identity.</li>
<li><a href="methods/">Methods</a> — method cards and typed dependency graph.</li>
<li><a href="research/consolidation/">Consolidation</a> — interactive Plotly synthetic research dashboard.</li>
<li><a href="lm10/w3/">LM-10 W3</a> — synthetic visual/calibration receipt.</li>
<li><a href="grandmission-i-b/">Grandmission I-B</a> — temporal PCA federation compendium.</li>
<li><a href="receipts/">Receipts</a> — machine-readable evidence registry.</li>
<li><a href="qa/">QA</a> — complete-test and coverage summary.</li>
</ul>
<p><a href="release-manifest.json">release-manifest.json</a> · <a href="SHA256SUMS">SHA256SUMS</a></p>
"""
    (site / "index.html").write_text(page("gg_MATH release navigator", home_body, ""), encoding="utf-8")

    for expected in EXPECTED_PAGES:
        if not (site / expected).is_file():
            raise FileNotFoundError(f"expected page not generated: {expected}")

    public_files = sorted(
        p for p in site.rglob("*")
        if p.is_file() and p.name not in {"release-manifest.json", "SHA256SUMS"}
    )
    manifest = {
        "schema": "gg-math-release-manifest/v1",
        "release_version": release_version,
        "repository": "GBOGEB/gg_MATH",
        "source_sha": source_sha,
        "build_timestamp": build_timestamp,
        "workflow_run_url": run_url,
        "authority_transfer": False,
        "formal_credit_delta": 0,
        "engineering_acceptance": False,
        "qps_threshold_authority": False,
        "junit": junit_summary,
        "coverage": coverage_summary,
        "expected_pages": list(EXPECTED_PAGES),
        "receipt_count": len(receipt_index),
        "workflow_count": len(workflow_index),
        "workflow_inventory": workflow_index,
        "files": [
            {
                "path": p.relative_to(site).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            }
            for p in public_files
        ],
    }
    manifest_path = site / "release-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checksum_targets = sorted(p for p in site.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    sums = "\n".join(
        f"{sha256(path)}  {path.relative_to(site).as_posix()}"
        for path in checksum_targets
    ) + "\n"
    (site / "SHA256SUMS").write_text(sums, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--release-version", required=True)
    parser.add_argument("--run-url", default="")
    parser.add_argument("--build-timestamp", required=True)
    parser.add_argument("--junit", type=Path, default=Path("release/test-results.xml"))
    parser.add_argument("--coverage", type=Path, default=Path("release/coverage.xml"))
    args = parser.parse_args()
    manifest = build_site(
        ROOT,
        (ROOT / args.site_dir).resolve(),
        args.source_sha.lower(),
        args.release_version,
        args.run_url,
        args.build_timestamp,
        (ROOT / args.junit).resolve(),
        (ROOT / args.coverage).resolve(),
    )
    print(json.dumps({
        "status": "PASS_RELEASE_SITE_BUILD",
        "source_sha": manifest["source_sha"],
        "files": len(manifest["files"]),
        "receipts": manifest["receipt_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
