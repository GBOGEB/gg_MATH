"""Read back a deployed gg_MATH GitHub Pages site against the admitted build (R5/R9).

The admitted ``site/`` directory (the exact Pages artifact) is the oracle:

1. poll the deployed ``release-manifest.json`` until it is byte-identical to
   the admitted one (bound to the source SHA and release identity);
2. fetch every manifest file plus ``SHA256SUMS`` and compare SHA-256;
3. fetch every published route (``/``, ``/status/`` ...) and compare it with
   the admitted ``index.html``;
4. resolve every local ``href``/``src`` in every deployed HTML page;
5. with ``--browser``: load every route and Plotly product in headless
   Chromium (desktop and mobile), failing on page errors, console errors,
   failed requests or a Plotly product that renders no graph.

Every fetch carries a ``?readback=<sha12>`` query so no CDN entry cached
before this deployment can answer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

ROUTES = (
    "",
    "status/",
    "methods/",
    "research/consolidation/",
    "lm10/w3/",
    "grandmission-i-b/",
    "receipts/",
    "qa/",
    "code-purple-w1/",
)
# Products that must render at least one Plotly graph in a real browser.
PLOTLY_PRODUCTS = ("research/consolidation/dashboard.html",)
VIEWPORTS = {"desktop": {"width": 1366, "height": 900}, "mobile": {"width": 390, "height": 844}}
SCHEMA = "gg-math-pages-readback/v1"
USER_AGENT = "gg-math-release-readback/1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, timeout: float = 30.0) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read() if exc.fp else b""
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        return 0, str(exc).encode()


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        for key, value in attrs:
            if key in {"href", "src"} and value:
                self.links.append(value)


def local_link_targets(page_path: str, body: bytes) -> tuple[list[str], set[str]]:
    """Site-relative paths referenced by one page, plus external hosts."""
    parser = LinkParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    targets, hosts = [], set()
    for raw in parser.links:
        split = urlsplit(raw)
        if raw.startswith(("#", "mailto:", "data:", "javascript:", "blob:")):
            continue
        if split.scheme or split.netloc:
            hosts.add(split.netloc)
            continue
        if not split.path:
            continue
        resolved = urlsplit(urljoin("https://site.invalid/" + page_path, split.path)).path.lstrip("/")
        if resolved == "" or resolved.endswith("/"):
            resolved += "index.html"
        targets.append(resolved)
    return targets, hosts


def wait_for_manifest(base: str, query: str, expected: bytes, wait_seconds: float, interval: float) -> tuple[int, bytes, int]:
    deadline = time.monotonic() + wait_seconds
    attempts = 0
    while True:
        attempts += 1
        status, body = fetch(f"{base}release-manifest.json{query}&attempt={attempts}")
        if status == 200 and body == expected:
            return status, body, attempts
        if time.monotonic() >= deadline:
            return status, body, attempts
        time.sleep(interval)


def browser_qa(base: str, query: str, out_dir: Path) -> tuple[list[dict], list[str]]:
    from playwright.sync_api import sync_playwright

    rows, errors = [], []
    targets = [route for route in ROUTES] + list(PLOTLY_PRODUCTS)
    shots = out_dir / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            for viewport_name, viewport in VIEWPORTS.items():
                context = browser.new_context(viewport=viewport)
                for rel in targets:
                    page = context.new_page()
                    console: list[str] = []
                    page_errors: list[str] = []
                    failed: list[str] = []
                    page.on("console", lambda msg, sink=console: sink.append(msg.text) if msg.type == "error" else None)
                    page.on("pageerror", lambda exc, sink=page_errors: sink.append(str(exc)))
                    page.on("requestfailed", lambda req, sink=failed: sink.append(f"FAILED {req.url}"))
                    page.on(
                        "response",
                        lambda resp, sink=failed: sink.append(f"{resp.status} {resp.url}") if resp.status >= 400 else None,
                    )
                    response = page.goto(base + rel + query, wait_until="load", timeout=90_000)
                    page.wait_for_timeout(2_000)
                    plots = sum(frame.locator(".js-plotly-plot").count() for frame in page.frames)
                    overflow = bool(page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1"))
                    name = (rel.rstrip("/").replace("/", "__") or "home") + f"--{viewport_name}.png"
                    page.screenshot(path=str(shots / name), full_page=False)
                    failed = [item for item in failed if not item.rstrip("/").endswith("favicon.ico")]
                    row = {
                        "path": rel or "/",
                        "viewport": viewport_name,
                        "http_status": response.status if response else 0,
                        "plotly_graphs": plots,
                        "console_errors": console,
                        "page_errors": page_errors,
                        "failed_requests": failed,
                        "horizontal_overflow": overflow,
                        "screenshot": f"screenshots/{name}",
                    }
                    problems = []
                    if row["http_status"] != 200:
                        problems.append(f"HTTP {row['http_status']}")
                    if console:
                        problems.append(f"{len(console)} console errors")
                    if page_errors:
                        problems.append(f"{len(page_errors)} page errors")
                    if failed:
                        problems.append(f"{len(failed)} failed requests")
                    if rel in PLOTLY_PRODUCTS and plots == 0:
                        problems.append("no Plotly graph rendered")
                    row["status"] = "FAIL" if problems else "PASS"
                    if problems:
                        errors.append(f"browser {viewport_name} {rel or '/'}: " + ", ".join(problems))
                    rows.append(row)
                    page.close()
                context.close()
        finally:
            browser.close()
    return rows, errors


def readback(
    base_url: str,
    site: Path,
    source_sha: str,
    release_version: str,
    wait_seconds: float = 600.0,
    interval: float = 10.0,
    browser: bool = False,
    out_dir: Path | None = None,
) -> dict:
    base = base_url.rstrip("/") + "/"
    query = f"?readback={source_sha[:12]}"
    expected_manifest = (site / "release-manifest.json").read_bytes()
    manifest = json.loads(expected_manifest)
    errors: list[str] = []
    if manifest.get("source_sha") != source_sha:
        errors.append("admitted manifest source_sha does not match --source-sha")
    if manifest.get("release_version") != release_version:
        errors.append("admitted manifest release_version does not match --release-version")

    status, deployed_manifest, attempts = wait_for_manifest(base, query, expected_manifest, wait_seconds, interval)
    if status != 200:
        errors.append(f"deployed release-manifest.json HTTP {status} after {attempts} attempts")
    elif deployed_manifest != expected_manifest:
        errors.append(f"deployed release-manifest.json differs from the admitted build after {attempts} attempts")

    expected = {entry["path"]: entry["sha256"] for entry in manifest.get("files", [])}
    for name in ("release-manifest.json", "SHA256SUMS"):
        expected[name] = sha256((site / name).read_bytes())

    def get(path: str) -> tuple[str, int, bytes]:
        code, body = fetch(base + path + query)
        return path, code, body

    with ThreadPoolExecutor(max_workers=8) as pool:
        fetched = {path: (code, body) for path, code, body in pool.map(get, sorted(expected))}
    files = []
    for path in sorted(expected):
        code, body = fetched[path]
        ok = code == 200 and sha256(body) == expected[path]
        files.append({"path": path, "http_status": code, "sha256": sha256(body) if code == 200 else None, "match": ok})
        if not ok:
            errors.append(f"{path}: HTTP {code}" if code != 200 else f"{path}: digest mismatch")

    routes = []
    for route in ROUTES:
        code, body = fetch(base + route + query)
        index = route + "index.html"
        ok = code == 200 and sha256(body) == expected.get(index)
        routes.append({"route": "/" + route, "http_status": code, "match": ok})
        if not ok:
            errors.append(f"route /{route}: HTTP {code}" if code != 200 else f"route /{route}: differs from admitted {index}")

    hosts: set[str] = set()
    broken = []
    for path in sorted(expected):
        if not path.endswith(".html") or fetched[path][0] != 200:
            continue
        targets, page_hosts = local_link_targets(path, fetched[path][1])
        hosts |= page_hosts
        for target in targets:
            if target not in expected:
                broken.append(f"{path} -> {target}")
    errors.extend(f"broken deployed link: {item}" for item in broken)

    browser_rows: list[dict] = []
    if browser:
        browser_rows, browser_errors = browser_qa(base, query, out_dir or Path("."))
        errors.extend(browser_errors)

    return {
        "schema": SCHEMA,
        "status": "FAIL" if errors else "PASS",
        "base_url": base,
        "source_sha": source_sha,
        "release_version": release_version,
        "manifest_sha256": sha256(expected_manifest),
        "manifest_attempts": attempts,
        "files_checked": len(files),
        "files_matched": sum(1 for item in files if item["match"]),
        "routes": routes,
        "external_hosts": sorted(hosts),
        "browser_qa": browser_rows,
        "browser_qa_executed": browser,
        "errors": errors,
        "files": files,
        "authority_transfer": False,
        "formal_credit_delta": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--release-version", required=True)
    parser.add_argument("--wait-seconds", type=float, default=600.0)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--browser", action="store_true", help="also run headless Chromium QA (needs Playwright)")
    parser.add_argument("--out-dir", type=Path, default=Path("release/pages-readback"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = readback(
        args.base_url,
        args.site_dir.resolve(),
        args.source_sha.lower(),
        args.release_version,
        wait_seconds=args.wait_seconds,
        interval=args.interval,
        browser=args.browser,
        out_dir=args.out_dir,
    )
    (args.out_dir / "pages-readback.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "base_url", "files_checked", "files_matched", "external_hosts")}, sort_keys=True))
    for item in result["errors"]:
        print(f"- {item}")
    if result["status"] != "PASS":
        raise SystemExit("FAIL_PAGES_READBACK")
    print("PASS_PAGES_READBACK")


if __name__ == "__main__":
    main()
