#!/usr/bin/env python3
"""Chromium/Playwright exact-file visual QA for generated Plotly report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VIEWPORTS = [("desktop", 1440, 1000), ("laptop", 1280, 800), ("mobile", 390, 844)]


def expected_card_count(*, observed: bool) -> int:
    """Source-owned 6-card synthetic and 9-card observed contracts."""
    return 9 if observed else 6


def assert_visible_card_count(page, expected: int, phase: str) -> int:
    """Fail closed on both missing and surplus visible charts."""
    actual = page.locator(".metric-card:visible").count()
    if actual != expected:
        raise AssertionError(
            f"G3P {phase}: visible metric cards {actual}, expected {expected}"
        )
    return actual


def run(html_path, outdir, *, observed=False):
    html_path = Path(html_path).resolve()
    outdir = Path(outdir)
    if not html_path.is_file():
        raise ValueError("rendered HTML missing")
    outdir.mkdir(parents=True, exist_ok=True)
    observations = []
    expected = expected_card_count(observed=observed)
    # Import lazily: pure cardinality guards can be unit tested without Chromium.
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        for name, width, height in VIEWPORTS:
            page = browser.new_page(viewport={"width":width, "height":height},
                                    device_scale_factor=1)
            errors = []
            page.on("pageerror",lambda error: errors.append("PAGE:"+str(error)))
            page.on("console",lambda msg: errors.append("CONSOLE:"+msg.text) if msg.type=="error" else None)
            page.goto(html_path.as_uri(),wait_until="load",timeout=120000)
            page.wait_for_function(
                f"() => document.querySelectorAll('.metric-card .main-svg').length >= {expected}",
                timeout=120000,
            )
            assert page.title()=="gg_MATH | G3P Convergence Observatory"
            assert page.locator(".metric-card").count()==expected
            assert_visible_card_count(page, expected, "initial render")
            assert page.locator("[data-filter]").count()==5
            assert page.locator("[data-zoom]").count()==expected
            assert page.locator(".tag").inner_text().find("NOT LIVE")!=-1
            geometry=page.evaluate("""() => ({
                document:document.documentElement.scrollWidth,
                viewport:window.innerWidth,
                broken:Array.from(document.images).filter(x => !x.complete || x.naturalWidth === 0).length,
                charts:document.querySelectorAll('.metric-card .main-svg').length
            })""")
            assert geometry["document"]<=geometry["viewport"]+1,geometry
            assert geometry["broken"]==0,geometry
            page.locator('[data-filter="Blockers"]').click()
            assert page.locator(".metric-card:visible").count()==1
            assert page.locator(".metric-card:visible").first.get_attribute("data-group")=="Blockers"
            page.locator('[data-filter="All"]').click()
            after_reset = assert_visible_card_count(page, expected, "filter reset")
            page.locator('[data-zoom="obs-black"]' if observed else '[data-zoom="plateau"]').click()
            assert page.locator("#detail").evaluate("(x) => x.open") is True
            page.wait_for_selector("#detailChart .main-svg",timeout=20000)
            assert ("obs-black" if observed else "plateau") in page.locator("#detailTitle").inner_text()
            page.locator("#closeDetail").click()
            assert page.locator("#detail").evaluate("(x) => x.open") is False
            page.screenshot(path=str(outdir/(name+".png")),full_page=True,timeout=60000)
            assert not errors,errors
            observations.append({"viewport":name,"width":width,"height":height,
                                 "page_errors":0,"console_errors":0,"body_horizontal_overflow":0,
                                 "broken_images":0,"inlay_plots":geometry["charts"],
                                 "filter_toggle":"PASS","filter_reset_visible_cards":after_reset,
                                 "expected_visible_cards":expected,"zoom_and_close":"PASS",
                                 "screenshot":name+".png"})
            page.close()
        browser.close()
    result={"schema":"gg_math.convergence_html_qa.v1",
            "evidence_class":"LOCAL_OR_NATIVE_EXACT_SOURCE_REQUIRED",
            "source_html":str(html_path),"checks":observations,
            "synthetic":not observed,"source_bound_unverified":observed,"authority_transfer":False,"merge_admission":"HOLD"}
    (outdir/"playwright-receipt.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n", encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--html",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--observed", action="store_true")
    args=p.parse_args()
    print(json.dumps(run(args.html,args.out_dir,observed=args.observed),sort_keys=True))


if __name__=="__main__":
    main()
