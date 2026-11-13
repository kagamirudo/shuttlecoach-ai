"""Exercise the local Streamlit UI in the installed Chrome (no browser download)."""
import os
from pathlib import Path
import sys
import re
import zipfile
import json

from playwright.sync_api import sync_playwright, expect

root=Path(__file__).resolve().parents[1]
selected_run=root/'results'/sys.argv[2] if len(sys.argv)>2 else max(p for p in (root/'results').iterdir() if (p/'analysis.json').exists())
expected=json.loads((selected_run/'coaching.json').read_text()) if (selected_run/'coaching.json').exists() else None
manifest=json.loads((selected_run/'analysis.json').read_text())
expected_duration=manifest['video']['times'][manifest['end_frame_exclusive']-1]-manifest['video']['times'][manifest['start_frame']]+1/manifest['video']['fps']
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",headless=True)
    page=browser.new_page(viewport={"width":1440,"height":1100})
    errors=[]
    page.on("pageerror",lambda error:errors.append(str(error)))
    page.goto(os.environ.get("SHUTTLECOACH_URL", "http://127.0.0.1:8501"),wait_until="domcontentloaded")
    page.get_by_role("heading",name="ShuttleCoach",exact=True).wait_for(timeout=60000)
    page.screenshot(path=str(root/".cache"/"dashboard-initial.png"),full_page=True)
    page.get_by_text("YouTube",exact=True).click()
    expect(page.get_by_role("button",name="Retrieve video")).to_be_disabled()
    page.get_by_text("I’m authorized to download this footage",exact=True).click()
    expect(page.get_by_role("button",name="Retrieve video")).to_be_enabled()
    page.get_by_text("Local file",exact=True).click()
    if len(sys.argv)>1:
        page.get_by_role("textbox",name="Absolute MP4/MOV path").fill(str(Path(sys.argv[1]).resolve()))
        page.get_by_role("button",name="Load local video").click()
        page.get_by_role("heading",name="2 · Mark the rallies").wait_for(timeout=180000)
        expect(page.get_by_role("button",name="Run player tracking")).to_be_disabled()
        page.get_by_text("Exactly two singles players are visible",exact=True).click()
        page.get_by_text("The selected camera view stays fixed without cuts",exact=True).click()
        page.get_by_role("spinbutton",name="Calibration time in source seconds").fill("20")
        page.get_by_role("spinbutton",name="Calibration time in source seconds").press("Enter")
        page.wait_for_timeout(1000)
        component=page.frame_locator('iframe[title="streamlit_image_coordinates.streamlit_image_coordinates"]').first
        # Four well-separated corners, preserving image scaling through the component.
        for index,(x,y) in enumerate([(115,384),(677,701),(1125,299),(796,278)]):
            box=component.locator("#image").bounding_box()
            component.locator("#image").click(position={"x":x*box['width']/1280,"y":y*box['height']/720})
            try:
                page.get_by_text(re.compile(f"{index+1}/4 corners")).wait_for(timeout=15000)
            except Exception:
                page.screenshot(path=str(root/".cache"/"dashboard-failure.png"),full_page=True)
                print(page.locator("body").inner_text())
                raise
        page.get_by_text("I reviewed the projected court lines",exact=True).click()
        page.wait_for_timeout(500)
        expect(page.get_by_role("button",name="Run player tracking")).to_be_enabled()
    page.screenshot(path=str(root/".cache"/"dashboard-browser.png"),full_page=True)
    page.get_by_role("tab",name="Review & validate").click()
    if len(sys.argv)>2:
        page.get_by_role("combobox",name="Analysis",exact=True).click()
        page.get_by_role("option",name=sys.argv[2],exact=True).click()
        page.get_by_role("heading",name="Phase 2 · Birdie tracking & evidence report",exact=True).wait_for()
        expect(page.get_by_role("radiogroup",name="Playback",exact=True)).to_have_count(1)
    if page.get_by_role("heading",name=re.compile("^Validation:")).count():
        page.get_by_role("heading",name=re.compile("^Validation:")).wait_for()
        page.wait_for_function("""() => {const v=[...document.querySelectorAll('video')].find(v=>v.offsetParent!==null); return v && v.readyState>=2 && v.duration>0;}""",timeout=30000)
        page.locator("video:visible").first.evaluate("v=>{v.muted=true;v.currentTime=1;return v.play()}")
        page.wait_for_function("""() => [...document.querySelectorAll('video')].some(v=>v.offsetParent!==null && v.currentTime>1.2)""",timeout=15000)
        page.locator("video:visible").first.evaluate("v=>v.pause()")
        page.screenshot(path=str(root/".cache"/"review-browser.png"),full_page=True)
        page.get_by_text("Full source video",exact=True).click()
        page.wait_for_function("""() => {const v=[...document.querySelectorAll('video')].find(v=>v.offsetParent!==null);return v && v.duration>1700;}""",timeout=60000)
        page.screenshot(path=str(root/".cache"/"full-video-browser.png"),full_page=True)
        page.get_by_text("Analyzed clip",exact=True).click()
        if page.get_by_text("Birdie tracking",exact=True).count():
            page.get_by_text("Birdie tracking",exact=True).click()
            page.wait_for_function("""duration => {const v=[...document.querySelectorAll('video')].find(v=>v.offsetParent!==null); return v && v.readyState>=2 && Math.abs(v.duration-duration)<.2;}""",arg=expected_duration,timeout=30000)
            page.locator("video:visible").first.evaluate("v=>{v.muted=true;v.currentTime=160;return v.play()}")
            page.wait_for_function("""() => [...document.querySelectorAll('video')].some(v=>v.offsetParent!==null && v.currentTime>160.2)""",timeout=15000)
            page.locator("video:visible").first.evaluate("v=>v.pause()")
            if expected['gate']=='blocked':expect(page.get_by_text(re.compile('^Coaching withheld:'))).to_be_visible()
            else:expect(page.get_by_text(re.compile('^Both tracking gates passed'))).to_be_visible()
            if page.get_by_text("Birdie court trajectory",exact=True).count():
                expect(page.get_by_text(re.compile("^(The green zone includes|Detection is restricted)"))).to_be_visible()
                with page.expect_download() as projection:
                    page.get_by_role("button",name="Download birdie court trajectory",exact=True).click()
                projection.value.save_as(root/".cache"/"browser-birdie-projection.png")
                assert (root/".cache"/"browser-birdie-projection.png").stat().st_size>1000
            page.screenshot(path=str(root/".cache"/"phase2-browser.png"),full_page=True)
            with page.expect_download() as phase2_report:
                page.get_by_role("button",name="Download Phase 2 report",exact=True).click()
            phase2_report.value.save_as(root/".cache"/"browser-phase2-report.md")
            assert f"**{expected['gate'].upper()}**" in (root/".cache"/"browser-phase2-report.md").read_text()
            page.get_by_text("Analyzed clip",exact=True).click()
        with page.expect_download() as report:
            page.get_by_role("button",name="Download report",exact=True).click()
        report.value.save_as(root/".cache"/"browser-report.md")
        page.get_by_role("button",name="Prepare analysis ZIP",exact=True).click()
        with page.expect_download() as exported:
            page.get_by_role("button",name="Download analysis ZIP",exact=True).click()
        archive=root/"results"/"phase2-validated-prototype.zip"
        exported.value.save_as(archive)
        with zipfile.ZipFile(archive) as bundle:
            assert {"report.md","observations.csv","overlay.mp4","screening.json","analysis.json"}<=set(bundle.namelist())
            if page.get_by_text("Birdie tracking",exact=True).count():
                assert {"coaching_report.md","coaching.json","shuttle.csv","shuttle_overlay.mp4","shuttle_validation.json"}<=set(bundle.namelist())
                report=json.loads(bundle.read('coaching.json'))
                assert report['gate']==expected['gate'] and report['coaching_enabled']==expected['coaching_enabled']
                if page.get_by_text("Birdie court trajectory",exact=True).count():
                    assert {"shuttle_court_projection.png","shuttle_zone_preview.png"}<=set(bundle.namelist())
    assert not errors,errors
    print("Browser checks passed: source controls, calibration gate, playback/seek and report/ZIP exports; Phase 2 checked:",bool(page.get_by_text("Birdie tracking",exact=True).count()))
    browser.close()
