"""Check cropped-court calibration using known synthetic geometry in installed Chrome."""
import os
import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from playwright.sync_api import expect, sync_playwright

from shuttlecoach.calibration import LANDMARKS, fit_calibration
from shuttlecoach.media import probe
from shuttlecoach.upstream import court, viz

root = Path(__file__).resolve().parents[1]
source = root / '.cache/cropped-court-test.mp4'
corners = np.array([[-200.,850.],[1480.,850.],[1020.,150.],[260.,150.]])
h = court.compute_homography(corners)
image = np.full((720,1280,3), (55,105,55), np.uint8)
for a,b in viz._segments():
    xy = court.image_to_court(np.array([a,b]),np.linalg.inv(h)).round().astype(int)
    cv2.line(image,tuple(xy[0]),tuple(xy[1]),(240,240,240),3,cv2.LINE_AA)
with av.open(str(source),'w') as out:
    stream=out.add_stream('libx264',rate=10);stream.width=1280;stream.height=720;stream.pix_fmt='yuv420p'
    for i in range(3):
        frame=av.VideoFrame.from_ndarray(image,format='rgb24');frame.pts=i;frame.time_base=Fraction(1,10)
        for packet in stream.encode(frame):out.mux(packet)
    for packet in stream.encode():out.mux(packet)
names=[f'{row} × {side} singles sideline' for row in ['Near short service','Far short service'] for side in ['left','right']]
names+=['Far baseline × centre line','Far doubles long service × centre line']
points=court.image_to_court(np.array([LANDMARKS[n] for n in names]),np.linalg.inv(h))
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(os.environ.get('SHUTTLECOACH_URL', 'http://127.0.0.1:8501'));page.get_by_role('textbox',name='Absolute MP4/MOV path').fill(str(source));page.get_by_role('button',name='Load local video',exact=True).click()
    page.get_by_text('Visible line intersections',exact=True).click()
    component=page.frame_locator('iframe[title="streamlit_image_coordinates.streamlit_image_coordinates"]').first
    def settled():
        page.wait_for_function("() => document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state') === 'notRunning'", timeout=60000)
    settled()
    for i,(name,point) in enumerate(zip(names,points),1):
        settled()
        page.get_by_role('combobox',name='Intersection to mark',exact=True).fill(name)
        page.get_by_role('combobox',name='Intersection to mark',exact=True).press('ArrowDown')
        page.get_by_role('combobox',name='Intersection to mark',exact=True).press('Enter')
        settled()
        image_el=component.locator('#image');box=image_el.bounding_box()
        assert box['x']>=0 and box['x']+box['width']<=page.viewport_size['width'],box
        image_el.click(position={'x':float(point[0]*box['width']/1280),'y':float(point[1]*box['height']/720)})
        expect(page.get_by_text(f'{i} intersections marked',exact=False)).to_be_visible()
        settled()
        print('Marked',i,name,flush=True)
        if i==3:expect(page.get_by_role('checkbox',name='I reviewed the projected court lines',exact=True)).to_be_disabled()
        # Also test coordinates after responsive resizing, with existing points retained.
        if i==4:page.set_viewport_size({'width':900,'height':1100})
    checkbox=page.get_by_role('checkbox',name='I reviewed the projected court lines',exact=True)
    expect(checkbox).to_be_enabled()
    page.get_by_text('I reviewed the projected court lines',exact=True).click()
    expect(checkbox).to_be_checked()
    # Checkbox display updates before the server rebuilds the downloadable JSON.
    page.wait_for_timeout(1000)
    settled()
    with page.expect_download() as download:page.get_by_role('button',name='Download reviewed configuration',exact=True).click()
    target=root/'.cache/browser-intersection-calibration.json';download.value.save_as(target)
    saved=json.loads(target.read_text());cal=saved['calibration'];assert cal['mode']=='intersections' and cal['reviewed']
    assert len(cal['points'])==6
    np.testing.assert_allclose([[p['image_x'],p['image_y']] for p in cal['points']],points,atol=3.)
    fitted,actual_corners=fit_calibration(cal,probe(source))
    assert (actual_corners[:2,1]>720).all()
    np.testing.assert_allclose(court.image_to_court(points,fitted),np.array([LANDMARKS[n] for n in names]),atol=.05)
    page.screenshot(path=str(root/'.cache/intersection-calibration-browser.png'),full_page=True)
    component.locator('#image').screenshot(path=str(root/'.cache/intersection-preview-browser.png'))
    page.locator('[data-testid=stImage]').first.screenshot(path=str(root/'.cache/intersection-reference-browser.png'))
    page.get_by_role('button',name='Undo intersection',exact=True).click()
    expect(page.get_by_text('5 intersections marked',exact=False)).to_be_visible();expect(checkbox).not_to_be_checked()
    page.get_by_role('button',name='Clear calibration',exact=True).click()
    expect(page.get_by_text('0 intersections marked',exact=False)).to_be_visible();expect(checkbox).to_be_disabled()
    # Switching methods starts a fresh calibration, preserving the original workflow.
    page.get_by_text('Four outer corners',exact=True).click();expect(page.get_by_text('0/4 corners',exact=False)).to_be_visible()
    for i,point in enumerate([(100,650),(1100,650),(1000,100),(200,100)],1):
        image_el=component.locator('#image');box=image_el.bounding_box()
        image_el.click(position={'x':point[0]*box['width']/1280,'y':point[1]*box['height']/720})
        expect(page.get_by_text(f'{i}/4 corners',exact=False)).to_be_visible();settled()
    expect(checkbox).to_be_enabled()
    assert not errors,errors
    browser.close()
print('Cropped-court browser checks passed: six intersections, responsive coordinates, off-screen corners, export, undo, reset, and method switching.')
