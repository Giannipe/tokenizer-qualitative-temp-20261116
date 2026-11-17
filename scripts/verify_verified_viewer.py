#!/usr/bin/env python
"""Smoke-test a complete verified-demo browser on local files, without a server."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def run(out,screenshots):
    metadata=json.loads((out/'provenance/turbovla_afa_manifest.json').read_text())
    episodes=[(ep['episode_index'],ep['length']) for ep in metadata['episodes']]
    first,first_n=episodes[0];last,last_n=episodes[-1]
    screenshots.mkdir(exist_ok=True,parents=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport={'width':1400,'height':1000})
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto((out/'viewer/index.html').resolve().as_uri())
        page.wait_for_function(f"document.getElementById('status').textContent.includes('Frame 0 / {first_n-1}')")
        assert page.locator('#episode').input_value()==str(first)
        modes={}
        for mode in ['overview','attention','routing','query','policies','geometry','representations']:
            page.select_option('#mode',mode)
            modes[mode]=page.locator('#canvas').evaluate('(c)=>[c.width,c.height]')
            assert modes[mode][0]>0 and modes[mode][1]>0
        page.select_option('#slot','15');page.select_option('#camera','1');page.select_option('#episode',str(last))
        page.locator('#frame').evaluate(f"e=>{{e.value='{last_n-1}';e.dispatchEvent(new Event('input'))}}")
        page.wait_for_function(f"document.getElementById('status').textContent.includes('Frame {last_n-1} / {last_n-1}')")
        with page.expect_download() as download:page.click('#download')
        download.value.save_as(str(screenshots/f'{out.name}_download.png'))
        page.select_option('#mode','attention')
        page.screenshot(path=str(screenshots/f'{out.name}_viewer.png'),full_page=True)
        assert not errors,errors
        report=dict(file_protocol=True,all_seven_modes_work=True,modes=modes,episode_indices=[eid for eid,_ in episodes],
            last_episode_last_frame_loaded=True,query15_and_wrist_camera_tested=True,canvas_png_download=True,page_errors=errors)
        (out/'viewer_validation.json').write_text(json.dumps(report,indent=2))
        print(out.name,'browser checks passed',json.dumps(report),flush=True)
        browser.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--screenshots',type=Path,default=Path('/tmp/verified-viewer-tests'))
    args=p.parse_args();run(args.output,args.screenshots)
