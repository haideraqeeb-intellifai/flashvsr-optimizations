"""Validate report numbers, artifact hashes, layout evidence and browser behavior."""
import json, os, threading
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from playwright.sync_api import sync_playwright
from benchmark import ROOT,sha256
BASE=ROOT/'outputs/lq-study'
def main():
 data=json.loads((BASE/'report-data.json').read_text())
 assert len(data['studies'])==4
 for s in data['studies']:
  run=json.loads((BASE/s['id']/'results.json').read_text())
  assert run['ffn']['mode']=='nvfp4' and run['decoder']['mode']=='fp8'
  assert len(s['cases'])==6
  assert s['aggregate']['all_integrity_checks_passed']
  for c in s['cases']:
   assert all(c['integrity'].values())
   assert c['attention_backend']['backend']=='headwise_bf16'
   assert c['attention_backend']['remote_calls']==0
   assert sha256(BASE/c['video'])==c['output_sha256']
   assert sha256(ROOT/'references'/f"{c['case']}.mp4")==c['quality']['shivam']['reference_sha256']
  if s['id'] in ('bf16_layout','combined'):
   assert all(x['channels_last_3d'] for x in s['layouts'].values())
  assert s['operations']['conv1']['calls']>s['operations']['conv2']['calls']
 assert data['validation']['instrumentation']['bitwise_equal_outputs_and_caches']
 assert data['validation']['fp8_support']['supported']
 assert data['kernel_evidence']['native_fp8_verified']
 assert data['kernel_evidence']['hardware_counters']['sm__ops_path_tensor_src_fp8_dst_fp32.sum']>0
 class Quiet(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
  def copyfile(self,source,outputfile):
   try:super().copyfile(source,outputfile)
   except (BrokenPipeError,ConnectionResetError):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(ROOT)))
 threading.Thread(target=server.serve_forever,daemon=True).start()
 errors=[];failed=[];checks=[]
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-gpu'])
   page=browser.new_page(viewport={'width':1440,'height':1000})
   page.on('pageerror',lambda e:errors.append(str(e)))
   page.on('response',lambda r:failed.append([r.status,r.url]) if r.status>=400 else None)
   page.goto(f'http://127.0.0.1:{server.server_port}/outputs/lq-study/report.html',wait_until='networkidle')
   assert page.locator('#summary tbody tr').count()==4
   assert page.locator('#operations tbody tr').count()==15
   for ref in ('shivam','stack','heldout_shivam'):
    page.select_option('#reference',ref)
    assert page.locator('#cases tbody tr').count()==(16 if ref=='heldout_shivam' else 24)
    assert 'NaN' not in page.locator('#summary').inner_text()
    for i,s in enumerate(data['studies']):
     row=page.locator('#summary tbody tr').nth(i).inner_text()
     assert f"{s['quality'][ref]['ssim']:.6f}" in row
   for ref in ('shivam','stack'):
    page.select_option('#videoRef',ref)
    for s in data['studies']:
     page.select_option('#variant',s['id'])
     for c in s['cases']:
      page.select_option('#clip',c['case'])
      page.wait_for_function("['left','right'].every(id=>document.getElementById(id).readyState>=1)")
      for video in ('#left','#right'):
       assert abs(page.locator(video).evaluate('(v)=>v.duration')-c['duration_seconds'])<.06
      checks.append(dict(reference=ref,mode=s['id'],case=c['case']))
   page.select_option('#reference','shivam');page.select_option('#clip','female_default')
   page.wait_for_function("['left','right'].every(id=>document.getElementById(id).readyState>=1)")
   page.locator('#play').click();page.wait_for_timeout(800)
   assert page.locator('#right').evaluate('(v)=>!v.paused')
   assert page.locator('#left').evaluate('(v)=>!v.paused')
   page.locator('#play').click()
   assert page.locator('#right').evaluate('(v)=>v.paused')
   page.locator('#reset').click()
   page.wait_for_function("['left','right'].every(id=>document.getElementById(id).currentTime<.1)")
   page.screenshot(path=str(BASE/'desktop-preview.png'),full_page=True)
   page.set_viewport_size({'width':390,'height':844})
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
   page.screenshot(path=str(BASE/'mobile-preview.png'),full_page=True)
   browser.close()
 finally:server.shutdown()
 assert not errors,errors
 assert not failed,failed
 (BASE/'browser-validation.json').write_text(json.dumps(dict(errors=errors,failed_requests=failed,video_checks=checks,controls_passed=True,mobile_overflow=False),indent=2))
 print('PASS: 24 generated video hashes, 48 video pairs, reference selectors, measured tables, playback, mobile layout')
if __name__=='__main__':main()
