"""Full-resolution, all-frame PSNR/SSIM against verified Shivam and stack controls."""
import concurrent.futures
import json
from pathlib import Path
from analyze_attention import compare
from benchmark import ROOT, sha256, probe, video_stream, write_json

BASE=ROOT/'outputs/lq-study'
MODES=('baseline','bf16_layout','fp8_conv2','combined')
def main():
 originals={c['case']:c for c in json.loads((ROOT/'provenance/benchmark_manifest.json').read_text())['cases']}
 control=json.loads((BASE/'baseline/results.json').read_text())
 control_cases={c['case']:c for c in control['cases']}
 tasks=[]
 for mode in MODES:
  results=json.loads((BASE/mode/'results.json').read_text())
  assert results['status']=='completed' and len(results['cases'])==6
  for case in results['cases']:
   candidate=BASE/mode/case['output']
   assert sha256(candidate)==case['output_sha256']
   st=video_stream(probe(candidate))
   assert (st['width'],st['height'],int(st['nb_frames']))==(1664,896,case['frames'])
   for refid in ('shivam','stack'):
    if refid=='shivam':
     ref=ROOT/'references'/f"{case['case']}.mp4";expected=originals[case['case']]['flashvsr_v11']['sha256']
    else:
     ref=BASE/'baseline'/control_cases[case['case']]['output'];expected=control_cases[case['case']]['output_sha256']
    assert sha256(ref)==expected
    rst=video_stream(probe(ref))
    assert (rst['width'],rst['height'],int(rst['nb_frames']),rst['r_frame_rate'])==(st['width'],st['height'],int(st['nb_frames']),st['r_frame_rate'])
    tasks.append((mode,case,refid,ref,candidate,expected))
 def score(task):
  mode,case,refid,ref,candidate,expected=task
  dest=BASE/mode/case['case']/f'quality-{refid}.json'
  if dest.exists():
   old=json.loads(dest.read_text())
   if old['reference_sha256']==expected and old['candidate_sha256']==case['output_sha256']:return old
  metrics=({'psnr_db':None,'ssim':1.0} if ref==candidate else compare(ref,candidate,dest.with_suffix('.log')))
  row=dict(mode=mode,case=case['case'],reference=refid,frames=case['frames'],reference_sha256=expected,
           candidate_sha256=case['output_sha256'],**metrics)
  write_json(dest,row);print(json.dumps(row),flush=True)
  return row
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(score,tasks))
 write_json(BASE/'quality.json',dict(method='FFmpeg SSIM All and PSNR average over all full-resolution decoded YUV420 frames. Equal clip weighting; PSNR means in dB. Null PSNR denotes infinity for identical output. Fidelity, not ground-truth restoration quality.',cases=rows))
if __name__=='__main__':main()
