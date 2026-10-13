"""Full-video fidelity plus sampled perceptual/temporal drift versus saved baseline."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np
from PIL import Image
from benchmark import ROOT, write_json

os.environ.setdefault('TORCH_HOME',str(ROOT/'.cache/torch'))

def compare(ref, candidate, log):
 p=subprocess.run(['ffmpeg','-hide_banner','-nostats','-threads','4','-i',str(ref),
   '-threads','4','-i',str(candidate),'-filter_complex_threads','4','-filter_complex',
   '[0:v]split[a][b];[1:v]split[c][d];[a][c]ssim[s];[b][d]psnr[p]',
   '-map','[s]','-map','[p]','-an','-f','null','-'],capture_output=True,text=True,check=True)
 log.write_text(p.stderr)
 s=re.search(r'SSIM .*?All:([0-9.]+)',p.stderr)
 n=re.search(r'PSNR .*?average:([0-9.]+|inf)',p.stderr)
 if not s or not n: raise RuntimeError('Failed to parse metrics')
 return {'ssim':float(s[1]),'psnr_db':float(n[1]) if n[1]!='inf' else None}

def decode_samples(path, nframes):
 indices=sorted(set(int(i) for i in np.linspace(0,nframes-2,24)))
 wanted=sorted(set(indices+[i+1 for i in indices]))
 expr='+'.join(f'eq(n\\,{i})' for i in wanted)
 data=subprocess.check_output(['ffmpeg','-v','error','-threads','4','-i',str(path),
   '-vf',f'select={expr},scale=416:224:flags=bicubic','-vsync','0','-f','rawvideo','-pix_fmt','rgb24','-'])
 frames=np.frombuffer(data,dtype=np.uint8).reshape(-1,224,416,3).copy()
 assert len(frames)==len(wanted),(len(frames),len(wanted))
 return frames,indices,wanted

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
 parser.add_argument('--reference',type=Path,default=ROOT/'outputs/baseline')
 args=parser.parse_args()
 run=args.run.resolve(); results=json.loads((run/'results.json').read_text())
 import torch,lpips
 torch.set_num_threads(4)
 model=lpips.LPIPS(net='alex',verbose=False).eval()
 comparisons=[]
 for case in results['cases']:
  ref=args.reference.resolve()/case['case']/'enhanced.mp4'; candidate=run/case['output']
  dest=run/case['case']; cache=dest/'quality.json'
  if cache.exists():
   cached=json.loads(cache.read_text())
   expected=json.loads((ref.parent/'metrics.json').read_text())['output_sha256']
   if cached['reference_sha256']==expected:
    comparisons.append(cached);continue
  metrics=compare(ref,candidate,dest/'quality-ffmpeg.log')
  r,indices,wanted=decode_samples(ref,case['frames']); c,_,_=decode_samples(candidate,case['frames'])
  positions=[wanted.index(i) for i in indices]
  def tensor(a):return torch.from_numpy(a).permute(0,3,1,2).float()/127.5-1
  values=[]
  with torch.no_grad():
   for start in range(0,len(positions),4):
    select=positions[start:start+4]
    values.extend(model(tensor(r[select]),tensor(c[select])).flatten().tolist())
  temporal=[]
  for i in indices:
   a,b=wanted.index(i),wanted.index(i+1)
   rd=r[b].astype(np.float32)-r[a].astype(np.float32)
   cd=c[b].astype(np.float32)-c[a].astype(np.float32)
   temporal.append(float(np.abs(rd-cd).mean()/255))
  # Three equally spaced, identical frame indices for visual comparison.
  for index in (positions[0],positions[len(positions)//2],positions[-1]):
   Image.fromarray(r[index]).save(dest/f'baseline-frame-{wanted[index]}.jpg',quality=95)
   Image.fromarray(c[index]).save(dest/f'variant-frame-{wanted[index]}.jpg',quality=95)
  metrics.update(case=case['case'],lpips_mean=float(np.mean(values)),lpips_max=float(np.max(values)),
    lpips_frames=indices,temporal_delta_mae_mean=float(np.mean(temporal)),
    sample_resolution=[416,224],reference_sha256=json.loads((ref.parent/'metrics.json').read_text())['output_sha256'])
  comparisons.append(metrics);write_json(cache,metrics);print(json.dumps(metrics),flush=True)
 write_json(run/'quality.json',{'reference':str(args.reference.resolve())+'; no ground-truth claim',
  'full_video_metrics':['SSIM','PSNR'], 'sampled_metrics':'LPIPS AlexNet: 24 frames at 416x224; temporal delta MAE: 24 adjacent frame pairs, /255',
  'cases':comparisons})

if __name__=='__main__':main()
