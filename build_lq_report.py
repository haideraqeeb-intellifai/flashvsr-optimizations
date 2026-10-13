"""Build the requested standalone Conv3D experiment report from measured artifacts."""
import csv, html, json, statistics
from pathlib import Path
from benchmark import ROOT,sha256
BASE=ROOT/'outputs/lq-study'
MODES=('baseline','bf16_layout','fp8_conv2','combined')
LABELS=dict(baseline='Requested stack control',bf16_layout='② BF16 tuning + 3D channels-last',fp8_conv2='③ FP8 Conv3D 2',combined='② + ③ combined')
read=lambda p:json.loads(p.read_text())
def main():
 quality=read(BASE/'quality.json');studies=[]
 for mode in MODES:
  run=read(BASE/mode/'results.json');assert run['status']=='completed'
  profiles=[read(BASE/'profiles'/f'{mode}-{i}.json') for i in (1,2,3)]
  ops={key:dict(median=statistics.median(p['lq_operations']['operations'][key]['seconds'] for p in profiles),
      minimum=min(p['lq_operations']['operations'][key]['seconds'] for p in profiles),
      maximum=max(p['lq_operations']['operations'][key]['seconds'] for p in profiles),
      calls=profiles[0]['lq_operations']['operations'][key]['calls']) for key in profiles[0]['lq_operations']['operations']}
  rows=[]
  for case in run['cases']:
   q={r['reference']:r for r in quality['cases'] if r['mode']==mode and r['case']==case['case']}
   rows.append({**case,'quality':q,'video':f"{mode}/{case['output']}"})
  stats={ref:dict(psnr_db=statistics.mean(r['quality'][ref]['psnr_db'] for r in rows) if all(r['quality'][ref]['psnr_db'] is not None for r in rows) else None,
                  ssim=statistics.mean(r['quality'][ref]['ssim'] for r in rows)) for ref in ('shivam','stack')}
  held=[r for r in rows if r['case'] not in ('female_default','male_existing')]
  stats['heldout_shivam']=dict(psnr_db=statistics.mean(r['quality']['shivam']['psnr_db'] for r in held),ssim=statistics.mean(r['quality']['shivam']['ssim'] for r in held))
  studies.append(dict(id=mode,label=LABELS[mode],cases=rows,quality=stats,operations=ops,aggregate=run['aggregate'],
      peak_allocated=max(c['peak_allocated_vram_gib'] for c in rows),lq=run['lq'],
      profile_wall_seconds=[p['inference_wall_seconds'] for p in profiles],
      layouts=profiles[0]['lq_operations']['layouts']))
 data=dict(studies=studies,validation=read(BASE/'validation.json'),method=quality['method'],
      kernel_evidence=read(BASE/'fp8-kernel-evidence.json') if (BASE/'fp8-kernel-evidence.json').exists() else None,
      source_hashes={p.name:sha256(p) for p in (BASE/'source').glob('*') if p.is_file()},
      published_videos={c['case']:c['flashvsr_v11']['url'] for c in read(ROOT/'provenance/benchmark_manifest.json')['cases']},
      published_source='https://s3.renderplatform.com/user-assets/preview/agent/flashvsr-six-realistic-20260917-35a25def6d/report_long.html')
 (BASE/'report-data.json').write_text(json.dumps(data,indent=2))
 with (BASE/'measurements.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['configuration','case','inference_seconds','PSNR_vs_Shivam_dB','SSIM_vs_Shivam','PSNR_vs_stack_dB','SSIM_vs_stack'])
  for s in studies:
   for c in s['cases']:w.writerow([s['id'],c['case'],c['inference_seconds'],c['quality']['shivam']['psnr_db'],c['quality']['shivam']['ssim'],c['quality']['stack']['psnr_db'],c['quality']['stack']['ssim']])
 template=(ROOT/'lq_report_template.html').read_text()
 (BASE/'report.html').write_text(template.replace('__DATA__',json.dumps(data).replace('</','<\\/')))
 print(BASE/'report.html')
if __name__=='__main__':main()
