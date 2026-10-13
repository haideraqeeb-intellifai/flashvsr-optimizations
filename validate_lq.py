"""Validate streaming semantics and native FP8 support at production dimensions."""
import copy, json, sys, types
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'vendor/FlashVSR/examples/WanVSR'),str(ROOT/'vendor/FlashVSR')]
from diffsynth.vram_management.layers import AutoWrappedModule
from utils.utils import Causal_LQ4x_Proj
from lq_optimizations import install_lq
from lq_fp8 import FP8Conv3d

torch.set_num_threads(8)
torch.manual_seed(2026)
out=ROOT/'outputs/lq-study';out.mkdir(exist_ok=True)
results={}
with torch.no_grad():
 m=Causal_LQ4x_Proj(3,1536,1).cuda().bfloat16()
 m.load_state_dict(torch.load(ROOT/'models/FlashVSR-v1.1/LQ_proj_in.ckpt',map_location='cpu'))
 for name in ('conv1','conv2'):
  wrapper=AutoWrappedModule(getattr(m,name),offload_dtype=torch.bfloat16,offload_device='cpu',onload_dtype=torch.bfloat16,onload_device='cuda',computation_dtype=torch.bfloat16,computation_device='cuda')
  wrapper.onload();setattr(m,name,wrapper)
 original=copy.deepcopy(m)
 pipe=types.SimpleNamespace(dit=types.SimpleNamespace(LQ_proj_in=m))
 _,timer=install_lq(pipe,'baseline',True);timer.enabled=True
 # Production spatial geometry, first streaming frame then two chunks exercises both caches.
 clips=[torch.randn(1,3,t,896,1664,device='cuda',dtype=torch.bfloat16) for t in (1,4,4)]
 m.clear_cache();original.clear_cache()
 refs=[]
 for clip in clips:
  a=original.stream_forward(clip);b=m.stream_forward(clip)
  if a is None: assert b is None
  else:
   assert all(torch.equal(x,y) for x,y in zip(a,b)), 'Instrumentation changed output'
   refs.append(a[0].clone())
  assert all(torch.equal(original.cache[k],m.cache[k]) for k in original.cache)
 torch.cuda.synchronize()
 results['instrumentation']={'bitwise_equal_outputs_and_caches':True,**timer.results()}
 del m,pipe
 # Exact Conv3D 2 padded input and weight dimensions.
 conv=original.conv2.module
 engine=FP8Conv3d(conv)
 x=torch.randn(1,2048,4,58,106,device='cuda',dtype=torch.bfloat16)
 try:
  y=engine(x);torch.cuda.synchronize()
  q,s=engine.quantize(x)
  ref=torch.nn.functional.conv3d(q.float()*s,engine.qweight.float()*engine.weight_scale,
      conv.bias.float(),stride=(2,1,1))
  rel=((ref-y.float()).square().mean()/ref.square().mean()).sqrt().item()
  assert rel<.01 and torch.isfinite(y).all(),rel
  results['fp8_support']=dict(supported=True,input=list(x.shape),weight=list(engine.qweight.shape),
      output=list(y.shape),output_dtype=str(y.dtype),relative_rmse_vs_dequantized_fp32=rel,
      backend='cuDNN native FP8 convolution, no BF16 fallback',cudnn=torch.backends.cudnn.version())
 except Exception as e:
  results['fp8_support']=dict(supported=False,error=repr(e))
 print(json.dumps(results['fp8_support']),flush=True)
 (out/'validation.json').write_text(json.dumps(results,indent=2))
 if not results['fp8_support']['supported']:sys.exit(2)
 del engine,x,y,ref,q
 for mode in ('bf16_layout','fp8_conv2','combined'):
  m=copy.deepcopy(original);pipe=types.SimpleNamespace(dit=types.SimpleNamespace(LQ_proj_in=m))
  _,timer=install_lq(pipe,mode,True);timer.enabled=True;m.clear_cache()
  errors=[]
  for i,clip in enumerate(clips):
   b=m.stream_forward(clip)
   if i:
    ref=refs[i-1].float();err=(b[0].float()-ref)
    errors.append(dict(relative_rmse=(err.square().mean()/ref.square().mean()).sqrt().item(),max_abs=err.abs().max().item()))
    assert torch.isfinite(b[0]).all()
  torch.cuda.synchronize()
  results[mode]=dict(stream_output_errors=errors,**timer.results())
  print(mode,errors,flush=True)
  del m,pipe
 (out/'validation.json').write_text(json.dumps(results,indent=2))
