"""Run FlashVSR in the LTX worker, retaining a CPU master of its model."""

import contextlib
import gc
import os
from pathlib import Path
import sys
import threading
import time
import types

from . import runner
from .runtime import cache_root


class InProcessRunner:
    def __init__(self):
        self.lock = threading.Lock()
        self.preloaded = None
        self.model_cache = {}
        self.cpu_master = None

    def _load(self):
        # ComfyUI and the official example both import a top-level `utils`.
        previous_path = sys.path[:]
        previous_utils = {key: value for key, value in sys.modules.items()
                          if key == "utils" or key.startswith("utils.")}
        for key in previous_utils:
            del sys.modules[key]
        package = types.ModuleType("utils")
        package.__path__ = [str(cache_root() / "upstream/examples/WanVSR/utils")]
        sys.modules["utils"] = package
        try:
            self.preloaded = runner.load_official(cache_root())
        finally:
            sys.path[:] = previous_path
            for key in tuple(sys.modules):
                if key == "utils" or key.startswith("utils."):
                    del sys.modules[key]
            sys.modules.update(previous_utils)

    def _capture_cpu_master(self):
        pipe = self.model_cache["pipe"]
        modules = []
        for module in pipe.modules():
            parameters = {name: value.detach() for name, value in module._parameters.items()
                          if value is not None}
            buffers = {name: value.detach() for name, value in module._buffers.items()
                       if value is not None}
            if any(value.device.type != "cpu" for value in (*parameters.values(), *buffers.values())):
                raise RuntimeError("FlashVSR CPU master contains a GPU tensor")
            modules.append((module, parameters, buffers))
        dit = pipe.denoising_model()
        context = [(block.cross_attn, block.cross_attn.cache_k, block.cross_attn.cache_v)
                   for block in dit.blocks]
        prompt_context = pipe.prompt_emb_posi["context"]
        extras = {name: getattr(pipe, name) for name in ("timestep", "t", "t_mod")}
        if any(tensor.device.type != "cpu" for _, key, value in context
               for tensor in (key, value)):
            raise RuntimeError("FlashVSR CPU master contains a GPU attention cache")
        self.cpu_master = (modules, context, prompt_context, extras)

    def _clear_transient_gpu_state(self, pipe):
        from .fast_inference.conv2d_fp8 import FP8Conv2d
        from .fast_inference.conv3d_fp8 import FP8Conv3d

        pipe.denoising_model().LQ_proj_in.clear_cache()
        pipe.TCDecoder.clean_mem()
        for module in pipe.modules():
            if isinstance(module, FP8Conv2d):
                module.cache.graphs.clear()
                module.cache.workspace = None
            elif isinstance(module, FP8Conv3d):
                module.graphs.clear()
                module.workspace = None

    def _release_gpu_weights(self, pipe):
        modules, context, prompt_context, extras = self.cpu_master
        self._clear_transient_gpu_state(pipe)
        torch = self.preloaded[0].torch
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for module, parameters, buffers in modules:
                if set(parameters) != {key for key, value in module._parameters.items() if value is not None}:
                    raise RuntimeError("FlashVSR parameter layout changed during inference")
                if set(buffers) != {key for key, value in module._buffers.items() if value is not None}:
                    raise RuntimeError("FlashVSR buffer layout changed during inference")
                for name, cpu_tensor in parameters.items():
                    parameter = module._parameters[name]
                    if parameter.shape != cpu_tensor.shape or parameter.dtype != cpu_tensor.dtype:
                        raise RuntimeError(f"FlashVSR parameter changed: {name}")
                    parameter.data = cpu_tensor
                for name, cpu_tensor in buffers.items():
                    current = module._buffers[name]
                    if current.shape != cpu_tensor.shape or current.dtype != cpu_tensor.dtype:
                        raise RuntimeError(f"FlashVSR buffer changed: {name}")
                    module._buffers[name] = cpu_tensor
                if hasattr(module, "offload_device") and hasattr(module, "state"):
                    module.state = 0
            for attention, cache_k, cache_v in context:
                attention.cache_k = cache_k
                attention.cache_v = cache_v
            pipe.prompt_emb_posi["context"] = prompt_context
            for name, value in extras.items():
                setattr(pipe, name, value)
        print(f"CPU_MASTER release_weights_seconds={time.perf_counter() - start:.6f}", flush=True)

    def run(self, arguments, log_path):
        with self.lock:
            original_cwd, original_argv = Path.cwd(), sys.argv
            with Path(log_path).open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    if self.preloaded is None:
                        self._load()
                    fast = "--fast-inference" in arguments
                    if self.model_cache.get("fast") != fast:
                        self.cpu_master = None
                    sys.argv = [str(Path(runner.__file__).resolve()), "--cache", str(cache_root()), *arguments]
                    runner.main(preloaded=self.preloaded, model_cache=self.model_cache)
                    pipe = self.model_cache["pipe"]
                    if self.cpu_master is None:
                        self._clear_transient_gpu_state(pipe)
                        pipe.load_models_to_device([])
                        pipe.cpu()
                        runner.move_cached_context(pipe, "cpu")
                        self._capture_cpu_master()
                    else:
                        self._release_gpu_weights(pipe)
                    del pipe
                    gc.collect()
                    torch = self.preloaded[0].torch
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    print(f"CPU_MASTER gpu_allocated_bytes={torch.cuda.memory_allocated()} "
                          f"gpu_reserved_bytes={torch.cuda.memory_reserved()}", flush=True)
                except BaseException:
                    # A failed request must never leave a partly restored model
                    # available to the next request or keep its GPU weights alive.
                    self.cpu_master = None
                    pipe = self.model_cache.pop("pipe", None)
                    self.model_cache.clear()
                    if pipe is not None:
                        try:
                            self._clear_transient_gpu_state(pipe)
                            pipe.load_models_to_device([])
                            pipe.cpu()
                            runner.move_cached_context(pipe, "cpu")
                        finally:
                            del pipe
                    gc.collect()
                    if self.preloaded is not None:
                        self.preloaded[0].torch.cuda.empty_cache()
                    raise
                finally:
                    os.chdir(original_cwd)
                    sys.argv = original_argv


_runner = InProcessRunner()


def run_official(arguments, log_path):
    return _runner.run(arguments, log_path)
