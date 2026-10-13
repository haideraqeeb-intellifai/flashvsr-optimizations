"""Bounded, asynchronous input preparation for FlashVSR's ordered LQ slices."""

from collections import deque
import queue
import threading
import time

import imageio
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class ChunkedLQ:
    """Act like the upstream CPU LQ tensor for the slices Tiny Long requests.

    In PIL mode the producer applies the exact upstream bicubic crop. In GPU
    modes it queues small decoded frames for GPU bicubic resizing on demand.
    It stays at most ``lookahead`` frames ahead of the consumer. The last 32
    consumed frames cover the encoder/decoder overlap at each model step.
    """

    def __init__(self, path, content_frames, total_frames, *, scale=4, dtype=torch.bfloat16,
                 lookahead=32, resize_mode="pil"):
        if lookahead < 8 or total_frames < content_frames or content_frames < 1:
            raise ValueError("Invalid frame counts or lookahead")
        if scale != 4:
            raise ValueError("This benchmark's chunked input requires scale=4")
        if resize_mode not in ("pil", "gpu", "gpu_exact"):
            raise ValueError("resize_mode must be pil, gpu, or gpu_exact")
        self.path = str(path)
        self.content_frames = content_frames
        self.total_frames = total_frames
        self.scale = scale
        self.dtype = dtype
        self.resize_mode = resize_mode
        self.queue = queue.Queue(maxsize=lookahead)
        self.cache = {}
        self.order = deque()
        self.next_index = 0
        self.producer_seconds = 0.0
        self.wait_seconds = 0.0
        self.error = None
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self._produce, name="flashvsr-input", daemon=True)
        self.worker.start()
        # Geometry is fixed for this study's 426x240 staged videos.
        self.height, self.width = 896, 1664

    def _produce(self):
        reader = None
        try:
            reader = imageio.get_reader(self.path)
            last = None
            for i in range(self.total_frames):
                if self.stop.is_set():
                    break
                started = time.perf_counter()
                if i < self.content_frames:
                    last = Image.fromarray(reader.get_data(i)).convert("RGB")
                    if last.size != (426, 240):
                        raise ValueError(f"Expected 426x240 staged input, got {last.size}")
                if self.resize_mode == "pil":
                    frame = last.resize((1704, 960), Image.BICUBIC).crop((20, 32, 1684, 928))
                    pixels = torch.from_numpy(np.asarray(frame, np.uint8)).to(dtype=torch.float32)
                    pixels = pixels.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
                    tensor = pixels.to(self.dtype)
                else:
                    tensor = torch.from_numpy(np.array(last, dtype=np.uint8, copy=True))
                self.producer_seconds += time.perf_counter() - started
                while not self.stop.is_set():
                    try:
                        self.queue.put((i, tensor), timeout=0.1)
                        break
                    except queue.Full:
                        pass
        except BaseException as exc:
            self.error = exc
        finally:
            if reader is not None:
                reader.close()

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 5 or not isinstance(key[2], slice):
            raise TypeError("Expected a 5D temporal slice")
        start, stop, step = key[2].indices(self.total_frames)
        if step != 1 or stop <= start:
            raise ValueError("Expected a nonempty forward temporal slice")
        waiting = time.perf_counter()
        while self.next_index < stop:
            try:
                index, tensor = self.queue.get(timeout=0.1)
            except queue.Empty:
                if self.error is not None:
                    raise RuntimeError("Frame producer failed") from self.error
                if not self.worker.is_alive():
                    raise RuntimeError("Frame producer ended before requested frame")
                continue
            assert index == self.next_index
            self.cache[index] = tensor
            self.order.append(index)
            self.next_index += 1
        self.wait_seconds += time.perf_counter() - waiting
        if start not in self.cache:
            raise RuntimeError(f"Frame {start} fell outside the retained overlap")
        if self.resize_mode in ("gpu", "gpu_exact"):
            pending = [i for i in range(start, stop) if self.cache[i].device.type == "cpu"]
            if pending:
                small = torch.stack([self.cache[i] for i in pending])
                if self.resize_mode == "gpu_exact":
                    from pil_exact_gpu import resize_exact_gpu, normalize_exact_gpu
                    pixels = normalize_exact_gpu(resize_exact_gpu(small))
                else:
                    small = small.to("cuda", dtype=torch.float32)
                    large = F.interpolate(small.permute(0, 3, 1, 2), size=(960, 1704),
                                          mode="bicubic", align_corners=False, antialias=True)
                    pixels = large[:, :, 32:928, 20:1684].round().clamp_(0, 255)
                    pixels = (pixels / 255.0 * 2.0 - 1.0).to(self.dtype)
                for i, frame in zip(pending, pixels):
                    self.cache[i] = frame
        result = torch.stack([self.cache[i] for i in range(start, stop)], dim=1).unsqueeze(0)
        # The largest backward request is the decoder's previous eight frames.
        while len(self.order) > 32:
            del self.cache[self.order.popleft()]
        return result[key[0], key[1], :, key[3], key[4]]

    def close(self):
        self.stop.set()
        self.worker.join(timeout=5)
        if self.error is not None:
            raise RuntimeError("Frame producer failed") from self.error
