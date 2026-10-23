"""Bounded asynchronous PIL preprocessing for ordered FlashVSR LQ slices."""

from collections import deque
import queue
import threading
import time

import imageio
import numpy as np
from PIL import Image
import torch


class ChunkedLQ:
    """Tensor-like source that decodes/resizes at most 32 frames ahead."""

    def __init__(self, path, content_frames, total_frames, *, lookahead=32, dtype=torch.bfloat16,
                 source_size=None):
        if lookahead < 8 or content_frames < 1 or total_frames < content_frames:
            raise ValueError("Invalid frame counts or lookahead")
        self.path = str(path)
        self.content_frames = content_frames
        self.total_frames = total_frames
        self.dtype = dtype
        self.queue = queue.Queue(maxsize=lookahead)
        self.cache = {}
        self.order = deque()
        self.next_index = 0
        self.producer_seconds = 0.0
        self.wait_seconds = 0.0
        self.error = None
        self.stop = threading.Event()
        if source_size is None:
            reader = imageio.get_reader(self.path)
            try:
                first = Image.fromarray(reader.get_data(0)).convert("RGB")
                w0, h0 = first.size
            finally:
                reader.close()
        else:
            w0, h0 = source_size
            if w0 <= 0 or h0 <= 0:
                raise ValueError("Invalid staged source dimensions")
        self.scaled_width, self.scaled_height = round(w0 * 4), round(h0 * 4)
        self.width = self.scaled_width // 128 * 128
        self.height = self.scaled_height // 128 * 128
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Staged input is too small for FlashVSR: {w0}x{h0}")
        self.left = (self.scaled_width - self.width) // 2
        self.top = (self.scaled_height - self.height) // 2
        self.worker = threading.Thread(target=self._produce, name="flashvsr-input", daemon=True)
        self.worker.start()

    def _produce(self):
        reader = None
        try:
            reader = imageio.get_reader(self.path)
            last = None
            for index in range(self.total_frames):
                if self.stop.is_set():
                    break
                started = time.perf_counter()
                if index < self.content_frames:
                    last = Image.fromarray(reader.get_data(index)).convert("RGB")
                resized = last.resize((self.scaled_width, self.scaled_height), Image.BICUBIC)
                cropped = resized.crop((self.left, self.top, self.left + self.width, self.top + self.height))
                pixels = torch.from_numpy(np.asarray(cropped, dtype=np.uint8).copy()).to(torch.float32)
                pixels = pixels.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
                tensor = pixels.to(self.dtype)
                self.producer_seconds += time.perf_counter() - started
                while not self.stop.is_set():
                    try:
                        self.queue.put((index, tensor), timeout=0.1)
                        break
                    except queue.Full:
                        continue
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
                    raise RuntimeError("FlashVSR input producer failed") from self.error
                if not self.worker.is_alive():
                    raise RuntimeError("FlashVSR input producer stopped before the requested frame")
                continue
            if index != self.next_index:
                raise RuntimeError(f"Out-of-order input frame: expected {self.next_index}, got {index}")
            self.cache[index] = tensor
            self.order.append(index)
            self.next_index += 1
        self.wait_seconds += time.perf_counter() - waiting
        if start not in self.cache:
            raise RuntimeError(f"Frame {start} fell outside the retained overlap")
        result = torch.stack([self.cache[index] for index in range(start, stop)], dim=1).unsqueeze(0)
        while len(self.order) > 32:
            del self.cache[self.order.popleft()]
        return result[key[0], key[1], :, key[3], key[4]]

    def close(self):
        self.stop.set()
        self.worker.join(timeout=30)
        if self.worker.is_alive():
            raise RuntimeError("FlashVSR input producer did not stop")
        if self.error is not None:
            raise RuntimeError("FlashVSR input producer failed") from self.error
