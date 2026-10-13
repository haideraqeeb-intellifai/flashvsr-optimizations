"""Sample PIL-versus-GPU bicubic input differences on all six staged clips."""

import json
import math
from pathlib import Path

import imageio
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/chunked-input-study/gpu-resize-input-quality.json"


def main():
    cases = json.loads((ROOT / "outputs/lq-study/combined/results.json").read_text())["cases"]
    rows = []
    for case in cases:
        path = ROOT / "outputs/lq-study/combined" / case["case"] / "low_resolution_padded.mp4"
        count = case["frames"] + case["tail_padding"]
        reader = imageio.get_reader(path)
        try:
            source = [Image.fromarray(reader.get_data(i)).convert("RGB")
                      for i in (0, count // 2, count - 1)]
        finally:
            reader.close()
        pil = np.stack([np.asarray(frame.resize((1704, 960), Image.BICUBIC)
                                    .crop((20, 32, 1684, 928)), dtype=np.uint8)
                        for frame in source])
        small = torch.from_numpy(np.stack([np.array(frame, dtype=np.uint8) for frame in source]))
        large = F.interpolate(small.to("cuda", dtype=torch.float32).permute(0, 3, 1, 2),
                              size=(960, 1704), mode="bicubic", align_corners=False,
                              antialias=True)
        gpu = large[:, :, 32:928, 20:1684].round().clamp_(0, 255).to(torch.uint8)
        gpu = gpu.permute(0, 2, 3, 1).cpu().numpy()
        diff = pil.astype(np.float32) - gpu.astype(np.float32)
        mse = float(np.mean(diff * diff))
        rows.append({"case": case["case"], "sampled_frames": 3,
                     "pixel_mae": float(np.mean(np.abs(diff))),
                     "pixel_max_difference": int(np.max(np.abs(diff))),
                     "equal_pixel_fraction": float(np.mean(diff == 0)),
                     "psnr_db": 10 * math.log10(255**2 / mse) if mse else None})
        print(rows[-1], flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
