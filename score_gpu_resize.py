"""Compare GPU-resize outputs with exact-PIL outputs over every decoded frame."""

import json
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/chunked-input-study"


def score(a, b, filter_name):
    run = subprocess.run([
        "ffmpeg", "-v", "info", "-nostats", "-i", str(a), "-i", str(b),
        "-an", "-lavfi", filter_name, "-f", "null", "-",
    ], capture_output=True, text=True, check=True)
    if filter_name == "psnr":
        match = re.search(r"PSNR y:.*?average:([\d.]+)", run.stderr)
    else:
        match = re.search(r"SSIM Y:.*?All:([\d.]+)", run.stderr)
    if not match:
        raise RuntimeError(f"Missing {filter_name} in FFmpeg output: {run.stderr[-1000:]}")
    return float(match.group(1))


def main():
    exact = json.loads((OUT / "all-six/results.json").read_text())
    gpu = json.loads((OUT / "gpu-all-six/results.json").read_text())
    assert exact["status"] == gpu["status"] == "completed"
    assert len(exact["cases"]) == len(gpu["cases"]) == 6
    exact_rows = {row["case"]: row for row in exact["cases"]}
    rows = []
    for case in gpu["cases"]:
        reference = exact_rows[case["case"]]
        assert case["frames"] == reference["frames"]
        assert all(case["integrity"].values())
        a = OUT / "all-six" / reference["output"]
        b = OUT / "gpu-all-six" / case["output"]
        row = {"case": case["case"], "frames": case["frames"],
               "psnr_db": score(a, b, "psnr"),
               "ssim": score(a, b, "ssim"),
               "output_hash_identical": case["output_sha256"] == reference["output_sha256"]}
        rows.append(row)
        print(row, flush=True)
    data = {"reference": "exact PIL chunked output", "scope": "all decoded output frames",
            "cases": rows, "mean_psnr_db": statistics.mean(row["psnr_db"] for row in rows),
            "mean_ssim": statistics.mean(row["ssim"] for row in rows)}
    (OUT / "gpu-resize-output-quality.json").write_text(json.dumps(data, indent=2) + "\n")
    print("MEANS", data["mean_psnr_db"], data["mean_ssim"])


if __name__ == "__main__":
    main()
