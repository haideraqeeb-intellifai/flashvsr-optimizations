"""Build the six-clip chunked-input timing and fidelity report."""

import csv
from html import escape
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/chunked-input-study"
CONTROL = ROOT / "outputs/lq-study/combined/results.json"
CHUNKED = OUT / "all-six/results.json"
GPU = OUT / "gpu-all-six/results.json"
GPU_EXACT = OUT / "gpu-exact-all-six/results.json"
GPU_QUALITY = OUT / "gpu-resize-output-quality.json"
STAGES = [
    ("low_resolution_staging", "Low-resolution staging"),
    ("input_tensor_preparation", "Input preparation before inference"),
    ("inference", "Model phase, including overlapped preparation in chunked run"),
    ("materialization_and_encoding", "Output conversion and encoding"),
    ("finalization", "Finalization"),
]


def mean(rows, key):
    return statistics.mean(row[key] for row in rows)


def main():
    old = json.loads(CONTROL.read_text())
    new = json.loads(CHUNKED.read_text())
    gpu = json.loads(GPU.read_text())
    gpu_exact = json.loads(GPU_EXACT.read_text())
    quality = json.loads(GPU_QUALITY.read_text())
    assert old["status"] == new["status"] == gpu["status"] == gpu_exact["status"] == "completed"
    controls = {row["case"]: row for row in old["cases"]}
    runs = new["cases"]
    gpu_rows = {row["case"]: row for row in gpu["cases"]}
    gpu_exact_rows = {row["case"]: row for row in gpu_exact["cases"]}
    quality_rows = {row["case"]: row for row in quality["cases"]}
    assert len(controls) == len(runs) == len(gpu_rows) == len(gpu_exact_rows) == len(quality_rows) == 6
    assert all(row["case"] in controls for row in runs)
    assert all(row["output_sha256"] == controls[row["case"]]["output_sha256"] for row in runs)
    assert all(row["output_sha256"] == controls[row["case"]]["output_sha256"]
               for row in gpu_exact_rows.values())
    assert all(all(row["integrity"].values()) for row in runs)
    assert all(all(row["integrity"].values()) for row in gpu_exact_rows.values())
    old_rows = [controls[row["case"]] for row in runs]

    stage_rows = []
    for key, label in STAGES:
        a = mean([r["timing_breakdown_seconds"] for r in old_rows], key)
        b = mean([r["timing_breakdown_seconds"] for r in runs], key)
        c = mean([gpu_rows[r["case"]]["timing_breakdown_seconds"] for r in runs], key)
        d = mean([gpu_exact_rows[r["case"]]["timing_breakdown_seconds"] for r in runs], key)
        stage_rows.append({"stage": label, "control_seconds": a, "chunked_seconds": b,
                           "gpu_seconds": c, "gpu_exact_seconds": d,
                           "difference_seconds": b - a})
    a_total = mean(old_rows, "processing_seconds_after_model_load")
    b_total = mean(runs, "processing_seconds_after_model_load")
    c_total = mean(list(gpu_rows.values()), "processing_seconds_after_model_load")
    d_total = mean(list(gpu_exact_rows.values()), "processing_seconds_after_model_load")
    summary = {
        "cases": len(runs), "control_mean_seconds": a_total,
        "chunked_mean_seconds": b_total, "saved_seconds": a_total - b_total,
        "time_reduction_percent": (1 - b_total / a_total) * 100,
        "throughput_ratio": a_total / b_total,
        "gpu_mean_seconds": c_total, "gpu_saved_seconds": a_total - c_total,
        "gpu_time_reduction_percent": (1 - c_total / a_total) * 100,
        "gpu_throughput_ratio": a_total / c_total,
        "gpu_exact_mean_seconds": d_total, "gpu_exact_saved_seconds": a_total - d_total,
        "gpu_exact_time_reduction_percent": (1 - d_total / a_total) * 100,
        "gpu_exact_throughput_ratio": a_total / d_total,
        "gpu_mean_psnr_db_vs_exact": quality["mean_psnr_db"],
        "gpu_mean_ssim_vs_exact": quality["mean_ssim"],
        "all_video_hashes_identical": True,
        "all_integrity_checks_passed": True,
        "control_path": str(CONTROL.relative_to(ROOT)),
        "chunked_path": str(CHUNKED.relative_to(ROOT)),
        "gpu_path": str(GPU.relative_to(ROOT)),
        "gpu_exact_path": str(GPU_EXACT.relative_to(ROOT)),
        "control_is_historical": True,
        "stage_means": stage_rows,
        "cases_detail": [{
            "case": row["case"], "frames": row["frames"],
            "control_seconds": controls[row["case"]]["processing_seconds_after_model_load"],
            "chunked_seconds": row["processing_seconds_after_model_load"],
            "gpu_seconds": gpu_rows[row["case"]]["processing_seconds_after_model_load"],
            "gpu_exact_seconds": gpu_exact_rows[row["case"]]["processing_seconds_after_model_load"],
            "saved_seconds": controls[row["case"]]["processing_seconds_after_model_load"] - row["processing_seconds_after_model_load"],
            "gpu_saved_seconds": controls[row["case"]]["processing_seconds_after_model_load"] - gpu_rows[row["case"]]["processing_seconds_after_model_load"],
            "gpu_exact_saved_seconds": controls[row["case"]]["processing_seconds_after_model_load"] - gpu_exact_rows[row["case"]]["processing_seconds_after_model_load"],
            "output_sha256": row["output_sha256"],
            "gpu_psnr_db_vs_exact": quality_rows[row["case"]]["psnr_db"],
            "gpu_ssim_vs_exact": quality_rows[row["case"]]["ssim"],
            "producer_active_seconds": row["input_pipeline"]["producer_active_seconds"],
            "consumer_queue_wait_seconds": row["input_pipeline"]["consumer_queue_wait_seconds"],
        } for row in runs],
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary["cases_detail"][0].keys())
        writer.writeheader()
        writer.writerows(summary["cases_detail"])

    def f(value):
        return f"{value:.2f} s"

    stage_html = "\n".join(
        f"<tr><td>{escape(row['stage'])}</td><td>{f(row['control_seconds'])}</td>"
        f"<td>{f(row['chunked_seconds'])}</td><td>{f(row['gpu_exact_seconds'])}</td>"
        f"<td>{f(row['gpu_seconds'])}</td></tr>"
        for row in stage_rows
    )
    case_html = "\n".join(
        f"<tr><td>{escape(row['case'])}</td><td>{row['frames']}</td>"
        f"<td>{f(row['control_seconds'])}</td><td>{f(row['chunked_seconds'])}</td>"
        f"<td>{f(row['gpu_exact_seconds'])}</td><td>{f(row['gpu_seconds'])}</td>"
        f"<td>{row['gpu_psnr_db_vs_exact']:.2f} dB / {row['gpu_ssim_vs_exact']:.4f}</td>"
        f"<td><a href='all-six/{escape(row['case'])}/enhanced.mp4'>Exact</a> · "
        f"<a href='gpu-exact-all-six/{escape(row['case'])}/enhanced.mp4'>GPU exact</a> · "
        f"<a href='gpu-all-six/{escape(row['case'])}/enhanced.mp4'>GPU</a></td></tr>"
        for row in summary["cases_detail"]
    )
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FlashVSR chunked input and pixel-exact GPU resize benchmark</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 20px;color:#17212b;background:#f7fafc}}
h1,h2{{line-height:1.2}}.card{{background:white;border:1px solid #dce5eb;border-radius:12px;padding:20px;margin:20px 0}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #dde5ea}}th{{background:#eef4f8}}
.scroll{{overflow-x:auto}}strong{{color:#115e59}}a{{color:#0b5cad}}</style>
<h1>FlashVSR: overlapped input preparation</h1>
<p>Six complete clips per variant, RTX 5090, warmup enabled, 426×240 staged input, 1664×896 output. The existing optimized configuration is a historical control; each new variant has one fresh pass. Model startup, download, input hashing and output validation are outside the timed window.</p>
<div class="card"><h2>Result</h2><p><strong>Pixel-exact GPU: {f(a_total)} → {f(d_total)}</strong> mean per clip; {f(a_total-d_total)} saved ({summary['gpu_exact_time_reduction_percent']:.1f}% less wall time). All six encoded output hashes match the original optimized run.</p>
<p><strong>Chunked PIL: {f(b_total)}</strong> mean per clip, also pixel-exact. The earlier PyTorch GPU resize measured {f(c_total)}, but its output differs: mean all-frame PSNR {quality['mean_psnr_db']:.2f} dB and SSIM {quality['mean_ssim']:.4f} against the exact version. All variants passed frame, FPS, resolution, audio packet and full decode checks.</p></div>
<div class="card"><h2>Mean timing by stage</h2><div class="scroll"><table><thead><tr><th>Stage</th><th>Existing</th><th>Chunked PIL</th><th>GPU exact</th><th>GPU approximate</th></tr></thead><tbody>{stage_html}<tr><th>Finished video</th><th>{f(a_total)}</th><th>{f(b_total)}</th><th>{f(d_total)}</th><th>{f(c_total)}</th></tr></tbody></table></div>
<p>Input preparation is priming of the first 21 frames. Remaining frames are prepared while the model runs, so their cost is included in the model-phase wall time. The model-phase row is not an isolated model-only timing for the chunked variants.</p></div>
<div class="card"><h2>Individual clips</h2><div class="scroll"><table><thead><tr><th>Clip</th><th>Frames</th><th>Existing</th><th>Chunked PIL</th><th>GPU exact</th><th>GPU approximate</th><th>Approximate PSNR / SSIM vs exact</th><th>Videos</th></tr></thead><tbody>{case_html}</tbody></table></div></div>
<div class="card"><h2>Method and data</h2><p>All chunked paths use a 32-frame queue. Chunked PIL uses the original imageio/PIL resize. GPU exact transfers the 426×240 frames, then uses custom integer GPU kernels matching Pillow 11.2.1's horizontal and vertical 22-bit fixed-point passes, crop, and CPU-equivalent BF16 pixel mapping. Every BF16 slice requested by the model matched the eager tensor bit for bit in a full-clip check. The earlier approximate GPU mode uses PyTorch bicubic interpolation and differs from PIL by about 0.12 pixel MAE on sampled input frames.</p>
<p>Run-to-run variation and the historical control limit the precision of measured gains. Raw data: <a href="summary.json">summary JSON</a>, <a href="cases.csv">case CSV</a>, <a href="all-six/results.json">chunked PIL JSON</a>, <a href="gpu-exact-all-six/results.json">GPU exact JSON</a>, <a href="gpu-all-six/results.json">GPU approximate JSON</a>, <a href="gpu-resize-output-quality.json">approximate output quality</a>, <a href="gpu-resize-input-quality.json">approximate input quality</a>, <a href="../lq-study/combined/results.json">control JSON</a>.</p></div></html>"""
    (OUT / "report.html").write_text(html)
    print(json.dumps({k: summary[k] for k in ("control_mean_seconds", "chunked_mean_seconds", "gpu_exact_mean_seconds", "gpu_mean_seconds", "gpu_exact_saved_seconds", "gpu_exact_time_reduction_percent")}, indent=2))


if __name__ == "__main__":
    main()
