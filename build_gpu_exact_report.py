"""Build the public original-vs-pixel-exact-GPU FlashVSR video report."""

from html import escape
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/chunked-input-study"
CONTROL = ROOT / "outputs/lq-study/combined"
EXACT = OUT / "gpu-exact-all-six"
REFERENCES = ROOT / "references"

STAGES = (
    ("low_resolution_staging", "Low-resolution staging"),
    ("input_tensor_preparation", "Input preparation before inference"),
    ("inference", "Model phase, including overlapped preparation"),
    ("materialization_and_encoding", "Output conversion and encoding"),
    ("finalization", "Finalization"),
)


def avg(rows, key):
    return statistics.mean(row[key] for row in rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    control = json.loads((CONTROL / "results.json").read_text())
    exact = json.loads((EXACT / "results.json").read_text())
    assert control["status"] == exact["status"] == "completed"
    old = {row["case"]: row for row in control["cases"]}
    rows = exact["cases"]
    assert len(old) == len(rows) == 6
    assert all(row["output_sha256"] == old[row["case"]]["output_sha256"] for row in rows)
    assert all(all(row["integrity"].values()) for row in rows)
    published = json.loads((ROOT / "outputs/lq-study/report-data.json").read_text())
    reference_hashes = {
        case["case"]: case["quality"]["shivam"]["reference_sha256"]
        for case in published["studies"][0]["cases"]
    }
    for row in rows:
        name = row["case"]
        assert (CONTROL / name / "enhanced.mp4").is_file()
        assert (EXACT / name / "enhanced.mp4").is_file()
        reference = REFERENCES / f"{name}.mp4"
        assert sha256(reference) == reference_hashes[name]
        target = OUT / "shivam" / f"{name}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.hardlink_to(reference)

    old_rows = [old[row["case"]] for row in rows]
    a = avg(old_rows, "processing_seconds_after_model_load")
    b = avg(rows, "processing_seconds_after_model_load")
    reduction = (1 - b / a) * 100

    stage_html = "".join(
        f"<tr><td>{label}</td><td>{avg([r['timing_breakdown_seconds'] for r in old_rows], key):.2f} s</td>"
        f"<td>{avg([r['timing_breakdown_seconds'] for r in rows], key):.2f} s</td></tr>"
        for key, label in STAGES
    )
    case_html = "".join(
        f"<tr><td><a href='#{escape(row['case'])}'>{escape(row['label'])}</a></td>"
        f"<td>{row['frames']}</td>"
        f"<td>{old[row['case']]['processing_seconds_after_model_load']:.2f} s</td>"
        f"<td>{row['processing_seconds_after_model_load']:.2f} s</td>"
        f"<td>Identical</td></tr>"
        for row in rows
    )
    comparisons = "".join(
        f"<section class='card case' id='{escape(row['case'])}'>"
        f"<div class='case-head'><div><h3>{escape(row['label'])}</h3>"
        f"<p>{row['frames']} frames · {old[row['case']]['processing_seconds_after_model_load']:.2f} s → "
        f"{row['processing_seconds_after_model_load']:.2f} s · Identical SHA-256</p></div>"
        f"<div class='controls'><button type='button' class='play'>Play both</button>"
        f"<button type='button' class='pause'>Pause both</button></div></div>"
        f"<div class='pair'><div><h4>Shivam's published run</h4><video controls playsinline preload='metadata' muted "
        f"src='shivam/{escape(row['case'])}.mp4'></video></div>"
        f"<div><h4>Chunked GPU · pixel-identical</h4><video controls playsinline preload='metadata' "
        f"src='gpu-exact-all-six/{escape(row['case'])}/enhanced.mp4'></video></div></div></section>"
        for row in rows
    )
    html = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>FlashVSR pixel-identical chunked GPU report</title>
<style>
:root{{color-scheme:dark}}body{{margin:0;background:#10151d;color:#edf3f8;font:16px/1.5 system-ui,sans-serif}}
main{{max-width:1420px;margin:auto;padding:28px 20px 80px}}h1,h2,h3,h4{{line-height:1.2}}
h1{{font-size:clamp(2rem,4vw,3.1rem);margin:.2em 0}}h2{{margin:0 0 16px}}h3{{margin:0 0 5px}}h4{{margin:0 0 8px}}
.lead{{font-size:1.12rem;max-width:850px;color:#bdcbd7}}.card{{background:#1a2430;border:1px solid #354456;border-radius:14px;
box-shadow:0 8px 24px #0003;padding:20px;margin:18px 0}}.metrics{{display:flex;gap:12px;flex-wrap:wrap}}
.metric{{background:#183a3b;border:1px solid #306c69;border-radius:10px;padding:14px 18px;min-width:180px}}
.metric strong{{display:block;font-size:1.6rem;color:#6ee5d2}}.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:10px 12px;border-bottom:1px solid #354456;text-align:left}}
th{{background:#263545;white-space:nowrap}}a{{color:#8bcaff}}.case-head{{display:flex;justify-content:space-between;
align-items:center;gap:12px;flex-wrap:wrap}}.case-head p{{margin:0;color:#adbdcb}}.controls{{display:flex;gap:8px}}
button{{border:1px solid #3e98a3;border-radius:8px;padding:9px 13px;background:#145c68;color:#f5ffff;font:inherit;cursor:pointer}}
button:hover{{background:#1b7482}}.pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:18px}}
video{{display:block;width:100%;aspect-ratio:13/7;background:#0d1117;border-radius:8px}}
@media(max-width:780px){{.pair{{grid-template-columns:1fr}}main{{padding:20px 12px 55px}}}}
</style></head><body><main>
<header><h1>FlashVSR: pixel-identical GPU input</h1>
<p class='lead'>Six complete input videos on the RTX 5090. Small frames are decoded ahead of the model, transferred to the GPU, and resized with Pillow-equivalent arithmetic. The video comparisons pair Shivam's published outputs with the chunked GPU outputs. Each chunked GPU MP4 is byte-identical to its original optimized counterpart.</p>
<div class='metrics'><div class='metric'><strong>{a:.2f} → {b:.2f} s</strong>Mean finished-video time</div>
<div class='metric'><strong>{a-b:.2f} s</strong>Saved per video</div>
<div class='metric'><strong>{reduction:.1f}%</strong>Less processing time</div>
<div class='metric'><strong>6 / 6</strong>Identical output hashes</div></div></header>
<section class='card'><h2>Mean timing</h2><div class='scroll'><table><thead><tr><th>Step</th>
<th>Original optimized</th><th>Chunked GPU · exact</th></tr></thead><tbody>{stage_html}
<tr><th>Finished video</th><th>{a:.2f} s</th><th>{b:.2f} s</th></tr></tbody></table></div>
<p>The 29.70 s original model timing excludes input and output work. The chunked GPU model-phase time includes preparation that overlaps inference. Both finished-video times start after model load and exclude asset checks and output validation. Each configuration had one warmed pass per clip; the original run is a historical control.</p></section>
<section class='card'><h2>Six input videos</h2><div class='scroll'><table><thead><tr><th>Clip</th><th>Frames</th>
<th>Original</th><th>Chunked GPU</th><th>Output hash</th></tr></thead><tbody>{case_html}</tbody></table></div></section>
<h2>Side-by-side videos</h2><p>Left: Shivam's published run. Right: pixel-identical chunked GPU run. Timing comparisons above use the original optimized run as the historical control.</p>{comparisons}
</main><script>
document.querySelectorAll('.case').forEach(section=>{{
 const videos=[...section.querySelectorAll('video')];
 section.querySelector('.play').addEventListener('click',()=>{{
  const time=videos[0].currentTime;
  videos.forEach(video=>{{video.currentTime=time;video.play().catch(()=>{{}})}});
 }});
 section.querySelector('.pause').addEventListener('click',()=>videos.forEach(video=>video.pause()));
 videos[0].addEventListener('timeupdate',()=>{{
  if(!videos[0].paused && Math.abs(videos[1].currentTime-videos[0].currentTime)>0.2)
   videos[1].currentTime=videos[0].currentTime;
 }});
}});
</script></body></html>"""
    path = OUT / "report.html"
    path.write_text(html)
    print(path, f"{a:.6f}", f"{b:.6f}", f"{reduction:.3f}%")


if __name__ == "__main__":
    main()
