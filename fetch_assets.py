"""Fetch the original benchmark inputs and pinned weights; no inference API."""
import argparse
import json
from pathlib import Path
import urllib.request

from benchmark import ROOT, sha256


def download(url, destination, expected):
    if destination.exists() and sha256(destination) == expected:
        print(f"Verified {destination.relative_to(ROOT)}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    print(f"Downloading {destination.relative_to(ROOT)}", flush=True)
    urllib.request.urlretrieve(url, temporary)
    actual = sha256(temporary)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {destination}: {actual}")
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", action="store_true")
    args = parser.parse_args()
    report = json.loads((ROOT / "provenance/benchmark_manifest.json").read_text())
    assets = json.loads((ROOT / "provenance/assets.json").read_text())
    for case in report["cases"]:
        info = case["input"]
        download(info["url"], ROOT / "inputs" / info["url"].split("/")[-1], info["sha256"])
        if args.references:
            info = case["flashvsr_v11"]
            download(info["url"], ROOT / "references" / (case["case"] + ".mp4"), info["sha256"])
    for filename, digest in assets["model_sha256"].items():
        url = f"https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/resolve/{assets['model_revision']}/{filename}"
        download(url, ROOT / "models/FlashVSR-v1.1" / filename, digest)


if __name__ == "__main__":
    main()
