"""Run the PR #1715 FlashVSR adapter on a local video copy."""

import argparse
from pathlib import Path
import shutil
import tempfile

from ml_modules.flashvsr_v11.video import enhance_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fast-inference", action="store_true")
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if source == output:
        parser.error("input and output must be different files")
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="flashvsr_public_") as job_dir:
        staged = Path(job_dir) / "source.mp4"
        shutil.copy2(source, staged)
        enhance_video(staged, target_width=args.width, target_height=args.height,
                      seed=args.seed, job_dir=job_dir, fast_inference=args.fast_inference)
        shutil.move(staged, output)
        print(output)


if __name__ == "__main__":
    main()
