"""Fetch pinned official source/weights without changing the worker environment."""

import argparse
import subprocess

from .runtime import MODEL_FILES, MODEL_ID, MODEL_REVISION, SOURCE_REVISION, SOURCE_URL, cache_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "upstream"
    if not source.exists():
        subprocess.run(["git", "clone", "--no-checkout", SOURCE_URL, str(source)], check=True, timeout=600)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", SOURCE_REVISION], check=True, timeout=120)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=30).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True, timeout=30)
    if revision != SOURCE_REVISION or dirty:
        raise RuntimeError(f"Expected unchanged official FlashVSR at {SOURCE_REVISION}: {source}")
    if not args.source_only:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=MODEL_ID, revision=MODEL_REVISION, allow_patterns=list(MODEL_FILES),
            local_dir=str(root / "models"), max_workers=3,
        )
    print(f"Official FlashVSR v1.1 assets: {root}")


if __name__ == "__main__":
    main()
