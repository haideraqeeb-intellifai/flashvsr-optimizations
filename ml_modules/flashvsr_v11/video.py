"""Media adaptation around the unchanged official 4x / 8n-3 implementation."""

from dataclasses import dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import tempfile

from .runtime import run_official


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frames: int
    fps: Fraction
    audio_streams: int


def probe_video(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=600,
    )
    streams = json.loads(result.stdout)["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    fps = Fraction(video["avg_frame_rate"])
    packets = video.get("nb_read_packets")
    # The LTX and delivery H.264 MP4 encoders emit one packet per frame.
    # Check against the container frame count; retain decoding for other media.
    if (video.get("codec_name") == "h264" and str(path).lower().endswith(".mp4")
            and packets is not None and video.get("nb_frames") == packets):
        frames = int(packets)
    else:
        counted = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)],
            capture_output=True, text=True, check=True, timeout=600,
        )
        frames = int(counted.stdout.strip())
    if fps <= 0 or frames <= 0:
        raise ValueError(f"Invalid video timing: {path}")
    return VideoInfo(int(video["width"]), int(video["height"]), frames, fps,
                     sum(s["codec_type"] == "audio" for s in streams))


@dataclass(frozen=True)
class Geometry:
    content_width: int
    content_height: int
    width: int
    height: int
    left: int
    top: int


def input_geometry(target_width, target_height):
    if target_width <= 0 or target_height <= 0:
        raise ValueError("FlashVSR output dimensions must be positive")
    # The official pipeline enlarges the LQ input 4x and requires 128px output
    # multiples. Pad instead of its default center crop to retain the full view.
    cw, ch = (target_width + 3) // 4, (target_height + 3) // 4
    width, height = (cw + 31) // 32 * 32, (ch + 31) // 32 * 32
    return Geometry(cw, ch, width, height, (width - cw) // 2, (height - ch) // 2)


def padded_frame_count(frames):
    # Tiny Long's first decode needs 25 inference frames; its output has F-4
    # frames. The example adds four itself, so supply at least 21 (=8*3-3).
    return max(21, frames + (5 - frames) % 8)


def prepare_video(source, destination, info, geometry):
    frames = padded_frame_count(info.frames)
    filters = (
        f"scale={geometry.content_width}:{geometry.content_height}:flags=bicubic,"
        f"pad={geometry.width}:{geometry.height}:{geometry.left}:{geometry.top},"
        f"fillborders=left={geometry.left}:right={geometry.width - geometry.content_width - geometry.left}:"
        f"top={geometry.top}:bottom={geometry.height - geometry.content_height - geometry.top}:mode=smear,"
        f"setsar=1,tpad=stop_mode=clone:stop={frames - info.frames}"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-map", "0:v:0", "-an",
         "-vf", filters, "-r", str(info.fps), "-frames:v", str(frames),
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0", "-pix_fmt", "yuv444p", str(destination)],
        check=True, timeout=1800,
    )


def finish_video(enhanced, source, destination, info, geometry, target_width, target_height):
    filters = (
        f"trim=end_frame={info.frames},setpts=PTS-STARTPTS,"
        f"crop={geometry.content_width * 4}:{geometry.content_height * 4}:"
        f"{geometry.left * 4}:{geometry.top * 4}:exact=1,"
        f"scale={target_width}:{target_height}:flags=bicubic,setsar=1"
    )
    # Preserve source audio packets; padding is removed from video only. No
    # -shortest: that can drop video frames with short generated audio tracks.
    pixel_format = "yuv420p" if target_width % 2 == target_height % 2 == 0 else "yuv444p"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(enhanced), "-i", str(source),
         "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "-1", "-vf", filters,
         "-r", str(info.fps), "-c:v", "libx264",
         "-preset", "medium", "-crf", "18", "-pix_fmt", pixel_format,
         "-c:a", "copy", "-movflags", "+faststart", str(destination)],
        check=True, timeout=1800,
    )
    output = probe_video(destination)
    expected = VideoInfo(target_width, target_height, info.frames, info.fps, info.audio_streams)
    if output != expected:
        raise RuntimeError(f"FlashVSR changed the delivery contract: expected {expected}, got {output}")


def enhance_video(source, *, target_width, target_height, seed, job_dir, fast_inference=False):
    """Replace only a fully validated result; leave the source intact on error."""
    source = Path(source).resolve()
    info = probe_video(source)
    geometry = input_geometry(target_width, target_height)
    with tempfile.TemporaryDirectory(prefix="flashvsr_v11_", dir=job_dir) as temp:
        staging = Path(temp)
        prepared, enhanced, output = (staging / name for name in ("input.mp4", "enhanced.mp4", "output.mp4"))
        prepare_video(source, prepared, info, geometry)
        arguments = ["--input", str(prepared), "--output", str(enhanced), "--seed", str(seed),
                     "--fps", str(info.fps), "--work-dir", str(staging),
                     "--content-frames", str(padded_frame_count(info.frames)),
                     "--input-width", str(geometry.width), "--input-height", str(geometry.height)]
        if fast_inference:
            arguments.append("--fast-inference")
        run_official(arguments, Path(job_dir) / "flashvsr_v11.log")
        finish_video(enhanced, source, output, info, geometry, target_width, target_height)
        os.replace(output, source)
