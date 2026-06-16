"""Chunked wrapper around occlude.

Splits the input video into fixed-duration chunks, runs occlude
on each in a *fresh* Python subprocess, then ffmpeg-concats the
results. Each subprocess load is ~5s of model-init overhead, but in
exchange the Python heap and macOS VM accounting reset between chunks
— so allocator drift / fragmentation that would otherwise compound
into a 40+ GB memory footprint on long videos stays bounded per chunk.

Audio is preserved end-to-end: each chunk carries its own audio
segment from the input, and `ffmpeg -f concat` preserves both streams.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from occlude.pipeline.config import DEFAULT_BLUR_KERNEL

# The interpreter running this script — works under the project venv
# locally and under the system python on Colab. Hardcoding
# .venv/bin/python broke the Colab path (no venv there).
PY = sys.executable


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunked occlude wrapper")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--blur-strength", type=int, default=DEFAULT_BLUR_KERNEL)
    parser.add_argument("--perception-batch", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument("--chunk-seconds", type=int, default=10)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 1

    output = args.output or (args.input.parent / f"{args.input.stem}_occluded.mp4")
    work = Path(tempfile.mkdtemp(prefix="occlude_chunked_"))
    chunks_dir = work / "chunks"
    outs_dir = work / "outs"
    chunks_dir.mkdir()
    outs_dir.mkdir()

    print(f"work dir: {work}")
    print(f"splitting {args.input.name} into {args.chunk_seconds}s chunks...")
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-i", str(args.input),
        "-c", "copy", "-map", "0",
        "-segment_time", str(args.chunk_seconds),
        "-f", "segment", "-reset_timestamps", "1",
        str(chunks_dir / "chunk_%04d.mp4"),
    ], check=True)

    chunks = sorted(chunks_dir.glob("chunk_*.mp4"))
    print(f"got {len(chunks)} chunks")

    out_files = []
    for i, chunk in enumerate(chunks):
        out = outs_dir / f"out_{i:04d}.mp4"
        print(f"\n=== chunk {i+1}/{len(chunks)}: {chunk.name} ===")
        rc = subprocess.run([
            str(PY), "-m", "occlude",
            "--input", str(chunk),
            "--output", str(out),
            "--blur-strength", str(args.blur_strength),
        ] + (
            ["--perception-batch", str(args.perception_batch)]
            if args.perception_batch is not None else []
        ) + [
            "--device", args.device,
        ]).returncode
        if rc != 0:
            print(f"error: chunk {i} failed (exit {rc})", file=sys.stderr)
            return 1
        if not out.exists():
            print(f"error: chunk {i} produced no output", file=sys.stderr)
            return 1
        out_files.append(out)

    list_file = work / "concat.txt"
    list_file.write_text("\n".join(f"file '{p.resolve()}'" for p in out_files) + "\n")

    print(f"\nconcatenating {len(out_files)} chunks → {output}")
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output),
    ], check=True)

    shutil.rmtree(work, ignore_errors=True)
    print(f"\ndone. output: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
