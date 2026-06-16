"""OCCLUDE — CLI entry point.

Usage:
    occlude --input <video> [--output <path>] [--blur-strength N]

Detects immodestly dressed people frame-by-frame and writes a clean
video with the original audio preserved. See OCCLUDE_SPEC.md.
"""
from __future__ import annotations

import os

# Diagnostic instrumentation: traps os._exit + sys.exit + fatal signals and
# dumps all-thread stacks to /tmp so a silent death pinpoints its origin.
# Activated only when OCCLUDE_DEBUG=1 to keep `--help` side-effect-free.
if os.getenv("OCCLUDE_DEBUG") == "1":
    import atexit as _diag_atexit
    import faulthandler as _diag_faulthandler
    import signal as _diag_signal
    import sys as _diag_sys
    import threading as _diag_threading
    import time as _diag_time
    import traceback as _diag_tb

    _DIAG_LOG = "/tmp/occlude_diag.log"
    _FAULT_LOG = "/tmp/occlude_faulthandler.log"

    def _diag_log(msg: str) -> None:
        with open(_DIAG_LOG, "a") as f:
            f.write(f"[{_diag_time.strftime('%H:%M:%S')}] {msg}\n")
            f.flush()

    _real_os_exit = os._exit

    def _traced_os_exit(code):  # noqa: ANN001
        with open(_DIAG_LOG, "a") as f:
            f.write(
                f"\n========== os._exit({code}) at "
                f"{_diag_time.strftime('%H:%M:%S')} ==========\n"
            )
            f.write(f"main thread: {_diag_threading.main_thread().ident}\n")
            f.write(
                f"current thread: {_diag_threading.current_thread().ident} "
                f"({_diag_threading.current_thread().name})\n"
            )
            f.write("--- python stack at exit ---\n")
            _diag_tb.print_stack(file=f)
            f.write("--- all threads ---\n")
            for tid, frame in _diag_sys._current_frames().items():
                f.write(f"\n>>> thread {tid}\n")
                _diag_tb.print_stack(frame, file=f)
            f.flush()
        _real_os_exit(code)

    os._exit = _traced_os_exit

    _real_sys_exit = _diag_sys.exit

    def _traced_sys_exit(code=0):  # noqa: ANN001
        with open(_DIAG_LOG, "a") as f:
            f.write(
                f"[{_diag_time.strftime('%H:%M:%S')}] sys.exit({code}) "
                "called from:\n"
            )
            _diag_tb.print_stack(file=f)
            f.flush()
        _real_sys_exit(code)

    _diag_sys.exit = _traced_sys_exit

    _diag_fault_fh = open(_FAULT_LOG, "w")
    _diag_faulthandler.enable(file=_diag_fault_fh, all_threads=True)
    for _diag_sig in (
        _diag_signal.SIGTERM,
        _diag_signal.SIGINT,
        _diag_signal.SIGHUP,
        _diag_signal.SIGUSR1,
        _diag_signal.SIGUSR2,
        _diag_signal.SIGPIPE,
        _diag_signal.SIGQUIT,
    ):
        try:
            _diag_faulthandler.register(
                _diag_sig, file=_diag_fault_fh, all_threads=True, chain=True
            )
        except Exception as _e:  # noqa: BLE001
            _diag_log(f"faulthandler.register({_diag_sig}) failed: {_e}")

    def _diag_on_atexit() -> None:
        _diag_log("atexit fired (orderly python shutdown)")

    _diag_atexit.register(_diag_on_atexit)
    _diag_log(
        f"=== occlude started, pid={os.getpid()}, "
        f"argv={_diag_sys.argv} ==="
    )

import argparse
import shutil
import sys
from pathlib import Path

from rich.console import Console


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="occlude",
        description="Blur immodestly dressed people in a video.",
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="path to the input video file",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="output video path (default: <input_stem>_occluded.mp4 next to input)",
    )
    # Import only lightweight constants here so `occlude --help` does not
    # pay torch/OpenCV/model-adjacent import costs.
    from occlude.pipeline.config import DEFAULT_BLUR_KERNEL
    parser.add_argument(
        "--blur-strength", type=int, default=DEFAULT_BLUR_KERNEL,
        help=f"Gaussian blur kernel size (must be odd, default {DEFAULT_BLUR_KERNEL})",
    )
    parser.add_argument(
        "--perception-batch", type=int, default=None,
        help="bundle this many consecutive frames into one YOLO+SegFormer forward pass (default: 4 on CUDA, 1 elsewhere)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="inference device (default: auto; CUDA is preferred when available)",
    )
    parser.add_argument(
        "--detector-model",
        type=str,
        default=None,
        help=(
            "YOLO person-detector weights (default yolov8n.pt). On a CUDA "
            "box with spare VRAM, yolov8m.pt or yolov8l.pt improves recall "
            "on small / distant / odd-angle people at some throughput cost; "
            "a missed detection is a missed blur"
        ),
    )
    parser.add_argument(
        "--require-cuda-io",
        action="store_true",
        help="fail unless CUDA/NVDEC decode and NVENC encode are available",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="benchmark the first --seconds of the input and print fps, GPU util, peak VRAM, and a frame-hash MD5; no output video is produced",
    )
    parser.add_argument(
        "--seconds", type=float, default=30.0,
        help="benchmark duration in seconds (only used with --benchmark, default 30)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from occlude.ui.ascii_art import get_header_panel
    console = Console()
    console.print(get_header_panel())

    input_path: Path = args.input
    if not input_path.exists() or not input_path.is_file():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 1

    if args.blur_strength <= 0 or args.blur_strength % 2 == 0:
        print(
            f"error: --blur-strength must be a positive odd integer, got {args.blur_strength}",
            file=sys.stderr,
        )
        return 1

    if args.perception_batch is not None and args.perception_batch < 1:
        print(
            f"error: --perception-batch must be >= 1, got {args.perception_batch}",
            file=sys.stderr,
        )
        return 1

    if shutil.which("ffmpeg") is None:
        print(
            "error: ffmpeg not found on PATH. Install via: brew install ffmpeg",
            file=sys.stderr,
        )
        return 1

    if args.benchmark:
        if args.seconds <= 0:
            print(f"error: --seconds must be > 0, got {args.seconds}", file=sys.stderr)
            return 1
        from occlude.pipeline.bench import run_benchmark
        try:
            result = run_benchmark(input_path, seconds=args.seconds)
        except KeyboardInterrupt:
            print("\ncancelled by user.", file=sys.stderr)
            return 130
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(result.format_line())
        return 0

    output_path: Path = args.output or (
        input_path.parent / f"{input_path.stem}_occluded.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Imported here so `--help` and arg validation don't pay the
    # multi-second model-loading cost.
    from occlude.pipeline.video import VideoProcessor

    try:
        processor = VideoProcessor(
            blur_kernel=args.blur_strength,
            perception_batch=args.perception_batch,
            device=args.device,
            detector_model=args.detector_model,
            require_cuda_io=args.require_cuda_io,
        )
        processor.process(input_path, output_path)
    except KeyboardInterrupt:
        print("\ncancelled by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"done. output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
