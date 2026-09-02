"""Progress reporting for the pipeline — human tqdm bars, plus an optional
machine-readable stream for GUI frontends.

OCCLUDE's own UX is the tqdm bars on stderr. But a host application that
embeds OCCLUDE (the OpenShot "blur on export" integration, for one) runs the
CLI as a subprocess and cannot scrape animated tqdm output reliably. When
machine progress is enabled — the ``--machine-progress`` CLI flag, or the
``OCCLUDE_MACHINE_PROGRESS=1`` environment variable — every progress bar
additionally emits self-describing lines to stdout, one JSON object per line:

    OCCLUDE-PROGRESS {"stage": "Pass 3/3 render", "done": 120, "total": 4000}

Lines are throttled to whole-percent changes (plus a first and a final line
per stage), so even a long render adds at most ~100 lines per pass. Frontends
should treat any stdout line NOT starting with ``OCCLUDE-PROGRESS `` as
ordinary log text. ``total`` is null when the stage length is unknown.

Kept deliberately free of cv2/torch imports so frontends and tests can use
:func:`parse_progress_line` without OCCLUDE's heavy dependencies installed.
"""
from __future__ import annotations

import json
import os
import sys

PROGRESS_PREFIX = "OCCLUDE-PROGRESS "
_ENV_FLAG = "OCCLUDE_MACHINE_PROGRESS"


def machine_progress_enabled() -> bool:
    return os.getenv(_ENV_FLAG, "").strip() not in ("", "0")


def enable_machine_progress() -> None:
    """Turn on machine progress for this process and its children."""
    os.environ[_ENV_FLAG] = "1"


def emit_progress(stage: str, done: int, total: int | None) -> None:
    """Write one machine-readable progress line to stdout (unthrottled)."""
    payload = {"stage": stage, "done": done, "total": total}
    sys.stdout.write(PROGRESS_PREFIX + json.dumps(payload) + "\n")
    sys.stdout.flush()


def parse_progress_line(line: str) -> dict | None:
    """Inverse of :func:`emit_progress` for frontends.

    Returns the ``{"stage", "done", "total"}`` dict for a machine progress
    line, or None for any other line (including malformed ones — a frontend
    should never crash on subprocess output).
    """
    line = line.strip()
    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        payload = json.loads(line[len(PROGRESS_PREFIX):])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "stage" not in payload:
        return None
    return payload


def pbar(total: int | None, desc: str, unit: str):
    """A tqdm bar that also emits machine progress lines when enabled."""
    # Imported here so parse_progress_line stays usable without tqdm.
    from tqdm import tqdm

    if not machine_progress_enabled():
        return tqdm(total=total, desc=desc, unit=unit)
    return _MachineProgressBar(total=total, desc=desc, unit=unit)


def _percent(done: int, total: int | None) -> int | None:
    if not total or total <= 0:
        return None
    return int(done * 100 / total)


class _MachineProgressBar:
    """tqdm wrapper emitting a stdout line per whole-percent change.

    Wraps rather than subclasses tqdm: the only surface the pipeline uses is
    update()/close(), and a wrapper can't be broken by tqdm internals.
    """

    def __init__(self, total: int | None, desc: str, unit: str) -> None:
        from tqdm import tqdm

        self._tqdm = tqdm(total=total, desc=desc, unit=unit)
        self._stage = desc
        self._total = total
        self._done = 0
        self._closed = False
        emit_progress(self._stage, 0, total)
        self._last_percent = _percent(0, total)

    def update(self, n: int = 1) -> None:
        self._tqdm.update(n)
        self._done += n
        pct = _percent(self._done, self._total)
        if pct != self._last_percent:
            self._last_percent = pct
            emit_progress(self._stage, self._done, self._total)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._tqdm.close()
        emit_progress(self._stage, self._done, self._total)
