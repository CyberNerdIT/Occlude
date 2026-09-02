"""Machine progress stream: emitted only when enabled, parseable, throttled."""
import json

import pytest

from occlude.pipeline.progress import (
    PROGRESS_PREFIX,
    emit_progress,
    enable_machine_progress,
    machine_progress_enabled,
    parse_progress_line,
    pbar,
)


@pytest.fixture
def machine_progress(monkeypatch):
    monkeypatch.delenv("OCCLUDE_MACHINE_PROGRESS", raising=False)
    enable_machine_progress()
    yield
    monkeypatch.delenv("OCCLUDE_MACHINE_PROGRESS", raising=False)


def _progress_lines(capsys) -> list[dict]:
    out = capsys.readouterr().out
    return [
        p for p in (parse_progress_line(line) for line in out.splitlines())
        if p is not None
    ]


def test_disabled_by_default(monkeypatch, capsys):
    monkeypatch.delenv("OCCLUDE_MACHINE_PROGRESS", raising=False)
    assert not machine_progress_enabled()
    bar = pbar(total=10, desc="Pass 1/3 detect+track", unit="frame")
    bar.update(5)
    bar.close()
    assert capsys.readouterr().out == ""


def test_emit_and_parse_roundtrip(capsys):
    emit_progress("Pass 3/3 render", 120, 4000)
    line = capsys.readouterr().out.strip()
    assert line.startswith(PROGRESS_PREFIX)
    assert parse_progress_line(line) == {
        "stage": "Pass 3/3 render", "done": 120, "total": 4000,
    }


def test_parse_rejects_other_lines():
    assert parse_progress_line("done. output: x.mp4") is None
    assert parse_progress_line(PROGRESS_PREFIX + "not json") is None
    assert parse_progress_line(PROGRESS_PREFIX + '["list"]') is None
    assert parse_progress_line("") is None


def test_bar_emits_start_progress_and_close(machine_progress, capsys):
    bar = pbar(total=4, desc="Pass 2/3 judge", unit="crop")
    for _ in range(4):
        bar.update(1)
    bar.close()
    payloads = _progress_lines(capsys)
    assert payloads[0] == {"stage": "Pass 2/3 judge", "done": 0, "total": 4}
    assert payloads[-1] == {"stage": "Pass 2/3 judge", "done": 4, "total": 4}
    dones = [p["done"] for p in payloads]
    assert dones == sorted(dones)


def test_bar_throttles_to_percent_changes(machine_progress, capsys):
    bar = pbar(total=10_000, desc="Pass 3/3 render", unit="frame")
    for _ in range(10_000):
        bar.update(1)
    bar.close()
    payloads = _progress_lines(capsys)
    # start line + one per whole percent + final close line, not one per frame
    assert len(payloads) <= 102


def test_bar_with_unknown_total(machine_progress, capsys):
    bar = pbar(total=None, desc="Pass 1/3 detect+track", unit="frame")
    bar.update(1)
    bar.update(1)
    bar.close()
    payloads = _progress_lines(capsys)
    assert payloads[0]["total"] is None
    assert payloads[-1]["done"] == 2
