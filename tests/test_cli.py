"""CLI robustness: hosts that pipe our output through a narrow encoding
(cp1252 pipes on Windows) must not crash us before processing starts."""
import io
import sys

from occlude.cli import main


def test_main_survives_cp1252_stdout(monkeypatch, tmp_path):
    raw = io.BytesIO()
    narrow = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", narrow)
    # Nonexistent input: main must reach its own error path (exit code 1),
    # not die printing the Unicode banner.
    assert main(["--input", str(tmp_path / "missing.mp4")]) == 1


def test_main_survives_stream_without_reconfigure(monkeypatch, tmp_path):
    class StrictAscii(io.TextIOBase):
        """No reconfigure(); rejects any non-ASCII write like a broken pipe
        encoding would."""

        def write(self, s):
            s.encode("ascii")
            return len(s)

    monkeypatch.setattr(sys, "stdout", StrictAscii())
    assert main(["--input", str(tmp_path / "missing.mp4")]) == 1
