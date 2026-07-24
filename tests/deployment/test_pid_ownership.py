"""Tests for PID ownership verification in Phase 7 stop scripts.

Verifies that ``verify_pid_owner`` and ``stop_service`` (defined in
``scripts/deploy/load_env.sh``) correctly:

* allow stopping a process we own,
* reject a stale PID file (process no longer running) without killing,
* reject a reused PID (start-time mismatch) without killing an unrelated
  process.

These tests require ``/proc`` (Linux) and ``bash``; they are skipped on
other platforms.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LOAD_ENV = REPO_ROOT / "scripts" / "deploy" / "load_env.sh"

pytestmark = pytest.mark.skipif(
    platform.system() != "Linux" or not shutil.which("bash"),
    reason="PID ownership tests require Linux /proc and bash",
)


def _run_bash(snippet: str, env: dict[str, str] | None = None) -> tuple[int, str, str]:
    """Run a bash snippet that sources load_env.sh; return (rc, stdout, stderr)."""
    full = (
        "set -euo pipefail\n"
        f'source "{LOAD_ENV}" 2>/dev/null\n'
        f"{snippet}\n"
    )
    proc = subprocess.run(
        ["bash", "-c", full],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _make_temp_pid_dir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="phase7_pidtest_")


def test_owned_process_can_be_stopped():
    # A process we started and recorded should be verified and stoppable.
    with _make_temp_pid_dir() as d:
        out = _run_bash(f'''
            PID_DIR="{d}"
            PID_FILE="$PID_DIR/test.pid"
            STATUS_FILE="$PID_DIR/test.status"
            # Start a dummy process we own.
            sleep 300 &
            PID=$!
            write_pid_meta "$PID_FILE" "$PID" "sleep" "test-session"
            # verify_pid_owner must succeed (return 0).
            if verify_pid_owner "$PID_FILE" "sleep"; then
                echo "VERIFY=ok"
            else
                echo "VERIFY=fail"
            fi
            # stop_service must kill it.
            stop_service "Test" "nonexistent-session-xyz" "$PID_FILE" "$STATUS_FILE" "sleep"
            # Process should be gone.
            if kill -0 "$PID" 2>/dev/null; then
                echo "ALIVE=yes"
            else
                echo "ALIVE=no"
            fi
        ''')
        stdout = out[1]
        assert "VERIFY=ok" in stdout, f"verify_pid_owner failed:\n{out[2]}"
        assert "verified PID" in stdout, f"stop_service did not verify:\n{stdout}"
        assert "ALIVE=no" in stdout, f"process still alive after stop:\n{stdout}"


def test_stale_pid_file_does_not_kill_unrelated_process():
    # A PID file pointing to a dead PID must not cause any kill, and the
    # stale PID file should be cleaned up.
    with _make_temp_pid_dir() as d:
        out = _run_bash(f'''
            PID_DIR="{d}"
            PID_FILE="$PID_DIR/stale.pid"
            STATUS_FILE="$PID_DIR/stale.status"
            # Write a PID that does not exist (very high, unlikely to be live).
            echo "999999" > "$PID_FILE"
            write_pid_meta "$PID_FILE" "999999" "sleep" "test-session"
            # verify_pid_owner must reject (return 1).
            if verify_pid_owner "$PID_FILE" "sleep"; then
                echo "VERIFY=ok"
            else
                echo "VERIFY=rejected"
            fi
            # stop_service must NOT send any signal; should report not running.
            stop_service "Stale" "nonexistent-session-xyz" "$PID_FILE" "$STATUS_FILE" "sleep"
            # Confirm PID 999999 was not killed (it wasn't running anyway; the
            # point is stop_service did not error and cleaned the file).
            if [ -f "$PID_FILE" ]; then
                echo "PIDFILE=exists"
            else
                echo "PIDFILE=removed"
            fi
        ''')
        stdout = out[1]
        assert "VERIFY=rejected" in stdout, \
            f"stale PID was not rejected:\n{stdout}\n{out[2]}"
        assert "not running" in stdout or "STALE_PID" in stdout, \
            f"stop_service misbehaved for stale PID:\n{stdout}"
        assert "PIDFILE=removed" in stdout, \
            f"stale PID file was not cleaned up:\n{stdout}"


def test_pid_reuse_is_rejected():
    # If the recorded start-time does not match the current process's
    # start-time (PID was reused by an unrelated process), verify_pid_owner
    # must reject and stop_service must NOT kill the unrelated process.
    with _make_temp_pid_dir() as d:
        out = _run_bash(f'''
            PID_DIR="{d}"
            PID_FILE="$PID_DIR/reuse.pid"
            STATUS_FILE="$PID_DIR/reuse.status"
            # Start a dummy process.
            sleep 300 &
            PID=$!
            # Record the real meta first.
            write_pid_meta "$PID_FILE" "$PID" "sleep" "test-session"
            # Now corrupt the recorded start-time to simulate PID reuse.
            META_FILE="$PID_FILE.meta"
            # Overwrite process_start_time with a clearly-wrong value.
            sed -i 's/^process_start_time=.*/process_start_time=1/' "$META_FILE"
            # verify_pid_owner must reject due to start-time mismatch.
            if verify_pid_owner "$PID_FILE" "sleep"; then
                echo "VERIFY=ok"
            else
                echo "VERIFY=rejected"
            fi
            # stop_service must NOT kill the process (it is "unrelated" now).
            stop_service "Reuse" "nonexistent-session-xyz" "$PID_FILE" "$STATUS_FILE" "sleep"
            # The process must still be alive (we did not kill it).
            if kill -0 "$PID" 2>/dev/null; then
                echo "ALIVE=yes"
            else
                echo "ALIVE=no"
            fi
            # Clean up the dummy process ourselves.
            kill "$PID" 2>/dev/null || true
        ''')
        stdout = out[1]
        assert "VERIFY=rejected" in stdout, \
            f"PID reuse was not detected:\n{stdout}\n{out[2]}"
        assert "STALE_PID" in stdout, \
            f"stop_service did not report STALE_PID:\n{stdout}"
        assert "ALIVE=yes" in stdout, \
            f"unrelated process was killed during PID reuse:\n{stdout}"
