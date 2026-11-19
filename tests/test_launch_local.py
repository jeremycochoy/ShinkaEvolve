"""The local launcher when a step after the start of the job process has an error."""

import errno
import logging
import select
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from shinka.launch import local

SLEEP_COMMAND = [sys.executable, "-c", "import time; time.sleep(60)"]
PRINT_COMMAND = [
    sys.executable,
    "-c",
    "import time; print('line of the job', flush=True); time.sleep(60)",
]


@pytest.mark.parametrize(
    "step, call, opened_logs",
    [("open", 1, 0), ("open", 2, 1), ("start", 1, 2), ("start", 2, 2)],
    ids=["stdout log open", "stderr log open", "stdout thread", "stderr thread"],
)
def test_a_failed_step_after_popen_kills_and_reaps_the_process(
    tmp_path, monkeypatch, step, call, opened_logs
):
    error = (
        OSError(errno.ENOSPC, "No space left on device")
        if step == "open"
        else RuntimeError("can't start new thread")
    )
    calls = {"open": 0, "start": 0}
    processes = []
    log_files = []

    def fail_at_the_selected_call(name):
        calls[name] += 1
        if (name, calls[name]) == (step, call):
            raise error

    def popen(*args, **kwargs):
        processes.append(subprocess.Popen(*args, **kwargs))
        return processes[-1]

    def open_log(*args, **kwargs):
        fail_at_the_selected_call("open")
        log_files.append(open(*args, **kwargs))
        return log_files[-1]

    class Thread(threading.Thread):
        def start(self):
            fail_at_the_selected_call("start")
            super().start()

    monkeypatch.setattr(
        local, "subprocess", SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE)
    )
    monkeypatch.setattr(local, "threading", SimpleNamespace(Thread=Thread))
    monkeypatch.setattr(local, "open", open_log, raising=False)

    with pytest.raises(type(error)) as raised:
        local.submit(str(tmp_path), SLEEP_COMMAND)
    returncode = processes[0].returncode
    pipes_closed = (processes[0].stdout.closed, processes[0].stderr.closed)
    processes[0].kill()
    processes[0].wait()

    assert raised.value is error
    assert returncode == -signal.SIGKILL
    assert pipes_closed == (True, True)
    assert len(log_files) == opened_logs
    assert all(log_file.closed for log_file in log_files)


def test_a_started_output_thread_writes_its_lines_before_the_logs_close(
    tmp_path, monkeypatch, caplog
):
    """The stderr thread start fails after the stdout thread started."""
    processes = []
    log_files = []
    threads = []

    def popen(*args, **kwargs):
        processes.append(subprocess.Popen(*args, **kwargs))
        # The line of the job is in the pipe before the launcher continues.
        select.select([processes[-1].stdout], [], [], 30.0)
        return processes[-1]

    def open_log(*args, **kwargs):
        log_files.append(open(*args, **kwargs))
        return log_files[-1]

    class Thread(threading.Thread):
        def start(self):
            threads.append(self)
            if len(threads) == 2:
                raise RuntimeError("can't start new thread")
            super().start()

        def run(self):
            # A slow thread: it reads its pipe after the error of the launcher.
            time.sleep(0.5)
            super().run()

    monkeypatch.setattr(
        local, "subprocess", SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE)
    )
    monkeypatch.setattr(local, "threading", SimpleNamespace(Thread=Thread))
    monkeypatch.setattr(local, "open", open_log, raising=False)

    with caplog.at_level(logging.ERROR, logger=local.logger.name):
        with pytest.raises(RuntimeError):
            local.submit(str(tmp_path), PRINT_COMMAND)
        pipes_closed = (processes[0].stdout.closed, processes[0].stderr.closed)
        threads[0].join(timeout=10.0)
    processes[0].kill()
    processes[0].wait()

    assert "Error in stream output thread" not in caplog.text
    assert pipes_closed == (True, True)
    assert len(log_files) == 2
    assert all(log_file.closed for log_file in log_files)
    assert (tmp_path / "job_log.out").read_text() == "line of the job\n"
