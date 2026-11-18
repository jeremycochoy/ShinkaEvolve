"""The local launcher when a step after the start of the job process has an error."""

import errno
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from shinka.launch import local

SLEEP_COMMAND = [sys.executable, "-c", "import time; time.sleep(60)"]


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
    processes[0].kill()
    processes[0].wait()

    assert raised.value is error
    assert returncode == -signal.SIGKILL
    assert len(log_files) == opened_logs
    assert all(log_file.closed for log_file in log_files)
