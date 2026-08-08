from __future__ import annotations

import os
import signal
import subprocess
import threading


class ProcessRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._stop_events: dict[str, threading.Event] = {}

    def register(self, task_id: str, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._processes[task_id] = process
            self._stop_events.setdefault(task_id, threading.Event())

    def unregister(self, task_id: str) -> None:
        with self._lock:
            self._processes.pop(task_id, None)
            self._stop_events.pop(task_id, None)

    def request_stop(self, task_id: str) -> bool:
        with self._lock:
            event = self._stop_events.setdefault(task_id, threading.Event())
            event.set()
            process = self._processes.get(task_id)
        if process is None or process.poll() is not None:
            return False
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except OSError:
            return False
        return True

    def stop_requested(self, task_id: str) -> bool:
        with self._lock:
            event = self._stop_events.get(task_id)
            return event.is_set() if event else False

    def active(self, task_id: str) -> bool:
        with self._lock:
            process = self._processes.get(task_id)
            return process is not None and process.poll() is None


process_registry = ProcessRegistry()
