from __future__ import annotations

from contextlib import contextmanager
import threading
from collections.abc import Iterator


class LocalComputeGate:
    """Serializes local training and automatic annotation on one machine."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @contextmanager
    def acquire(self) -> Iterator[None]:
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()


local_compute_gate = LocalComputeGate()
