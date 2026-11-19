"""Graceful-stop flag set by SIGTERM / SIGINT.

Long loops call ``check_stop()`` after each batch; it raises ``JobInterrupted`` once a stop was
requested, after which the caller saves ``last.pt`` and the scheduler marks the job pending.
"""
from __future__ import annotations

import signal
import threading

_STOP = threading.Event()


class JobInterrupted(Exception):
    """Raised inside a job when SIGTERM/SIGINT was received."""


def _handler(signum, frame) -> None:  # noqa: ARG001
    _STOP.set()


def install_handlers() -> None:
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def stop_requested() -> bool:
    return _STOP.is_set()


def request_stop() -> None:
    _STOP.set()


def check_stop() -> None:
    if _STOP.is_set():
        raise JobInterrupted("stop requested")
