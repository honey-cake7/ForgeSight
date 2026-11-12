"""One outstanding advisory at a time; inspection callers never wait for Llama."""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock


class AdvisoryQueue:
    """Run one advisory in the background and refuse overlap immediately.

    The worker must enforce its own runtime deadline. The Jetson advisory
    controller does this with a subprocess timeout and writes its own report.
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forgeassist")
        self._lock = Lock()
        self._busy = False
        self._closed = False

    def submit(self, worker):
        """Return a Future, or None if a job is running or the queue is closed."""
        with self._lock:
            if self._busy or self._closed:
                return None
            self._busy = True
            try:
                return self._executor.submit(self._run, worker)
            except RuntimeError:
                self._busy = False
                return None

    def _run(self, worker):
        try:
            return worker()
        except Exception as error:
            return {"status": "ADVICE_UNAVAILABLE", "error": type(error).__name__}
        finally:
            with self._lock:
                self._busy = False

    def close(self):
        """Reject new work; an accepted job may finish under its own deadline."""
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=False)
