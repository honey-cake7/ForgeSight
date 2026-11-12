"""The inspection path must stay available while a bounded advice job runs."""

from threading import Event
import time
import unittest

from forgesight.assist_queue import AdvisoryQueue


class AdvisoryQueueTests(unittest.TestCase):
    def test_busy_refusal_and_inspection_path(self):
        entered, release = Event(), Event()
        queue = AdvisoryQueue()
        try:
            def slow_advice():
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("test worker deadline")
                return {"status": "GENERATED"}

            first = queue.submit(slow_advice)
            self.assertTrue(entered.wait(1))
            started = time.perf_counter()
            self.assertIsNone(queue.submit(lambda: {"status": "DUPLICATE"}))
            busy_ms = (time.perf_counter() - started) * 1000
            self.assertLess(busy_ms, 100)
            inspection = {"grade": "REVIEW", "advice_status": "PENDING"}
            self.assertEqual(inspection["grade"], "REVIEW")
            self.assertFalse(first.done())
            release.set()
            self.assertEqual(first.result(timeout=2)["status"], "GENERATED")
            self.assertEqual(queue.submit(lambda: {"status": "NEXT"}).result(timeout=2)["status"], "NEXT")
        finally:
            release.set()
            queue.close()

    def test_failure_is_available_as_status_and_queue_recovers(self):
        queue = AdvisoryQueue()
        try:
            def failed():
                raise TimeoutError("runtime expired")

            self.assertEqual(queue.submit(failed).result(timeout=2),
                             {"status": "ADVICE_UNAVAILABLE", "error": "TimeoutError"})
            self.assertEqual(queue.submit(lambda: "ready").result(timeout=2), "ready")
        finally:
            queue.close()
        self.assertIsNone(queue.submit(lambda: "closed"))


if __name__ == "__main__":
    unittest.main()
