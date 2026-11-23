"""Exercise launcher failures without starting models or opening a real window."""

from contextlib import ExitStack, redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from unittest.mock import Mock, patch

if os.name == "posix":
    from scripts import jetson_desktop_launcher as launcher
else:
    launcher = None


@unittest.skipUnless(os.name == "posix", "The Jetson launcher uses POSIX file locking")
class DesktopLauncherTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory(prefix="forgesight launcher ")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.port = 18772
        self.command = ["/usr/bin/python3", "-m", "forgesight.jetson_live_ui"]
        self.station = self.root / "station.json"
        self.station.write_text(json.dumps({"port": self.port, "command": self.command,
                                            "cwd": str(self.root)}))

    def test_supported_combined_worker_startup_is_not_killed_at_120_seconds(self):
        clock = {"now": 0.0}
        process = Mock(pid=456)
        process.poll.return_value = None

        def sleep(seconds):
            clock["now"] += seconds

        with patch.object(launcher.time, "monotonic", side_effect=lambda: clock["now"]), \
                patch.object(launcher.time, "sleep", side_effect=sleep), \
                patch.object(launcher, "station_ready", side_effect=lambda _: clock["now"] >= 305), \
                patch.object(launcher.subprocess, "Popen", return_value=process), \
                patch.object(launcher.os, "killpg") as kill:
            try:
                launcher.start_station(self.port, self.command, self.root, self.cache)
            except RuntimeError as error:
                self.fail("Launcher killed a supported combined worker startup: " + str(error))
        kill.assert_not_called()

    def test_startup_timeout_still_stops_its_own_process_group(self):
        clock = {"now": 0.0}
        process = Mock(pid=456)
        process.poll.return_value = None

        def sleep(seconds):
            clock["now"] += seconds

        with patch.object(launcher.time, "monotonic", side_effect=lambda: clock["now"]), \
                patch.object(launcher.time, "sleep", side_effect=sleep), \
                patch.object(launcher, "station_ready", return_value=False), \
                patch.object(launcher.subprocess, "Popen", return_value=process), \
                patch.object(launcher.os, "killpg") as kill:
            with self.assertRaisesRegex(RuntimeError, "dashboard did not start"):
                launcher.start_station(self.port, self.command, self.root, self.cache)
        self.assertGreaterEqual(clock["now"], 300)
        self.assertLessEqual(clock["now"], 400)
        kill.assert_called_once_with(process.pid, 15)

    def test_existing_station_is_reused(self):
        with patch.object(launcher, "station_ready", return_value=True), \
                patch.object(launcher.subprocess, "Popen") as spawn:
            launcher.start_station(self.port, self.command, self.root, self.cache)
        spawn.assert_not_called()

    def run_main(self, process=None, spawn_error=None, gtk_error=None,
                 browser="/usr/bin/chromium", gtk_status=0):
        window = Mock(return_value=gtk_status)
        errors = StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["launcher", "--station", str(self.station)]))
            stack.enter_context(patch.dict(os.environ, {"XDG_CACHE_HOME": str(self.cache)}))
            stack.enter_context(patch.object(launcher, "browser_executable", return_value=browser))
            stack.enter_context(patch.object(launcher, "start_station"))
            stack.enter_context(patch.object(launcher.shutil, "which", return_value=None))
            stack.enter_context(patch.object(launcher.subprocess, "Popen", return_value=process,
                                            side_effect=spawn_error))
            gtk = stack.enter_context(patch.object(launcher, "webkit_window", return_value=window,
                                                   side_effect=gtk_error))
            stack.enter_context(redirect_stderr(errors))
            status = launcher.main()
        return status, gtk, window, errors.getvalue()

    def test_browser_early_failure_opens_webkit(self):
        process = Mock()
        process.wait.return_value = 1
        status, gtk, window, _ = self.run_main(process)
        self.assertEqual(status, 0)
        gtk.assert_called_once_with()
        window.assert_called_once_with(self.port)

    def test_browser_spawn_error_opens_webkit(self):
        status, gtk, window, _ = self.run_main(spawn_error=OSError("browser missing"))
        self.assertEqual(status, 0)
        gtk.assert_called_once_with()
        window.assert_called_once_with(self.port)

    def test_failed_browser_and_missing_webkit_report_failure(self):
        process = Mock()
        process.wait.return_value = 1
        status, _, _, error = self.run_main(process, gtk_error=RuntimeError("GTK unavailable"))
        self.assertEqual(status, 1)
        self.assertIn("GTK unavailable", error)
        self.assertIn("browser", error.lower())

    def test_nonzero_webkit_fallback_exit_reports_both_viewer_failures(self):
        process = Mock()
        process.wait.return_value = 1
        status, _, _, error = self.run_main(process, gtk_status=1)
        self.assertEqual(status, 1)
        self.assertIn("Browser failed", error)
        self.assertIn("GTK WebKit exited with status 1", error)

    def test_nonzero_webkit_only_exit_reports_failure(self):
        status, _, _, error = self.run_main(browser=None, gtk_status=1)
        self.assertEqual(status, 1)
        self.assertIn("GTK WebKit exited with status 1", error)

    def test_running_browser_does_not_open_a_second_viewer(self):
        process = Mock()
        process.wait.side_effect = subprocess.TimeoutExpired("chromium", 2)
        status, gtk, _, _ = self.run_main(process)
        self.assertEqual(status, 0)
        gtk.assert_not_called()

    def test_existing_browser_handoff_exit_zero_is_success(self):
        process = Mock()
        process.wait.return_value = 0
        status, gtk, _, _ = self.run_main(process)
        self.assertEqual(status, 0)
        gtk.assert_not_called()

    def test_webkit_only_path_remains_available(self):
        status, gtk, window, _ = self.run_main(browser=None)
        self.assertEqual(status, 0)
        gtk.assert_called_once_with()
        window.assert_called_once_with(self.port)

    def test_real_failed_browser_falls_back_and_keeps_existing_station(self):
        class ReadyHandler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                payload = b'{"camera": {}, "history": [], "advice": {}}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), ReadyHandler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_port
        self.station.write_text(json.dumps({"port": port, "command": self.command,
                                            "cwd": str(self.root)}))
        browser = self.root / "failed browser"
        browser.write_text("#!/bin/sh\nprintf 'browser probe failed\\n' >&2\nexit 7\n")
        browser.chmod(0o755)
        window = Mock(return_value=0)
        try:
            with patch.object(sys, "argv", ["launcher", "--station", str(self.station)]), \
                    patch.dict(os.environ, {"XDG_CACHE_HOME": str(self.cache)}), \
                    patch.object(launcher, "browser_executable", return_value=str(browser)), \
                    patch.object(launcher, "webkit_window", return_value=window):
                self.assertEqual(launcher.main(), 0)
            window.assert_called_once_with(port)
            self.assertTrue(launcher.station_ready(port))
            log = self.cache / "forgesight-desktop" / str(port) / "browser.log"
            self.assertIn("browser probe failed", log.read_text())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
