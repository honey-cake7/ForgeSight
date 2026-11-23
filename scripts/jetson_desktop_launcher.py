#!/usr/bin/env python3
"""Start the configured Jetson station and open it in a desktop app window."""

import argparse
from datetime import datetime
import fcntl
from http.client import HTTPConnection, HTTPException
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from threading import Thread
import time
from urllib.parse import urlsplit
from uuid import uuid4


# Backend initialization can wait 120 s for vision and 180 s for Llama,
# before binding HTTP. Allow another 60 s for imports and cleanup/fallback.
STATION_STARTUP_TIMEOUT_SECONDS = 360
BROWSER_STARTUP_CHECK_SECONDS = 2


def station_ready(port):
    connection = HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        connection.request("GET", "/api/state")
        response = connection.getresponse()
        if response.status != 200:
            return False
        state = json.loads(response.read(2 * 1024 * 1024))
        return isinstance(state, dict) and {"camera", "history", "advice"} <= state.keys()
    except (OSError, HTTPException, ValueError):
        return False
    finally:
        connection.close()


def browser_executable():
    for name in ("chromium", "chromium-browser", "google-chrome", "microsoft-edge"):
        path = shutil.which(name)
        if path:
            return path
    return None


def webkit_window():
    # The Jetson desktop can create the WebKit view and load its title while
    # leaving the entire content area blank on the accelerated compositor.
    os.environ.setdefault("WEBKIT_DISABLE_COMPOSITING_MODE", "1")
    try:
        import gi
        gi.require_version("Gtk", "3.0")
        gi.require_version("WebKit2", "4.0")
        from gi.repository import GLib, Gtk, WebKit2
    except (ImportError, ValueError) as error:
        raise RuntimeError("ForgeSight needs GTK WebKit or a Chromium-compatible browser.") from error

    def open_window(port):
        app = Gtk.Application(application_id="local.forgesight.Station")
        print_jobs = {}

        def local_uri(uri):
            target = urlsplit(uri)
            return target.scheme == "http" and target.netloc == f"127.0.0.1:{port}"

        def is_report(uri):
            return (local_uri(uri) and re.fullmatch(
                r"/api/history/[0-9a-fA-F-]{36}/report\.html", urlsplit(uri).path))

        def save_report(view, uri):
            def feedback(message):
                script = ("window.dispatchEvent(new CustomEvent('forgesight-report-status',"
                          f"{{detail:{json.dumps(message)}}}));")
                view.run_javascript(script, None, None, None)

            def finish(message, error=None):
                print_jobs.pop(destination, None)
                if error:
                    print(f"ForgeSight PDF export: {error}", file=sys.stderr)
                feedback(message)
                app.release()

            folder = Path(GLib.get_user_special_dir(GLib.UserDirectory.DIRECTORY_DESKTOP)
                          or (Path.home() / "Desktop")) / "ForgeSight Results"
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            destination = folder / f"ForgeSight-report-{stamp}-{uuid4().hex[:8]}.pdf"
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                print(f"ForgeSight PDF export: {error}", file=sys.stderr)
                feedback("Could not create the ForgeSight Results folder on the Desktop.")
                return

            def render_report(html):
                print_view = WebKit2.WebView.new_with_context(view.get_context())
                print_view.get_settings().set_enable_javascript(False)
                print_jobs[destination] = print_view

                def render_pdf(web_view, event):
                    if event != WebKit2.LoadEvent.FINISHED:
                        return
                    web_view.disconnect(handler)
                    settings = Gtk.PrintSettings.new()
                    settings.set_printer("Print to File")
                    settings.set(Gtk.PRINT_SETTINGS_OUTPUT_FILE_FORMAT, "pdf")
                    settings.set(Gtk.PRINT_SETTINGS_OUTPUT_URI, destination.as_uri())
                    page = Gtk.PageSetup.new()
                    page.set_paper_size(Gtk.PaperSize.new(Gtk.PAPER_NAME_A4))
                    job = WebKit2.PrintOperation.new(web_view)
                    job.set_print_settings(settings)
                    job.set_page_setup(page)
                    print_jobs[destination] = (web_view, job)

                    failures = []
                    job.connect("failed", lambda _, error: failures.append(error))

                    def printed(_):
                        if failures or not destination.is_file():
                            destination.unlink(missing_ok=True)
                            finish("Could not save the PDF.", failures[0] if failures else destination)
                        else:
                            finish(f"PDF saved in Desktop / ForgeSight Results / {destination.name}")

                    job.connect("finished", printed)
                    job.print_()

                handler = print_view.connect("load-changed", render_pdf)
                print_view.load_html(html, f"http://127.0.0.1:{port}/")

            def fetch_report():
                connection = HTTPConnection("127.0.0.1", port, timeout=15)
                try:
                    connection.request("GET", urlsplit(uri).path)
                    response = connection.getresponse()
                    if response.status != 200 or response.headers.get_content_type() != "text/html":
                        raise ValueError(f"Report unavailable (HTTP {response.status}).")
                    GLib.idle_add(render_report, response.read().decode("utf-8"))
                except (OSError, HTTPException, ValueError) as error:
                    GLib.idle_add(finish, "Could not load the saved inspection for PDF export.", error)
                finally:
                    connection.close()

            app.hold()
            feedback("Saving PDF…")
            Thread(target=fetch_report, daemon=True).start()

        def activate(application):
            window = application.get_active_window()
            if window is None:
                icon = launcher_icon()
                view = WebKit2.WebView()
                view.get_settings().set_hardware_acceleration_policy(
                    WebKit2.HardwareAccelerationPolicy.NEVER)
                view.get_settings().set_user_agent_with_application_details("ForgeSight", "1.0")

                def new_window(web_view, width, height):
                    panel = Gtk.ApplicationWindow(application=application, title="ForgeSight")
                    panel.set_default_size(width, height)
                    if icon.is_file():
                        panel.set_icon_from_file(str(icon))
                    panel.add(web_view)
                    web_view.connect("notify::title", lambda page, _: panel.set_title(page.get_title() or "ForgeSight"))
                    web_view.connect("close", lambda *_: panel.close())
                    web_view.connect("create", create_view)
                    web_view.connect("decide-policy", decide_policy)
                    return panel

                def create_view(parent, navigation):
                    if not local_uri(navigation.get_request().get_uri()):
                        return None
                    child = WebKit2.WebView.new_with_related_view(parent)
                    panel = new_window(child, 1160, 820)
                    child.connect("ready-to-show", lambda *_: (panel.show_all(), panel.present()))
                    return child

                def decide_policy(web_view, decision, decision_type):
                    if decision_type == WebKit2.PolicyDecisionType.NAVIGATION_ACTION:
                        uri = decision.get_navigation_action().get_request().get_uri()
                        if not is_report(uri):
                            return False
                        decision.ignore()
                        save_report(web_view, uri)
                        return True
                    return False

                window = new_window(view, 1280, 820)
                view.load_uri(f"http://127.0.0.1:{port}/")
                window.show_all()
            try:
                from gi.repository import GdkX11
                timestamp = GdkX11.x11_get_server_time(window.get_window())
            except (ImportError, AttributeError, TypeError):
                timestamp = 0
            window.present_with_time(timestamp)

        app.connect("activate", activate)
        return app.run([])

    return open_window


def run_webkit_window(port, open_window):
    status = open_window(port)
    if status != 0:
        raise RuntimeError(f"GTK WebKit exited with status {status}.")
    return 0


def load_station(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    command = config["command"]
    port = config["port"]
    if (type(port) is not int or not 1024 <= port <= 65535
            or not isinstance(command, list) or not command
            or not all(isinstance(value, str) and value for value in command)):
        raise ValueError("The station configuration must contain a port and a startup command.")
    return port, command, Path(config.get("cwd", str(path.parent))).resolve()


def start_station(port, command, working_dir, cache):
    with (cache / "station.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if station_ready(port):
            return
        log_path = cache / "station.log"
        with log_path.open("ab") as log:
            service = subprocess.Popen(
                command, cwd=working_dir, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        deadline = time.monotonic() + STATION_STARTUP_TIMEOUT_SECONDS
        while service.poll() is None and time.monotonic() < deadline:
            if station_ready(port):
                if "--runs-root" in command:
                    runs_root = working_dir / command[command.index("--runs-root") + 1]
                    pid_path = runs_root / "app.pid"
                    if pid_path.is_file():
                        pending = pid_path.with_suffix(".launcher-tmp")
                        pending.write_text(f"{service.pid}\n", encoding="utf-8")
                        pending.replace(pid_path)
                return
            time.sleep(0.6)
        # The subprocess may still be starting. Track its group and stop that
        # group before releasing the launch lock to another attempt.
        try:
            os.killpg(service.pid, 15)
        except ProcessLookupError:
            pass
        try:
            service.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(service.pid, 9)
            service.wait(timeout=5)
        raise RuntimeError(f"The dashboard did not start. Check {log_path}.")


def launcher_icon():
    data = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    return data / "icons/forgesight-station.png"


def install_entry(station):
    script = Path(__file__).resolve()
    # Launcher paths are local deployment paths, with any spaces quoted.
    paths = [str(script), str(station)]
    if any(any(char in value for char in '"\\\n\r%$`') for value in paths):
        raise ValueError("The desktop launcher paths contain unsupported characters.")
    icon = launcher_icon()
    desktop_file = (
        "[Desktop Entry]\nType=Application\nName=ForgeSight\n"
        "Comment=Open the offline inspection station\n"
        f'Exec=/usr/bin/python3 -B "{script}" --station "{station}"\n'
        f"Icon={icon if icon.is_file() else 'applications-graphics'}\n"
        "StartupWMClass=Forgesight_desktop_launcher.py\nTerminal=false\nCategories=Utility;\n")
    data = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    applications = data / "applications"
    applications.mkdir(parents=True, exist_ok=True)
    entry = applications / "forgesight.desktop"
    entry.write_text(desktop_file, encoding="utf-8")
    print(entry)
    desktop = Path.home() / "Desktop"
    if shutil.which("xdg-user-dir"):
        target = subprocess.check_output(["xdg-user-dir", "DESKTOP"], text=True).strip()
        if target:
            desktop = Path(target)
    if desktop.is_dir() and desktop != Path.home():
        entry = desktop / "ForgeSight.desktop"
        entry.write_text(desktop_file, encoding="utf-8")
        entry.chmod(0o755)
        print(entry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", type=Path, required=True,
                        help="The existing Jetson station activation JSON.")
    parser.add_argument("--install-icon", action="store_true",
                        help="Add ForgeSight to the application menu and desktop.")
    args = parser.parse_args()
    station = args.station.resolve()
    try:
        port, command, working_dir = load_station(station)
        if args.install_icon:
            install_entry(station)
            return 0
        browser = browser_executable()
        open_window = None if browser else webkit_window()
        cache_home = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        cache = cache_home / "forgesight-desktop" / str(port)
        cache.mkdir(parents=True, exist_ok=True)
        start_station(port, command, working_dir, cache)
        if open_window is not None:
            return run_webkit_window(port, open_window)
        browser_log = cache / "browser.log"
        try:
            with browser_log.open("ab") as log:
                viewer = subprocess.Popen(
                    [browser, f"--user-data-dir={cache / 'browser'}",
                     f"--app=http://127.0.0.1:{port}/", "--no-first-run"],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = viewer.wait(timeout=BROWSER_STARTUP_CHECK_SECONDS)
            except subprocess.TimeoutExpired:
                return 0  # The viewer is still running after the startup check.
            if status == 0:
                return 0  # A running browser may accept the window and exit its wrapper.
            browser_error = f"{browser} exited with status {status}"
        except OSError as error:
            browser_error = f"{browser} could not start: {error}"
        try:
            return run_webkit_window(port, webkit_window())
        except (OSError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                f"Browser failed: {browser_error}; GTK fallback failed: {error}. "
                f"Check {browser_log}.") from error
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ForgeSight: {error}", file=sys.stderr)
        if shutil.which("notify-send"):
            subprocess.run(["notify-send", "ForgeSight could not start", str(error)], check=False)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
