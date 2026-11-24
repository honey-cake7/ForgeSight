"""Exit an owned worker when its station process exits, not its launch thread."""
import os
import signal
import sys
from threading import Thread
import time


def watch_station_parent():
    value = os.environ.get('FORGESIGHT_STATION_PID')
    if not value or not sys.platform.startswith('linux'):
        return
    parent_pid = int(value)
    if parent_pid <= 0:
        raise ValueError('Invalid station process identity.')

    def watch():
        # Linux reparents children when the parent process exits. A short-lived
        # respawn thread exiting leaves getppid() pointing at the live station.
        while os.getppid() == parent_pid:
            time.sleep(.25)
        os.kill(os.getpid(), signal.SIGTERM)

    Thread(target=watch, name='station-lifetime', daemon=True).start()
