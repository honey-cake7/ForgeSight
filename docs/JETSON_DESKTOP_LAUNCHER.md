# ForgeSight desktop launcher on Jetson

The launcher opens the installed ForgeSight station in a dedicated browser app
window. It uses a Chromium-compatible browser or the installed GTK WebKit
viewer and the JSON from the successful station activation. It uses the recorded startup command and
checks the dashboard on loopback. It does not contact the network to start
ForgeSight. Launch it from the Jetson graphical session.

Copy `scripts/jetson_desktop_launcher.py` to a persistent path on Jetson and
`assets/forgesight-station.png` to `~/.local/share/icons/forgesight-station.png`, then
run the following with the path to the active station's configuration.

```sh
python3 -B ~/.local/bin/forgesight_desktop_launcher.py \
  --station /absolute/path/to/station_activation.json --install-icon
```

Open **ForgeSight** from the application menu or desktop. Some desktop
environments require selecting **Allow Launching** on a new desktop icon.
The launcher checks for an existing station before starting another. A launch
failure is reported through the desktop notification service when available.
Dashboard startup output is under `~/.cache/forgesight-desktop/<port>/station.log`.

The startup deadline is six minutes: the backend may wait up to two minutes
for vision and three minutes for Llama before opening HTTP, with one minute
reserved for imports and fallback cleanup. The launcher stops its own process
group if this deadline expires.

If Chromium cannot be executed or exits unsuccessfully during the two-second
startup check, the launcher tries GTK WebKit. A running Chromium process or
a successful handoff to an existing browser is accepted. Browser output is
saved beside the station log as `browser.log`; failure of both viewers is
reported through the existing notification path. The short check does not
detect a browser failure that occurs later.

To remove the shortcut, remove the generated application entry and desktop
icon. The station, saved inspections, and desktop-independent startup command
are separate from these shortcuts.
