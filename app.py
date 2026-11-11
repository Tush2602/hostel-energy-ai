"""
app.py - START THE LIVE SYSTEM:   python app.py      then open  http://127.0.0.1:8000

It starts the monitor (watches the live CSV, trips, alerts, forecasts) in the background
and serves the dashboard. The dashboard asks /api/state every 2 seconds for fresh numbers.
"""
import logging
import threading
import webbrowser
import math
from flask import Flask, jsonify, send_from_directory

import config
from monitor import Monitor

app = Flask(__name__, static_folder=None)
logging.getLogger("werkzeug").setLevel(logging.ERROR)      # keep the console for alerts / trips only
monitor = Monitor()


@app.route("/")
def index():
    return send_from_directory("dashboard", "index.html")


@app.route("/<path:name>")
def files(name):
    return send_from_directory("dashboard", name)


def no_nan(x):
    """Blank readings (NaN) are not allowed in JSON -> send them as null (empty) instead."""
    if isinstance(x, dict):
        return {k: no_nan(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [no_nan(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


@app.route("/api/state")
def state():
    return jsonify(no_nan(monitor.get_state()))


if __name__ == "__main__":
    print(f"Watching   : {config.LIVE_CSV}")
    print(f"AI models  : forecast {'OK' if monitor.forecaster else 'MISSING (run train.py)'}, "
          f"anomaly {'OK' if monitor.anomaly else 'MISSING (run train.py)'}")
    print(f"TRIP rules : > {config.TRIP_INSTANT_W} W instantly, or > {config.TRIP_SUSTAINED_W} W "
          f"for {config.TRIP_SUSTAINED_SECONDS} s")
    stop = threading.Event()
    threading.Thread(target=monitor.run_forever, args=(stop,), daemon=True).start()
    url = f"http://127.0.0.1:{config.DASHBOARD_PORT}"
    print(f"Dashboard  : {url}   (Ctrl+C to stop)\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=config.DASHBOARD_PORT, debug=False, threaded=True)
