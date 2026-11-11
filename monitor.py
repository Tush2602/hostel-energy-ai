"""
monitor.py - the LIVE brain of the project.

Every POLL_SECONDS it:
    1. reads the NEW rows the hardware added to the live CSV
    2. cleans them (same cleaning as training)
    3. SAFETY RULES  -> TRIP signal (printed on screen) if power > 1500 W, or > 1000 W for 30 s
    4. AI anomaly model -> which heating appliance is running (kettle / iron / induction / heater)
    5. other alerts: sensor fault, over/under voltage, no data
    6. at midnight: AI forecast of the next 24 hours is made and saved ("day-ahead request to substation")
It also prepares all numbers for the dashboard (get_state).
"""
import glob
import io
import os
import threading
from collections import deque

import numpy as np
import pandas as pd

import config
from data_cleaning import clean_and_add_energy, hourly_energy, prepare_columns
from models import NORMAL, AnomalyModel, ForecastModel, anomaly_features, full_hourly_series

RED, YELLOW, GREEN, BOLD, END = "\033[91m", "\033[93m", "\033[92m", "\033[1m", "\033[0m"


# ====================================================================== reading the growing CSV
class CsvFollower:
    """Remembers how far the file was read, so every call returns ONLY the new complete rows."""

    def __init__(self, path):
        self.path, self.file, self.offset, self.header = path, None, 0, None

    def current_file(self):
        if os.path.isdir(self.path):                              # a folder -> newest CSV inside
            files = glob.glob(os.path.join(self.path, "*.csv"))
            return max(files, key=os.path.getmtime) if files else None
        return self.path if os.path.exists(self.path) else None

    def read_new(self):
        f = self.current_file()
        if f is None:
            return None
        if f != self.file or os.path.getsize(f) < self.offset:     # new file, or file was restarted
            self.file, self.offset, self.header = f, 0, None
        with open(f, "rb") as fh:
            fh.seek(self.offset)
            data = fh.read()
        end = data.rfind(b"\n")
        if end < 0:                                                # no complete new line yet
            return None
        self.offset += end + 1
        lines = data[:end + 1].decode("utf-8-sig", errors="replace").splitlines()
        if self.header is None:
            self.header, lines = lines[0], lines[1:]
        lines = [ln for ln in lines if ln.strip()]
        if not lines:
            return None
        df = pd.read_csv(io.StringIO(self.header + "\n" + "\n".join(lines)), on_bad_lines="skip")
        return prepare_columns(df)


# ====================================================================== the monitor
class Monitor:
    def __init__(self):
        self.reader = CsvFollower(config.LIVE_CSV)
        self.data = pd.DataFrame()
        self.lock = threading.Lock()
        self.version = 0
        self._cache = (None, None)
        self.forecaster = ForecastModel.load(config.FORECAST_MODEL) if os.path.exists(config.FORECAST_MODEL) else None
        self.anomaly = AnomalyModel.load(config.ANOMALY_MODEL) if os.path.exists(config.ANOMALY_MODEL) else None
        self.metrics = pd.read_json(config.METRICS_FILE, typ="series").to_dict() if os.path.exists(config.METRICS_FILE) else {}
        self.forecasts = self._load_csv(config.FORECAST_LOG, ["date", "time", "forecast_wh", "made_at"])
        self.events = deque(self._load_csv(config.EVENT_LOG, ["time", "type", "severity", "message"])
                            .tail(50).to_dict("records"), maxlen=50)
        # live state of the rules
        self.active = {}                    # alert type -> message (conditions happening right now)
        self.clear_since = {}               # alert type -> time its condition disappeared
        self.above_since = None             # when power first went above TRIP_SUSTAINED_W
        self.last_trip = None               # dict(time, reason, power)
        self.ml_streak, self.ml_kind, self.ml_prob, self.ml_power = 0, NORMAL, 0.0, 0.0
        self.ml_votes = deque(maxlen=10)
        self.started = False
        self.no_data_logged = False

    @staticmethod
    def _load_csv(path, cols):
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=cols)

    # ------------------------------------------------------------------ one poll
    def poll(self):
        new = self.reader.read_new()
        with self.lock:
            if new is not None and len(new):
                first_time = not self.started
                self._add_rows(new)
                if first_time:
                    print(f"Loaded {len(self.data)} rows of history from {self.reader.file} (no alerts for old rows)")
                else:
                    for row in self.data.tail(len(new)).itertuples(index=False):
                        self._check_rules(row)
                self.started = True
                self._make_forecasts()
                self.version += 1
            self._check_no_data()

    def _add_rows(self, new):
        if len(self.data):
            last_time = self.data["time"].iloc[-1]
            new = new[new["time"] > last_time]                     # ignore rows we already have
            if new.empty:
                return
        n_new = len(new)
        keep = ["time", "voltage", "current", "power", "energy", "pf", "frequency"]
        combined = pd.concat([self.data[keep], new[keep]], ignore_index=True) if len(self.data) else new[keep]
        cleaned = clean_and_add_energy(combined)
        # AI anomaly model on the new rows (with 3 minutes of older rows as context for the rolling features)
        cleaned["appliance"], cleaned["heating_prob"] = NORMAL, 0.0
        if len(self.data):
            cleaned.loc[:len(self.data) - 1, ["appliance", "heating_prob"]] = \
                self.data[["appliance", "heating_prob"]].to_numpy()
        if self.anomaly is not None:
            first_new = len(cleaned) - n_new
            ctx_start = cleaned["time"].iloc[first_new] - pd.Timedelta(minutes=3)
            ctx = cleaned[cleaned["time"] >= ctx_start]
            kind, prob = self.anomaly.predict(anomaly_features(ctx))
            ok = (ctx["status"] == "OK").to_numpy()
            kind, prob = np.where(ok, kind, NORMAL), np.where(ok, prob, 0.0)
            cleaned.loc[ctx.index[-n_new:], "appliance"] = kind[-n_new:]
            cleaned.loc[ctx.index[-n_new:], "heating_prob"] = prob[-n_new:]
        self.data = cleaned

    # ------------------------------------------------------------------ rules for every new reading
    def _check_rules(self, r):
        now = r.time
        if r.status == "SENSOR_FAULT":
            self._condition("SENSOR_FAULT", True, "WARNING", "Sensor fault / supply off (all readings 0)", now)
            return
        self._condition("SENSOR_FAULT", False, "", "", now)

        # ---- SAFETY: trip rules (fixed rules, not ML)
        if r.power > config.TRIP_SUSTAINED_W:
            self.above_since = self.above_since or now
        else:
            self.above_since = None
        reason = None
        if r.power > config.TRIP_INSTANT_W:
            reason = f"Instant overload: {r.power:.0f} W > {config.TRIP_INSTANT_W} W"
        elif self.above_since is not None and (now - self.above_since).total_seconds() >= config.TRIP_SUSTAINED_SECONDS:
            secs = (now - self.above_since).total_seconds()
            reason = f"Sustained overload: above {config.TRIP_SUSTAINED_W} W for {secs:.0f} s (now {r.power:.0f} W)"
        if reason and (self.last_trip is None or
                       (now - self.last_trip["time"]).total_seconds() > config.TRIP_HOLD_SECONDS):
            self._send_trip(now, reason, r.power)

        # ---- voltage (with 3 V hysteresis, like a relay, so an alert does not flicker on/off at the limit)
        over = r.voltage > config.OVER_VOLTAGE_V - (3 if "OVER_VOLTAGE" in self.active else 0)
        under = r.voltage < config.UNDER_VOLTAGE_V + (3 if "UNDER_VOLTAGE" in self.active else 0)
        self._condition("OVER_VOLTAGE", over, "WARNING",
                        f"Over-voltage {r.voltage:.0f} V (limit {config.OVER_VOLTAGE_V} V)", now)
        self._condition("UNDER_VOLTAGE", under, "WARNING",
                        f"Under-voltage {r.voltage:.0f} V (limit {config.UNDER_VOLTAGE_V} V)", now)

        # ---- AI: heating appliance (needs N heating readings in a row to start an alert)
        heating = r.appliance != NORMAL and r.heating_prob >= config.ANOMALY_PROBABILITY
        self.ml_streak = self.ml_streak + 1 if heating else 0
        if heating:
            self.ml_votes.append(r.appliance)              # appliance type = majority of the last 10 votes
            self.ml_prob, self.ml_power = r.heating_prob, r.power
        if self.ml_streak >= config.ANOMALY_CONFIRM_READINGS:
            kind = max(set(self.ml_votes), key=list(self.ml_votes).count)
            banned = kind in config.BANNED_APPLIANCES
            message = (f"AI: {kind.upper()} detected ({self.ml_prob:.0%} sure, {self.ml_power:.0f} W)"
                       + (" - NOT ALLOWED in hostel rooms" if banned else ""))
            if "ANOMALY" in self.active and kind != self.ml_kind:      # type became clearer (e.g. heater -> iron)
                self.active["ANOMALY"] = message
                self._log(now, "ANOMALY", "ALERT" if banned else "INFO", message + " (updated)")
            self.ml_kind = kind
            self._condition("ANOMALY", True, "ALERT" if banned else "INFO", message, now)
        elif not heating:
            self._condition("ANOMALY", False, "", "", now)
            if "ANOMALY" not in self.active:
                self.ml_votes.clear()
                self.ml_kind = NORMAL

    def _send_trip(self, now, reason, power):
        self.last_trip = {"time": now, "reason": reason, "power": float(power)}
        line = "=" * 78
        print(f"\n{RED}{BOLD}{line}\n  >>> TRIP SIGNAL SENT TO RELAY <<<   {now:%Y-%m-%d %H:%M:%S}\n  {reason}\n{line}{END}\n")
        self._log(now, "TRIP", "CRITICAL", f"TRIP signal sent. {reason}")

    def _condition(self, kind, is_on, severity, message, now):
        """Log an alert once when it STARTS and once when it ENDS (no spam every 20 s).
        It only ends after the condition has been gone for ALERT_END_SECONDS (no flickering)."""
        if is_on:
            self.clear_since.pop(kind, None)
            if kind not in self.active:
                self.active[kind] = message
                self._log(now, kind, severity, message)
        elif kind in self.active:
            since = self.clear_since.setdefault(kind, now)
            if (now - since).total_seconds() >= config.ALERT_END_SECONDS:
                del self.active[kind], self.clear_since[kind]
                ended = {"ANOMALY": "Heating appliance switched off", "SENSOR_FAULT": "Sensor back to normal"}
                self._log(now, kind + "_END", "INFO", ended.get(kind, kind.replace("_", " ").title() + " ended"))

    def _check_no_data(self):
        if not self.started:
            return
        age = (pd.Timestamp.now(tz=config.TIMEZONE) - self.data["time"].iloc[-1]).total_seconds()
        if age > config.NO_DATA_SECONDS and not self.no_data_logged:
            self.no_data_logged = True
            self._log(pd.Timestamp.now(tz=config.TIMEZONE), "NO_DATA", "WARNING",
                      f"No new reading for {age:.0f} s - check the hardware / logger")
        elif age <= config.NO_DATA_SECONDS:
            self.no_data_logged = False

    def _log(self, time, kind, severity, message):
        ev = {"time": pd.Timestamp(time).strftime("%Y-%m-%d %H:%M:%S"), "type": kind, "severity": severity,
              "message": message}
        self.events.append(ev)
        pd.DataFrame([ev]).to_csv(config.EVENT_LOG, mode="a", header=not os.path.exists(config.EVENT_LOG), index=False)
        colour = {"CRITICAL": RED, "ALERT": RED, "WARNING": YELLOW}.get(severity, GREEN)
        if kind != "TRIP":
            print(f"{colour}[{ev['time']}] {severity:8s} {message}{END}")

    # ------------------------------------------------------------------ forecasting (at midnight)
    def normal_hourly(self):
        """Hourly energy WITHOUT banned appliances (= the normal demand the forecast predicts)."""
        d = self.data
        banned = d["appliance"].isin(config.BANNED_APPLIANCES)
        normal = hourly_energy(d.assign(energy_wh=d["energy_wh"].where(~banned, 0.0)))["energy_wh"]
        total = hourly_energy(d)["energy_wh"]
        return full_hourly_series(normal), full_hourly_series(total)

    def _make_forecasts(self):
        """Make sure a forecast exists for today (and for the previous days we have data for).
        A forecast for day D uses ONLY data from before D 00:00 - exactly what would have been
        known at midnight - so it is always an honest 'made the night before' forecast."""
        if self.forecaster is None:
            return
        today = self.data["time"].iloc[-1].normalize()
        first_day = self.data["time"].iloc[0].normalize()
        done = set(self.forecasts["date"].astype(str))
        normal, _ = self.normal_hourly()
        for k in range(7, -1, -1):
            day = today - pd.Timedelta(days=k)
            if day < first_day or str(day.date()) in done:
                continue
            fc = self.forecaster.predict_day(normal, day)
            rows = pd.DataFrame({"date": str(day.date()), "time": fc["time"].dt.strftime("%Y-%m-%d %H:%M"),
                                 "forecast_wh": fc["forecast_wh"].round(2),
                                 "made_at": pd.Timestamp.now(tz=config.TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")})
            rows.to_csv(config.FORECAST_LOG, mode="a", header=not os.path.exists(config.FORECAST_LOG), index=False)
            self.forecasts = pd.concat([self.forecasts, rows], ignore_index=True)
            peak = fc.loc[fc["forecast_wh"].idxmax()]
            print(f"{GREEN}[forecast] Day-ahead request for {day.date()}: {fc['forecast_wh'].sum() / 1000:.3f} kWh "
                  f"in 24 hourly values, peak hour {peak['time']:%H}:00 ({peak['forecast_wh']:.0f} Wh){END}")

    # ------------------------------------------------------------------ everything the dashboard needs
    def get_state(self):
        with self.lock:
            if not self.started or self.data.empty:
                return {"ready": False, "csv": config.LIVE_CSV}
            minute = pd.Timestamp.now(tz=config.TIMEZONE).floor("min")
            if self._cache[0] == (self.version, minute):
                state = dict(self._cache[1])
            else:
                state = self._build_state()
                self._cache = ((self.version, minute), state)
                state = dict(state)
            # fast-changing parts are always fresh
            age = (pd.Timestamp.now(tz=config.TIMEZONE) - self.data["time"].iloc[-1]).total_seconds()
            state["age_seconds"] = age
            state["online"] = age <= config.NO_DATA_SECONDS
            state["active_alerts"] = list(self.active.values())
            held = "ANOMALY" in self.active                           # confirmed alert, not one noisy reading
            state["appliance"] = {"kind": self.ml_kind if held else NORMAL, "prob": self.ml_prob if held else 0.0}
            state["events"] = list(self.events)[::-1][:25]
            trip = self.last_trip
            state["trip"] = None if trip is None else {
                "time": trip["time"].strftime("%H:%M:%S"), "reason": trip["reason"],
                "active": (self.data["time"].iloc[-1] - trip["time"]).total_seconds() <= config.TRIP_HOLD_SECONDS}
            return state

    def _build_state(self):
        d = self.data
        last = d.iloc[-1]
        now = last["time"]
        today = now.normalize()

        # live power, last 30 minutes
        recent = d[d["time"] >= now - pd.Timedelta(minutes=30)]
        live = {"t": recent["time"].dt.strftime("%H:%M:%S").tolist(),
                "p": recent["power"].where(recent["status"] == "OK").round(1).tolist()}

        # today: forecast (made at midnight) vs actual, hour by hour
        normal, total = self.normal_hourly()
        hours = pd.date_range(today, periods=24, freq="h")
        fc_today = self.forecasts[self.forecasts["date"].astype(str) == str(today.date())]
        forecast = fc_today["forecast_wh"].astype(float).round(1).tolist() if len(fc_today) == 24 else [None] * 24
        act_normal = normal.reindex(hours)
        act_banned = (total - normal).reindex(hours)
        done = hours < now.floor("h")                               # completed hours
        actual = [round(v, 1) if (ok and pd.notna(v)) else None for v, ok in zip(act_normal, done)]
        banned = [round(v, 1) if (ok and pd.notna(v) and v > 0.05) else None for v, ok in zip(act_banned, done)]
        running = total.reindex(hours).iloc[now.hour]                 # this hour so far (everything)
        running = float(running) if pd.notna(running) else 0.0

        # today so far (completed hours): forecast energy vs actual energy
        pairs = [(f, a) for f, a in zip(forecast, actual) if f is not None and a is not None]
        fc_sofar, act_sofar = sum(f for f, _ in pairs), sum(a for _, a in pairs)
        today_acc = max(0.0, 100 * (1 - abs(fc_sofar - act_sofar) / act_sofar)) if act_sofar > 1 else None

        # last 7 days: forecast total vs actual total
        hist = []
        for k in range(7, 0, -1):
            day = today - pd.Timedelta(days=k)
            fc = self.forecasts[self.forecasts["date"].astype(str) == str(day.date())]
            act = normal.reindex(pd.date_range(day, periods=24, freq="h"))
            if len(fc) == 24 and act.notna().sum() >= 20:
                hist.append({"date": day.strftime("%d %b"), "forecast": round(fc["forecast_wh"].astype(float).sum(), 1),
                             "actual": round(act.sum(), 1)})

        week_acc = None
        if hist and sum(h["actual"] for h in hist) > 0:
            week_acc = 100 * (1 - sum(abs(h["forecast"] - h["actual"]) for h in hist) / sum(h["actual"] for h in hist))

        # energy and bill
        today_wh = d.loc[d["time"] >= today, "energy_wh"].sum()
        month_start = today.replace(day=1)
        month_wh = d.loc[d["time"] >= month_start, "energy_wh"].sum()
        days_in_month = month_start.days_in_month
        fc_total = sum(f for f in forecast if f is not None) if forecast[0] is not None else today_wh
        remaining_today = sum(f for f, ok in zip(forecast, done) if f is not None and not ok) if forecast[0] is not None else 0
        month_est_wh = month_wh + remaining_today + (days_in_month - now.day) * fc_total
        rate = config.RATE_PER_KWH

        peak_idx = int(np.nanargmax([f if f is not None else -1 for f in forecast])) if forecast[0] is not None else None
        return {
            "ready": True,
            "csv": self.reader.file,
            "last_time": now.strftime("%d %b %Y, %H:%M:%S"),
            "reading": {"voltage": float(last["voltage"]), "current": float(last["current"]),
                        "power": float(last["power"]), "pf": float(last["pf"]), "frequency": float(last["frequency"]),
                        "status": last["status"]},
            "appliance": None,                                    # filled in get_state (always fresh)
            "live": live,
            "today": {"hours": [f"{h:02d}" for h in range(24)], "forecast": forecast, "actual": actual,
                      "banned": banned, "current_hour": now.hour, "running_wh": round(float(running), 1),
                      "accuracy": today_acc, "forecast_sofar_wh": fc_sofar, "actual_sofar_wh": act_sofar,
                      "forecast_total_wh": fc_total if forecast[0] is not None else None,
                      "peak_hour": peak_idx},
            "history": hist,
            "week_accuracy": week_acc,
            "energy": {"today_wh": float(today_wh), "today_cost": today_wh / 1000 * rate,
                       "month_kwh": month_wh / 1000, "month_cost": month_wh / 1000 * rate,
                       "month_est_kwh": month_est_wh / 1000, "month_est_cost": month_est_wh / 1000 * rate,
                       "rate": rate},
            "limits": {"instant": config.TRIP_INSTANT_W, "sustained": config.TRIP_SUSTAINED_W,
                       "sustained_s": config.TRIP_SUSTAINED_SECONDS},
            "models": self._model_summary(),
            "rows": int(len(d)),
        }

    def _model_summary(self):
        m = self.metrics
        if not m:
            return None
        fa, an = m["forecast"]["accuracy"], m["anomaly"]
        return {"trained_at": m.get("trained_at"),
                "forecast_hostel_daily": fa["AI model"]["hostel_daily"],
                "forecast_hostel_hourly": fa["AI model"]["hostel_hourly"],
                "forecast_room_daily": fa["AI model"]["room_daily"],
                "baseline_hostel_daily": fa["same as yesterday"]["hostel_daily"],
                "baseline_room_daily": fa["average of last 7 days"]["room_daily"],
                "anomaly_precision": an["precision"], "anomaly_recall": an["recall"],
                "anomaly_type_acc": an["type_accuracy"],
                "real_events": f"{an['real_day']['events_detected']}/{an['real_day']['kettle_events']}",
                "real_false": an["real_day"]["false_alarm_readings"],
                "has_forecaster": self.forecaster is not None, "has_anomaly": self.anomaly is not None}

    # ------------------------------------------------------------------ background loop
    def run_forever(self, stop_event):
        while not stop_event.is_set():
            try:
                self.poll()
            except Exception as err:                       # never stop monitoring because of one bad poll
                print(f"{YELLOW}[monitor] problem while reading: {err}{END}")
            stop_event.wait(config.POLL_SECONDS)
