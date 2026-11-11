"""
models.py - the two AI models and the features they use.

1) ForecastModel   (Random Forest regression)
   At midnight it predicts the energy (Wh) the room will use in EACH of the next 24 hours.
   It only looks at the past: same hour yesterday, same hour last week, average of last 7 days,
   yesterday's total, the weekday, the hour ... (no future data = honest forecast).

2) AnomalyModel    (Random Forest classification)
   For every new reading it decides: is a high-power heating appliance running?
   And which one: kettle / iron / induction / heater (room heater or immersion rod).
   It uses the electrical "signature": real power, reactive power, power factor, how fast power changed,
   power corrected to 230 V (so a voltage change is not mistaken for a new appliance), and how often the
   load switched ON/OFF in the last 5 minutes (an iron's thermostat cycles, a heater stays on).

Safety TRIP decisions are NOT made by the ML model - they are simple fixed rules (see monitor.py),
because a safety rule must be 100 % predictable. The ML model gives early warnings.
"""
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import config


def is_exam_day(day):
    """True if the date is inside an exam period from config.py (college calendar)."""
    day = pd.Timestamp(day).tz_localize(None).normalize() if pd.Timestamp(day).tzinfo else pd.Timestamp(day).normalize()
    return any(pd.Timestamp(a) <= day <= pd.Timestamp(b) for a, b in config.EXAM_PERIODS)

# ======================================================================= FORECAST
FORECAST_FEATURES = ["hour", "day_of_week", "is_weekend", "is_exam", "typical", "lag_1d", "lag_7d", "mean_7d",
                     "same_weekday_mean", "yesterday_total", "yesterday_evening", "week_avg_total"]


def full_hourly_series(hourly_wh):
    """Make sure every hour exists (missing hours become NaN)."""
    if hourly_wh.empty:
        return hourly_wh
    idx = pd.date_range(hourly_wh.index.min().normalize(), hourly_wh.index.max(), freq="h")
    return hourly_wh.reindex(idx)


def forecast_features(hourly_wh, day, typical):
    """24 rows of features for the 24 hours of `day`, using ONLY data from before `day` 00:00.
    hourly_wh : Series of Wh per hour (index = hour start, IST)
    typical   : 7 x 24 table of typical Wh (row = weekday Mon..Sun, column = hour), learned in training"""
    day = pd.Timestamp(day).normalize()
    hours = pd.date_range(day, periods=24, freq="h")
    past = hourly_wh[hourly_wh.index < day]

    def value_at(times):
        return past.reindex(times).to_numpy(dtype=float)

    dow = day.dayofweek
    typ = np.asarray(typical[dow], dtype=float)
    lags = np.stack([value_at(hours - pd.Timedelta(days=k)) for k in range(1, 8)])     # 7 x 24
    weekly = np.stack([value_at(hours - pd.Timedelta(days=k)) for k in (7, 14, 21)])
    yesterday = value_at(hours - pd.Timedelta(days=1))
    last_week = lags

    def fill(arr):                         # missing past data -> use the typical value
        return np.where(np.isnan(arr), typ, arr)

    with warnings.catch_warnings():                 # all-missing columns simply give NaN (filled below)
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_7d = np.nanmean(last_week, axis=0)
        same_weekday = np.nanmean(weekly, axis=0)
    yesterday_f = fill(yesterday)
    daily_totals = [np.nansum(fill(row)) for row in last_week]

    return pd.DataFrame({
        "time": hours,
        "hour": np.arange(24),
        "day_of_week": dow,
        "is_weekend": int(dow >= 5),
        "is_exam": int(is_exam_day(day)),
        "typical": typ,
        "lag_1d": yesterday_f,
        "lag_7d": fill(lags[6]),
        "mean_7d": fill(mean_7d),
        "same_weekday_mean": fill(same_weekday),
        "yesterday_total": yesterday_f.sum(),
        "yesterday_evening": yesterday_f[18:].sum(),
        "week_avg_total": np.mean(daily_totals),
    })


def typical_profile(hourly_wh):
    """Average Wh for every weekday and hour - the 'normal week' of this room (7 x 24 numbers)."""
    s = hourly_wh.dropna()
    table = s.groupby([s.index.dayofweek, s.index.hour]).mean().unstack().reindex(index=range(7), columns=range(24))
    table = table.fillna(table.mean(axis=0)).fillna(0)          # missing weekday/hour -> average of that hour
    return table.to_numpy()


class ForecastModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, min_samples_leaf=40, max_features=0.5,
                                           random_state=42, n_jobs=-1)
        self.typical = None

    def build_rows(self, hourly_wh, days):
        rows = []
        for day in days:
            f = forecast_features(hourly_wh, day, self.typical)
            f["actual"] = hourly_wh.reindex(f["time"]).to_numpy()
            rows.append(f)
        return pd.concat(rows, ignore_index=True)

    def fit(self, rows):
        rows = rows.dropna(subset=["actual"])
        self.model.fit(rows[FORECAST_FEATURES], rows["actual"])
        return self

    def predict_rows(self, rows):
        return np.clip(self.model.predict(rows[FORECAST_FEATURES]), 0, None)

    def predict_day(self, hourly_wh, day):
        """The 24-hour forecast for `day` (DataFrame: time, forecast_wh)."""
        f = forecast_features(hourly_wh, day, self.typical)
        return pd.DataFrame({"time": f["time"], "forecast_wh": self.predict_rows(f)})

    def save(self, path):
        joblib.dump({"model": self.model, "typical": self.typical}, path, compress=3)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        m = cls()
        m.model, m.typical = d["model"], d["typical"]
        return m


# ======================================================================= ANOMALY
ANOMALY_FEATURES = ["power", "power_230v", "reactive_var", "pf", "current", "step",
                    "mean_1min", "max_2min", "min_2min", "std_2min", "reactive_mean_1min",
                    "switches_5min", "on_share_5min"]
NORMAL = "none"


def anomaly_features(df):
    """Features for every reading. Only PAST readings are used (works live, row by row).
    df needs: time, voltage, current, power, pf, status."""
    d = df[["time", "voltage", "current", "power", "pf"]].copy()
    ok = df["status"] == "OK"
    volts = d["voltage"].where(ok & (d["voltage"] > 0))
    d["power_230v"] = d["power"] * (230.0 / volts) ** 2              # what the power would be at 230 V
    apparent = (d["voltage"] * d["current"]).clip(lower=0)
    d["reactive_var"] = np.sqrt(np.clip(apparent ** 2 - d["power"] ** 2, 0, None))   # chargers/laptops have it, heaters do not
    d["step"] = d["power"].diff().fillna(0)

    roll = d.set_index("time")
    p = roll["power_230v"]
    d["mean_1min"] = p.rolling("60s").mean().to_numpy()
    d["max_2min"] = p.rolling("120s").max().to_numpy()
    d["min_2min"] = p.rolling("120s").min().to_numpy()
    d["std_2min"] = p.rolling("120s").std().fillna(0).to_numpy()
    d["reactive_mean_1min"] = roll["reactive_var"].rolling("60s").mean().to_numpy()
    # thermostat behaviour: an iron switches ON/OFF every minute, a heater or kettle stays ON
    switched = (roll["power_230v"].diff().abs() > 300).astype(float)
    d["switches_5min"] = switched.rolling("300s").sum().to_numpy()
    d["on_share_5min"] = (p > 400).astype(float).rolling("300s").mean().to_numpy()
    return d[ANOMALY_FEATURES].fillna(0)


def heating_class(heating_type):
    """Room heater and immersion rod look the same electrically (1-2 kW, pf 1) -> one class."""
    t = heating_type.fillna("").replace({"heater": "heater/rod", "immersion_rod": "heater/rod", "": NORMAL})
    return t


class AnomalyModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, class_weight="balanced_subsample",
                                            random_state=42, n_jobs=-1)

    def fit(self, features, labels):
        self.model.fit(features, labels)
        return self

    def predict(self, features):
        """Returns (class name, probability that it is a heating appliance) for every row."""
        proba = self.model.predict_proba(features)
        classes = self.model.classes_
        normal_col = list(classes).index(NORMAL)
        heating_prob = 1 - proba[:, normal_col]
        proba_heating = proba.copy()
        proba_heating[:, normal_col] = -1                      # best heating appliance type
        kind = classes[proba_heating.argmax(axis=1)]
        return np.where(heating_prob >= 0.5, kind, NORMAL), heating_prob

    def save(self, path):
        joblib.dump(self.model, path, compress=3)

    @classmethod
    def load(cls, path):
        m = cls()
        m.model = joblib.load(path)
        return m
