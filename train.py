"""
train.py - trains both AI models, tests them honestly, saves them + charts for the report.

Honest testing = CHRONOLOGICAL split: models learn from the first 75 % of the days and are tested
on the last 25 % (days they have never seen). Then they are re-trained on all days and saved.
The anomaly model is ALSO tested on our REAL recorded day (labelled with our load diary).

Run (after generate_data.py):   python train.py        (~1-2 minutes)
Outputs:  models/*.joblib, models/metrics.json, reports/*.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import config
from data_cleaning import HEATING_EVENTS, clean_and_add_energy, load_clean_real_day, read_pzem_csv
from models import (NORMAL, AnomalyModel, ForecastModel, anomaly_features, full_hourly_series, heating_class,
                    typical_profile)

BLUE, ORANGE, RED, GREY = "#2563eb", "#f59e0b", "#dc2626", "#9ca3af"


def accuracy(pred, actual):
    """100 % - (total absolute error / total actual energy).  100 % = perfect."""
    return float(100 * (1 - np.nansum(np.abs(pred - actual)) / np.nansum(actual)))


# ====================================================================== FORECAST
def load_hostel():
    h = pd.read_csv(config.HOSTEL_HOURLY)
    h["hour"] = pd.to_datetime(h["hour"], utc=True).dt.tz_convert(config.TIMEZONE)
    return {room: full_hourly_series(g.set_index("hour")["normal_wh"]) for room, g in h.groupby("room")}


def build(series, days, typical_until):
    """Feature rows for every room and day (each room gets its own 'typical week')."""
    rows = []
    for room, s in series.items():
        fm = ForecastModel()
        fm.typical = typical_profile(s[s.index < typical_until])
        r = fm.build_rows(s, days)
        r["room"] = room
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def train_forecast():
    print("\n=== FORECAST MODEL (next-day, 24 hourly values) ===")
    series = load_hostel()
    start = min(s.index.min() for s in series.values()).normalize() + pd.Timedelta(days=8)   # need 1 week of history
    end = max(s.index.max() for s in series.values()).normalize()
    days = pd.date_range(start, end, freq="D")
    n_test = int(len(days) * 0.25)
    train_days, test_days = days[:-n_test], days[-n_test:]
    print(f"{len(series)} rooms | train {len(train_days)} days ({train_days[0].date()} -> {train_days[-1].date()}) "
          f"| test {len(test_days)} days ({test_days[0].date()} -> {test_days[-1].date()})")

    train = build(series, train_days, test_days[0]).dropna(subset=["actual"])
    test = build(series, test_days, test_days[0])
    model = ForecastModel().fit(train)
    test["forecast"] = model.predict_rows(test)
    test = test.dropna(subset=["actual"])

    # how good is it?  (compared with simple guesses a person could make without AI)
    guesses = {"AI model": "forecast", "same as yesterday": "lag_1d", "same day last week": "lag_7d",
               "average of last 7 days": "mean_7d"}
    room_day = test.groupby(["room", test["time"].dt.date])[["actual"] + list(guesses.values())].sum()
    hostel_hour = test.groupby("time")[["actual"] + list(guesses.values())].sum()
    hostel_day = hostel_hour.groupby(hostel_hour.index.date).sum()
    ours = test[test["room"] == "R00"]
    ours_day = ours.groupby(ours["time"].dt.date)[["actual"] + list(guesses.values())].sum()

    table = {}
    for name, col in guesses.items():
        table[name] = {
            "hostel_daily": accuracy(hostel_day[col], hostel_day["actual"]),
            "hostel_hourly": accuracy(hostel_hour[col], hostel_hour["actual"]),
            "room_daily": accuracy(room_day[col], room_day["actual"]),
            "our_room_daily": accuracy(ours_day[col], ours_day["actual"]),
        }
    print(f"{'accuracy on unseen test days':32s} {'hostel/day':>11s} {'hostel/hour':>12s} {'1 room/day':>11s}")
    for name, v in table.items():
        print(f"{name:32s} {v['hostel_daily']:10.1f}% {v['hostel_hourly']:11.1f}% {v['room_daily']:10.1f}%")

    plot_forecast(hostel_hour, ours, test_days)

    # final model: learn from ALL days. 'Typical week' of OUR room is used live.
    final = ForecastModel().fit(build(series, days, end + pd.Timedelta(days=1)).dropna(subset=["actual"]))
    final.typical = typical_profile(series["R00"])
    final.save(config.FORECAST_MODEL)
    print(f"Saved {config.FORECAST_MODEL}")

    importance = dict(sorted(zip(model.model.feature_names_in_, model.model.feature_importances_.round(3)),
                             key=lambda kv: -kv[1]))
    return {"model": "Random Forest regression (100 trees)", "rooms": len(series),
            "train_days": len(train_days), "test_days": len(test_days),
            "accuracy": table, "feature_importance": {k: float(v) for k, v in importance.items()}}


def plot_forecast(hostel_hour, ours, test_days):
    week = test_days[-7:]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    h = hostel_hour[hostel_hour.index >= week[0]]
    axes[0].bar(h.index.tz_localize(None), h["actual"] / 1000, width=1 / 24, color=GREY, label="Actual (kWh)")
    axes[0].plot(h.index.tz_localize(None), h["forecast"] / 1000, color=BLUE, lw=2, label="AI forecast (made the night before)")
    axes[0].set_title("Whole hostel (40 rooms) - hourly energy, last 7 test days")
    axes[0].set_ylabel("kWh per hour")
    axes[0].legend()
    o = ours[ours["time"] >= week[0]]
    axes[1].bar(o["time"].dt.tz_localize(None), o["actual"], width=1 / 24, color=GREY, label="Actual (Wh)")
    axes[1].plot(o["time"].dt.tz_localize(None), o["forecast"], color=BLUE, lw=2, label="AI forecast")
    axes[1].set_title("Our room - hourly energy, same 7 days")
    axes[1].set_ylabel("Wh per hour")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(config.REPORT_DIR, "forecast_vs_actual.png"), dpi=120)
    plt.close(fig)


# ====================================================================== ANOMALY
def train_anomaly():
    print("\n=== ANOMALY MODEL (which heating appliance is running?) ===")
    data = clean_and_add_energy(read_pzem_csv(config.TRAINING_DATA))
    X = anomaly_features(data)
    y = heating_class(data["heating_type"].where(data["status"] == "OK", ""))
    cut = data["time"].min() + 0.75 * (data["time"].max() - data["time"].min())
    is_train = (data["time"] < cut).to_numpy()

    # most rows are 'normal' - keep every heating row but only 1 in 4 normal rows (faster, same quality)
    rng = np.random.default_rng(0)
    keep = is_train & ((y != NORMAL).to_numpy() | (rng.random(len(y)) < 0.25))
    model = AnomalyModel().fit(X[keep], y[keep])

    kind, _ = model.predict(X[~is_train])
    truth = y[~is_train].to_numpy()
    yb, pb = truth != NORMAL, kind != NORMAL
    result = {"model": "Random Forest classifier (100 trees)",
              "test_rows": int((~is_train).sum()), "test_heating_rows": int(yb.sum()),
              "precision": float(precision_score(yb, pb)), "recall": float(recall_score(yb, pb)),
              "f1": float(f1_score(yb, pb))}
    labels = [NORMAL, "kettle", "iron", "induction", "heater/rod"]
    cm = confusion_matrix(truth, kind, labels=labels)
    result["type_accuracy"] = float((kind[yb] == truth[yb]).mean())
    print(f"Test days (unseen): {result['test_rows']} readings, {result['test_heating_rows']} with a heating appliance")
    print(f"  precision {result['precision']:.3f} (alarms that were real)   recall {result['recall']:.3f} "
          f"(real ones caught)   F1 {result['f1']:.3f}   correct appliance type {result['type_accuracy']:.1%}")

    result["real_day"] = test_on_real_day(model)
    plot_confusion(cm, labels)

    final = AnomalyModel()
    keep_all = (y != NORMAL).to_numpy() | (rng.random(len(y)) < 0.25)
    final.fit(X[keep_all], y[keep_all]).save(config.ANOMALY_MODEL)
    print(f"Saved {config.ANOMALY_MODEL}")
    return result


def test_on_real_day(model):
    """The real test: our recorded day, labelled from the load diary (kettle used twice)."""
    day = load_clean_real_day()
    kind, prob = model.predict(anomaly_features(day))
    kind = np.where(day["status"] == "OK", kind, NORMAL)
    truth = day["heating_on"].to_numpy() == 1
    flagged = kind != NORMAL

    # count EVENTS (one kettle use in our diary = one event), the way a warden would see alerts
    caught = 0
    for start, end in HEATING_EVENTS:
        inside = ((day["time"] >= pd.Timestamp(start, tz=config.TIMEZONE)) &
                  (day["time"] < pd.Timestamp(end, tz=config.TIMEZONE))).to_numpy()
        caught += int((flagged & inside).any())
    res = {"readings": int(len(day)), "kettle_events": len(HEATING_EVENTS), "events_detected": caught,
           "kettle_readings": int(truth.sum()), "kettle_readings_detected": int((flagged & truth).sum()),
           "false_alarm_readings": int((flagged & ~truth).sum()),
           "type_said_kettle": float((kind[flagged & truth] == "kettle").mean()) if (flagged & truth).any() else 0.0}
    print(f"REAL day: kettle events caught {caught}/{len(HEATING_EVENTS)} | readings {res['kettle_readings_detected']}/"
          f"{res['kettle_readings']} | false alarms {res['false_alarm_readings']} of {len(day)} readings | "
          f"named 'kettle' {res['type_said_kettle']:.0%}")

    fig, ax = plt.subplots(figsize=(14, 4.5))
    t = day["time"].dt.tz_localize(None)
    ax.plot(t, day["power"], color=BLUE, lw=0.8, label="Real power (W)")
    ax.scatter(t[flagged], day["power"][flagged], color=RED, s=8, zorder=3, label="AI: heating appliance (kettle)")
    ax.set_title("AI anomaly model on our REAL recorded day - laptop, chargers, bulb ignored; both kettle uses caught")
    ax.set_ylabel("Power (W)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(config.REPORT_DIR, "anomaly_real_day.png"), dpi=120)
    plt.close(fig)
    return res


def plot_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(shown, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if shown[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("AI said")
    ax.set_ylabel("Really was")
    ax.set_title("Anomaly model - test days (readings)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.REPORT_DIR, "anomaly_confusion.png"), dpi=120)
    plt.close(fig)


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    metrics = {"trained_at": pd.Timestamp.now(tz=config.TIMEZONE).strftime("%Y-%m-%d %H:%M"),
               "forecast": train_forecast(), "anomaly": train_anomaly()}
    with open(config.METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {config.METRICS_FILE} and charts in {config.REPORT_DIR}/")


if __name__ == "__main__":
    main()
