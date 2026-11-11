"""
data_cleaning.py - turns the raw PZEM-004T log into clean, usable data.

The same functions are used in 3 places:
    1. here (run as a script) to clean the real day we recorded
    2. generate_data.py / train.py  (training data goes through the same cleaning)
    3. monitor.py (every new live row is cleaned the same way)

Problems found in our real log, and what we do about them:
    - time is in UTC                -> converted to Indian time (IST)
    - sensor fault / supply off     -> all values become 0 (voltage = 0). Row is marked SENSOR_FAULT
    - energy counter reads 0 during a fault, then jumps back -> we do NOT trust the counter;
                                       energy is calculated as power x time instead
    - impossible values (pf > 1 ...) -> row marked INVALID
    - missing rows (logger stopped)  -> no energy is added across a silence longer than MAX_GAP_SECONDS
    - duplicate / out-of-order rows -> removed / sorted

Run:   python data_cleaning.py
"""
import os

import numpy as np
import pandas as pd

import config

COLUMNS = ["voltage", "current", "power", "energy", "pf", "frequency"]


# ------------------------------------------------------------------ reading
def read_pzem_csv(path):
    """Read a CSV written by the hardware (columns: id, created_at, voltage, current, power, energy, pf, frequency)."""
    df = pd.read_csv(path)
    return prepare_columns(df)


def prepare_columns(df):
    """Make types correct: 'time' column in IST, numbers as floats."""
    df = df.copy()
    time = pd.to_datetime(df["created_at"], errors="coerce", format="ISO8601")
    if time.dt.tz is None:                       # no timezone written -> assume it is already Indian time
        time = time.dt.tz_localize(config.TIMEZONE)
    df["time"] = time.dt.tz_convert(config.TIMEZONE)
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time"])              # rows with an unreadable time are useless
    extra = [c for c in df.columns if c not in ["id", "created_at", "time"] + COLUMNS]   # e.g. labels
    return df[["time"] + COLUMNS + extra]


# ------------------------------------------------------------------ cleaning
def clean(df):
    """Adds a 'status' column: OK / SENSOR_FAULT / INVALID. Nothing is deleted except duplicates."""
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    # voltage 0 = sensor lost supply. Below 150 V = the moment it is switching off/on (not a real reading)
    sensor_fault = (df["voltage"] < 150) | df[["voltage", "power", "pf"]].isna().any(axis=1)
    invalid = ((df["voltage"] > 300) | (df["current"] < 0) | (df["power"] < 0)
               | (df["pf"] > 1) | (df["power"] > 10000))

    df["status"] = "OK"
    df.loc[invalid, "status"] = "INVALID"
    df.loc[sensor_fault, "status"] = "SENSOR_FAULT"
    return df


def add_energy(df, max_gap=config.MAX_GAP_SECONDS):
    """Energy used between two readings = power x time.
    (power of the previous reading is assumed to stay the same until the next reading)"""
    seconds = df["time"].diff().dt.total_seconds()
    prev_power = df["power"].shift()
    prev_ok = df["status"].shift() == "OK"
    usable = prev_ok & (seconds > 0) & (seconds <= max_gap)
    df["energy_wh"] = np.where(usable, prev_power * seconds / 3600.0, 0.0)   # W x s / 3600 = Wh
    return df


def clean_and_add_energy(df, max_gap=config.MAX_GAP_SECONDS):
    return add_energy(clean(df), max_gap)


# ------------------------------------------------------------------ hourly totals
def hourly_energy(df):
    """Energy (Wh) used in every clock hour, plus how much of the hour had good data."""
    hour = df["time"].dt.floor("h")
    ok = df["status"] == "OK"
    out = pd.DataFrame({
        "energy_wh": df["energy_wh"].groupby(hour).sum(),
        "good_readings": ok.groupby(hour).sum(),
    })
    out.index.name = "hour"
    return out


# ------------------------------------------------------------------ the real day (what we wrote in our notebook)
# The load diary we noted while recording (local time). Used to LABEL the real day so we can test the models.
LOAD_DIARY = [
    ("2026-11-04 09:50", "80W phone charger + 0.5W LED"),
    ("2026-11-04 10:07", "0.5W LED"),
    ("2026-11-04 10:17", "44W phone charger"),
    ("2026-11-04 10:48", "Idle 0 W"),
    ("2026-11-04 14:10", "9W LED + 65W laptop charger"),
    ("2026-11-04 15:15", "Sensor fault"),
    ("2026-11-04 15:28", "9W LED"),
    ("2026-11-04 19:08", "600W KETTLE + 9W LED"),
    ("2026-11-04 19:26", "80W phone charger + 9W LED"),
    ("2026-11-04 20:32", "9W LED"),
    ("2026-11-04 20:37", "Hotspot on + 9W LED"),
    ("2026-11-04 21:32", "9W LED + 80W phone charger"),
    ("2026-11-04 22:00", "9W LED + power bank"),
    ("2026-11-04 23:56", "KETTLE + phone charger + LED"),
    ("2026-11-05 00:16", "Phone charger + LED"),
    ("2026-11-05 00:50", "0.5W LED"),
    ("2026-11-05 08:45", "9W LED"),
    ("2026-11-05 09:22", "Bulb 100 W"),
    ("2026-11-05 09:31", "Fault (zero readings)"),
    ("2026-11-05 09:50", "Off"),
]
# kettle = high-power heating appliance (from the diary above)
HEATING_EVENTS = [("2026-11-04 19:08", "2026-11-04 19:26"), ("2026-11-04 23:56", "2026-11-05 00:16")]
REAL_DAY_START, REAL_DAY_END = "2026-11-04 09:40", "2026-11-05 09:51"


def load_clean_real_day():
    """Read the file saved by main() back (times are saved as IST)."""
    df = pd.read_csv(config.CLEAN_REAL_DAY)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(config.TIMEZONE)
    return df


def label_real_day(df):
    """heating_on = 1 while the kettle (from the diary) is actually drawing power.
    The diary was written to the minute, so 1 minute is allowed on each side."""
    df["heating_on"] = 0
    for start, end in HEATING_EVENTS:
        inside = (df["time"] >= pd.Timestamp(start, tz=config.TIMEZONE) - pd.Timedelta(minutes=1)) & \
                 (df["time"] < pd.Timestamp(end, tz=config.TIMEZONE) + pd.Timedelta(minutes=1))
        df.loc[inside & (df["power"] > 300), "heating_on"] = 1
    return df


def plot_real_day(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    t = df["time"].dt.tz_localize(None)
    ax.plot(t, df["power"], lw=0.8, color="#2563eb", label="Active power (W)")
    fault = df["status"] == "SENSOR_FAULT"
    ax.scatter(t[fault], df["power"][fault], s=6, color="#dc2626", label="Sensor fault rows", zorder=3)
    for when, text in LOAD_DIARY:
        x = pd.Timestamp(when)
        ax.axvline(x, color="#9ca3af", lw=0.5, ls="--")
        ax.text(x, ax.get_ylim()[1] * 0.98, text, rotation=90, fontsize=7, va="top", ha="right", color="#374151")
    ax.set_title("Real recorded day (4-5 Nov 2026) - power with our load diary")
    ax.set_ylabel("Power (W)")
    ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    raw = read_pzem_csv(config.RAW_REAL_LOG)
    df = clean_and_add_energy(raw)
    print(f"Rows read: {len(raw)}   after removing duplicates: {len(df)}")
    print(f"Time range (IST): {df['time'].min()}  ->  {df['time'].max()}")
    print("Row status:\n" + df["status"].value_counts().to_string())

    gaps = df["time"].diff().dt.total_seconds()
    print(f"Typical time between readings: {gaps.median():.1f} s")
    print(f"Silences longer than {config.MAX_GAP_SECONDS} s: {(gaps > config.MAX_GAP_SECONDS).sum()}")

    # keep only the continuous recorded day
    day = df[(df["time"] >= pd.Timestamp(REAL_DAY_START, tz=config.TIMEZONE)) &
             (df["time"] < pd.Timestamp(REAL_DAY_END, tz=config.TIMEZONE))].reset_index(drop=True)
    day = label_real_day(day)
    counter = day.loc[day["status"] == "OK", "energy"]
    print(f"\nReal day: {len(day)} rows")
    print(f"Energy by power x time : {day['energy_wh'].sum():.0f} Wh")
    print(f"Energy by meter counter: {(counter.max() - counter.min()) * 1000:.0f} Wh   (should be close)")
    print(f"Max power: {day['power'].max():.0f} W   Voltage range: {day.loc[day.status == 'OK', 'voltage'].min():.0f}"
          f"-{day['voltage'].max():.0f} V")

    os.makedirs("data", exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    out = day.assign(time=day["time"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    out.to_csv(config.CLEAN_REAL_DAY, index=False)
    plot_real_day(day, os.path.join(config.REPORT_DIR, "real_day.png"))
    print(f"\nSaved {config.CLEAN_REAL_DAY} and {config.REPORT_DIR}/real_day.png")


if __name__ == "__main__":
    main()
