"""
generate_data.py - makes 180 days of realistic training data for ONE hostel room.

Why?  We recorded only ONE real day - far too little to train a model.
How?  Every appliance below was MEASURED from our real day (power, power factor, how it behaves):
          LED 9 W, phone charger, fast charger, laptop charger, power bank, 100 W bulb, 600 W kettle
      Appliances we could NOT safely test in the hostel (iron, induction, room heater, immersion rod)
      use their rated power.
      A student timetable decides WHEN things are switched on (classes, sleep, exam weeks, weekends),
      so the data has real daily and weekly patterns, plus random day-to-day differences.
      Voltage follows the hour-by-hour pattern we measured (high in the afternoon, low in the evening).
      Sensor faults (all zeros) and missing rows are added like in the real log.

Outputs:
    data/training_data.csv.gz  our room, one row every 20 s - SAME columns as the hardware CSV, plus 2 labels:
                               heating_on (1 = heating appliance in use) and heating_type (kettle / iron ...)
    data/hostel_hourly.csv.gz  hourly energy of 40 rooms with different habits (for the forecast model)

Run:   python generate_data.py              (180 days, ends the day before our real recording, ~2 min)
"""
import argparse

import numpy as np
import pandas as pd

import config
from models import is_exam_day

# ------------------------------------------------------------------ appliance library
# watts      : power at 230 V (resistive loads change with voltage, chargers do not)
# pf         : power factor measured with our PZEM
# resistive  : True -> power follows (V / 230)^2  (kettle, bulb, heater ...)
# heating    : True -> high-power heating appliance (what the anomaly model must detect)
APPLIANCES = {
    "standby":       dict(watts=0.6,  pf=1.00, resistive=False, heating=False),  # hotspot / idle chargers (night)
    "night_lamp":    dict(watts=0.45, pf=1.00, resistive=False, heating=False),  # 0.5 W LED
    "led":           dict(watts=8.0,  pf=1.00, resistive=True,  heating=False),  # 9 W LED (measured 6.5-8.9 W)
    "phone_charger": dict(watts=15.0, pf=0.50, resistive=False, heating=False),  # measured 12-16 W
    "fast_charger":  dict(watts=60.0, pf=0.55, resistive=False, heating=False),  # measured 75 W falling to 38 W
    "laptop":        dict(watts=70.0, pf=0.55, resistive=False, heating=False),  # measured 75 W falling to 25 W
    "power_bank":    dict(watts=7.5,  pf=0.80, resistive=False, heating=False),
    "bulb":          dict(watts=100., pf=1.00, resistive=True,  heating=False),  # measured 108-134 W at 250 V
    "kettle":        dict(watts=600., pf=1.00, resistive=True,  heating=True),   # measured 491-598 W
    "iron":          dict(watts=1000, pf=1.00, resistive=True,  heating=True),   # rated value
    "induction":     dict(watts=1600, pf=0.96, resistive=False, heating=True),   # rated 1200-2000 W
    "heater":        dict(watts=1500, pf=1.00, resistive=True,  heating=True),   # rated 1000-2000 W
    "immersion_rod": dict(watts=1000, pf=1.00, resistive=True,  heating=True),   # rated value
}

# measured median voltage for every hour of the day (from our real day)
VOLTAGE_BY_HOUR = [232, 235, 224, 229, 229, 229, 224, 224, 239, 253, 258, 240,
                   241, 247, 261, 246, 227, 223, 214, 214, 217, 215, 218, 228]

# class timetable (hours away from the room) for Mon..Sun
TIMETABLE = {0: [(9, 13), (14, 17)], 1: [(9, 12)], 2: [(9, 13), (14, 16)], 3: [(10, 13)],
             4: [(9, 13), (14, 17)], 5: [(10, 12)], 6: []}


# ------------------------------------------------------------------ the student living in the room
# Our recorded room. Other rooms of the hostel get random habits (see random_room) for the hostel forecast.
OUR_ROOM = dict(shift=0.0, wake=7.6, sleep=24.6, study=0.85, laptop_w=75, led_w=8.0,
                kettle_tea=0.7, kettle_night=0.25, sports_days=(1, 3, 5), banned=1.0)

BANNED = ("iron", "induction", "heater", "immersion_rod")      # appliances not allowed in hostel rooms


def random_room(rng):
    """Habits of a random student (used to build a whole hostel)."""
    return dict(shift=float(rng.choice([-1, 0, 0, 1])), wake=rng.uniform(7.0, 8.5), sleep=rng.uniform(23.5, 26.0),
                study=rng.uniform(0.5, 0.95), laptop_w=rng.uniform(45, 95), led_w=float(rng.choice([8, 8, 15, 20])),
                kettle_tea=rng.uniform(0.1, 0.8), kettle_night=rng.uniform(0.0, 0.4),
                sports_days=tuple(rng.choice(7, 3, replace=False)), banned=rng.uniform(0.3, 2.0))


# ------------------------------------------------------------------ one day of the student's life
def plan_day(date, rng, busy=0.0, room=OUR_ROOM):
    """List of (appliance, start, end, extra) for one day. Times are pandas Timestamps (IST).
    busy = how study-heavy the student is these days (-1 relaxed ... +1 very busy). It changes slowly
    from day to day, like real habits do."""
    day = pd.Timestamp(date, tz=config.TIMEZONE).normalize()
    dow = day.dayofweek
    weekend = dow >= 5
    exam_week = is_exam_day(day)
    winter = max(0.0, (day.dayofyear - 290) / 60)                  # grows from late October
    at = lambda hours: day + pd.Timedelta(hours=float(hours))
    jitter = lambda sd: rng.normal(0, sd)
    plan = []

    def add(name, t_on, t_off, **extra):
        if t_off > t_on:
            plan.append((name, t_on, t_off, extra))

    wake = at((9.5 if weekend else room["wake"]) + jitter(0.5))
    late = 1.5 if (dow in (4, 5) or exam_week) else 0.0           # Fri/Sat night and exam weeks: sleep later
    sleep = at(room["sleep"] + late + 0.5 * busy + jitter(0.5))

    # away for classes (sometimes the class is skipped)
    sh = room["shift"]
    away = [(at(a + sh + jitter(0.15)), at(b + sh + jitter(0.2))) for a, b in TIMETABLE[dow] if rng.random() > 0.07]
    back_home = max([b for _, b in away], default=at(13 + jitter(1)))
    if rng.random() < (0.8 if dow in room["sports_days"] else 0.15):  # sports on fixed evenings (Tue/Thu/Sat)
        away.append((at(17.5 + jitter(0.3)), at(19.0 + jitter(0.3))))
    dinner = (at(20.2 + jitter(0.2)), at(20.9 + jitter(0.2)))

    def present(t):
        return not any(a <= t < b for a, b in away + [dinner])

    # lights: morning (if up early), and evening until sleep
    if wake < at(8.3):
        add("led", wake, at(8.6 + jitter(0.2)), watts=room["led_w"])
    evening = max(back_home, at(16.5 + jitter(0.4)))
    add("led", evening, sleep, watts=room["led_w"])
    if not away or rng.random() < 0.3:                              # lights during a free day
        add("led", at(11 + jitter(1)), at(13.5 + jitter(1)), watts=room["led_w"])
    if rng.random() < 0.5:
        add("night_lamp", sleep, at(24 + 7.5 + jitter(0.5)))
    add("standby", at(21 + jitter(0.5)), at(24 + 8.5 + jitter(0.5)))
    if rng.random() < 0.15:
        t = at(rng.choice([8.3, 19.5, 22.0]) + jitter(0.3))
        add("bulb", t, t + pd.Timedelta(minutes=float(rng.uniform(10, 50))))

    # phones: one charge at night, one during the day (sometimes a fast charger)
    t = sleep - pd.Timedelta(hours=1.5 + jitter(0.3))
    add("phone_charger", t, t + pd.Timedelta(hours=1.6 + jitter(0.3)), start=18, end=11)
    t = at(rng.choice([10.0, 13.5, 16.5, 19.3]) + jitter(0.4))
    if present(t):
        kind = "fast_charger" if rng.random() < 0.35 else "phone_charger"
        add(kind, t, t + pd.Timedelta(hours=1.0 + jitter(0.2)), start=75 if kind == "fast_charger" else 17,
            end=35 if kind == "fast_charger" else 12)
    if rng.random() < 0.3:
        t = at(22 + jitter(0.5))
        add("power_bank", t, t + pd.Timedelta(hours=1.5))

    # laptop study sessions
    if weekend and rng.random() < 0.7:
        t = at(14.5 + jitter(0.8))
        add("laptop", t, t + pd.Timedelta(hours=2.5 + jitter(0.5)), start=room["laptop_w"], end=30)
    if exam_week or rng.random() < room["study"] + 0.15 * busy:
        t = at(21.0 - 0.5 * busy + jitter(0.4))
        hours = (sleep - t).total_seconds() / 3600 if exam_week else 2.2 + 0.8 * busy + jitter(0.4)
        add("laptop", t, t + pd.Timedelta(hours=hours), start=room["laptop_w"], end=25)
    if not weekend and len(away) < len(TIMETABLE[dow]) and rng.random() < 0.6:   # skipped a class
        t = at(10.5 + jitter(0.5))
        add("laptop", t, t + pd.Timedelta(hours=1.5), start=70, end=40)

    # HIGH-POWER HEATING APPLIANCES (labelled anomalies)
    # kettle: evening tea is a habit; late-night maggi happens more when busy / in exams
    kettle_times = [(19.3, room["kettle_tea"]),
                    (23.7, room["kettle_night"] + 0.25 * busy + (0.3 if exam_week else 0)), (8.0, 0.1)]
    for hour, chance in kettle_times:
        t = at(hour + jitter(0.25))
        if rng.random() < chance and present(t) and t < sleep:
            add("kettle", t, t + pd.Timedelta(minutes=float(rng.uniform(6, 16))))
    k = room["banned"]                                              # some students break the rules more often
    if rng.random() < 0.05 * k:
        t = at(19.6 + jitter(0.6))
        add("induction", t, t + pd.Timedelta(minutes=float(rng.uniform(12, 35))))
    if not weekend and rng.random() < 0.08 * k:
        t = wake + pd.Timedelta(minutes=10)
        add("iron", t, t + pd.Timedelta(minutes=float(rng.uniform(6, 15))))
    if rng.random() < (0.02 + 0.06 * winter) * k:
        t = at(22.5 + jitter(0.8))
        add("heater", t, t + pd.Timedelta(minutes=float(rng.uniform(25, 90))),
            watts=float(rng.choice([1000, 2000])))
    if rng.random() < 0.04 * k:
        t = wake + pd.Timedelta(minutes=5)
        add("immersion_rod", t, t + pd.Timedelta(minutes=float(rng.uniform(8, 18))))

    # nothing is switched on while the student is away: such a start waits until they are back
    result = []
    for name, t_on, t_off, extra in plan:
        if name not in ("standby", "night_lamp", "heater"):
            for a, b in away + [dinner]:
                if a <= t_on < b:
                    t_off, t_on = t_off + (b - t_on), b
        result.append((name, t_on, t_off, extra))
    return result


# ------------------------------------------------------------------ appliance power over time
def appliance_power(name, elapsed, duration, voltage, rng, extra):
    """Power (W) of ONE appliance at the given seconds since it was switched on."""
    spec = APPLIANCES[name]
    watts = extra.get("watts", spec["watts"])
    if "start" in extra:                                   # chargers / laptop: power falls as battery fills
        frac = np.clip(elapsed / max(duration, 1), 0, 1)
        watts = extra["end"] + (extra["start"] - extra["end"]) * np.exp(-3 * frac)
    p = np.full(len(elapsed), float(watts)) if np.isscalar(watts) else watts.astype(float)
    if name == "iron":                                     # thermostat: on ~50 s, off ~60 s
        p = p * ((elapsed % 110) < 50)
    if name == "induction":                                # power level changed every few minutes
        level = np.array([1200, 1600, 2000, 1600])[(elapsed // 240).astype(int) % 4]
        p = level.astype(float)
    if spec["resistive"]:
        p = p * (voltage / 230.0) ** 2
    return p * rng.normal(1, 0.015, len(p))


def make_readings(intervals, times, rng):
    """Build the sensor readings (like the PZEM would report) for the given time points."""
    n = len(times)
    secs = (times - times[0]).total_seconds().to_numpy()
    hour = (times.hour + times.minute / 60).to_numpy()

    # voltage: measured hourly pattern + slow day-to-day drift + small noise
    base = np.interp(hour, np.arange(25), VOLTAGE_BY_HOUR + [VOLTAGE_BY_HOUR[0]])
    drift = np.zeros(n)
    if n > 10000:                                          # long runs: slow wandering of the supply voltage
        drift = np.cumsum(rng.normal(0, 0.05, n))
        drift -= np.convolve(drift, np.ones(4321) / 4321, mode="same")  # keep it around 0
    voltage = base + drift + rng.normal(0, 0.3, n)

    power = np.zeros(n)
    apparent = np.zeros(n)                                 # volt-amperes, to get current and pf
    heating_type = np.full(n, "", dtype=object)
    for name, start, end, extra in intervals:
        i0, i1 = np.searchsorted(times, [start, end])
        if i1 <= i0:
            continue
        elapsed = secs[i0:i1] - (start - times[0]).total_seconds()
        p = appliance_power(name, elapsed, (end - start).total_seconds(), voltage[i0:i1], rng, extra)
        pf = APPLIANCES[name]["pf"]
        power[i0:i1] += p
        apparent[i0:i1] += p / pf
        if APPLIANCES[name]["heating"]:
            in_use = (p > 50) | (name == "iron")           # an iron is "in use" even when its thermostat is off
            heating_type[i0:i1] = np.where(in_use, name, heating_type[i0:i1])

    voltage -= power / 1000 * 3.0                          # big loads pull the voltage down a little
    current = apparent / voltage
    pf = np.where(power > 0.3, np.clip(power / np.maximum(apparent, 1e-9), 0, 1), 0.0)
    pf = np.where((power > 0.3) & (power < 1.0), 1.0, pf)
    energy = np.cumsum(power * np.diff(secs, prepend=secs[0])) / 3.6e6       # kWh counter
    df = pd.DataFrame({"time": times, "voltage": voltage.round(1), "current": current.round(3),
                       "power": power.round(1), "energy": energy.round(3), "pf": pf.round(2),
                       "frequency": (50 + rng.normal(0, 0.08, n)).round(1),
                       "heating_on": (heating_type != "").astype(int), "heating_type": heating_type})
    df.loc[df["power"] < 0.4, "power"] = 0.0               # PZEM shows 0 below ~0.4 W
    return df


def add_sensor_problems(df, rng, days):
    """Sensor faults (all values 0, like our real log) and short silences (rows missing)."""
    n = len(df)
    per_day = n // days
    for _ in range(int(days * 0.15)):                       # ~1 fault every 7 days, 1-15 minutes
        s = int(rng.integers(0, n - 50))
        length = int(rng.integers(3, 45))
        df.loc[s:s + length, ["voltage", "current", "power", "energy", "pf"]] = 0.0
        df.loc[s:s + length, ["heating_on", "heating_type"]] = [0, ""]
    keep = np.ones(n, bool)
    for _ in range(int(days * 0.25)):                       # logger silent for 1-10 minutes
        s = int(rng.integers(0, n - per_day // 24))
        keep[s:s + int(rng.integers(3, 30))] = False
    return df[keep].reset_index(drop=True)


def generate(start_date, days, seed=42, sample_seconds=config.SAMPLE_SECONDS, room=OUR_ROOM):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date, tz=config.TIMEZONE)
    end = start + pd.Timedelta(days=days)
    steps = rng.normal(sample_seconds, 1.5, int(days * 86400 / sample_seconds) + 10).clip(sample_seconds - 4)
    times = start + pd.to_timedelta(np.cumsum(steps), unit="s")
    times = pd.DatetimeIndex(times[times < end])

    intervals, busy = [], 0.0
    for d in pd.date_range(start - pd.Timedelta(days=1), end, freq="D"):    # day before: its night spills over
        busy = float(np.clip(0.9 * busy + rng.normal(0, 0.3), -1, 1))      # habits change slowly
        intervals += plan_day(d.date(), rng, busy, room)
    df = make_readings(intervals, times, rng)
    return add_sensor_problems(df, rng, days)


def to_hardware_format(df):
    """Same columns as the hardware CSV (created_at in UTC)."""
    out = df.copy()
    out.insert(0, "created_at", out.pop("time").dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M:%S.%f+00"))
    out.insert(0, "id", np.arange(1, len(out) + 1))
    return out


def hostel_hourly(start_date, days, n_rooms, seed=1000):
    """Hourly energy of MANY rooms with different habits (one reading per minute is enough here).
    normal_wh = everything except banned appliances, banned_wh = energy of banned appliances."""
    from data_cleaning import clean_and_add_energy, hourly_energy
    tables = []
    for r in range(n_rooms):
        rng = np.random.default_rng(seed + r)
        room = OUR_ROOM if r == 0 else random_room(rng)
        df = clean_and_add_energy(generate(start_date, days, seed + r, sample_seconds=60, room=room), max_gap=150)
        banned = df["heating_type"].isin(BANNED)
        total = hourly_energy(df)["energy_wh"]
        normal = hourly_energy(df.assign(energy_wh=df["energy_wh"].where(~banned, 0)))["energy_wh"]
        tables.append(pd.DataFrame({"hour": total.index, "room": f"R{r:02d}", "normal_wh": normal.round(2),
                                    "banned_wh": (total - normal).round(2)}))
        print(f"\r  hostel room {r + 1}/{n_rooms}", end="", flush=True)
    print()
    return pd.concat(tables, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--end", default="2026-11-04", help="first day NOT generated (default: our real day)")
    ap.add_argument("--rooms", type=int, default=40, help="rooms in the simulated hostel")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    start = pd.Timestamp(a.end) - pd.Timedelta(days=a.days)

    # 1) our room, every 20 s, with labels -> anomaly model + forecast model
    df = generate(start.date(), a.days, a.seed)
    to_hardware_format(df).to_csv(config.TRAINING_DATA, index=False, compression="gzip")
    daily = df.set_index("time")["power"].resample("D").mean() * 24      # Wh per day (approx.)
    print(f"Our room: {len(df)} rows for {a.days} days ({start.date()} -> {pd.Timestamp(a.end).date()})")
    print(f"  energy per day: min {daily.min():.0f} Wh, average {daily.mean():.0f} Wh, max {daily.max():.0f} Wh"
          f"   (our real day: 519 Wh)")
    print(f"  max power {df['power'].max():.0f} W | heating appliance in use: "
          + ", ".join(f"{k} {v}" for k, v in df.loc[df.heating_on == 1, "heating_type"].value_counts().items())
          + " (rows)")
    print(f"Saved {config.TRAINING_DATA}")

    # 2) a whole hostel of rooms with different habits -> forecast model learns from many students
    print(f"Hostel: {a.rooms} rooms ...")
    hostel_hourly(start.date(), a.days, a.rooms).to_csv(config.HOSTEL_HOURLY, index=False, compression="gzip")
    print(f"Saved {config.HOSTEL_HOURLY}")


if __name__ == "__main__":
    main()
