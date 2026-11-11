"""
simulate_live.py - a FAKE hardware logger, for testing and demos WITHOUT the ESP32.

It writes rows to the live CSV in exactly the hardware format (id, created_at, voltage, current, power,
energy, pf, frequency), one row every 20 s, using the same appliance models as generate_data.py.

    python simulate_live.py                  start writing live rows now
    python simulate_live.py --history 3      first write 3 days of past data (so forecasts can be shown)

While it runs, type a command + Enter to switch something on:
    kettle | iron | induction | heater | bulb | laptop | fault | quit
(heater / induction draw > 1000 W -> the monitor will TRIP after 30 s)
"""
import argparse
import os
import threading
import time

import numpy as np
import pandas as pd

import config
from generate_data import generate, make_readings, plan_day, to_hardware_format

DEMO_LOADS = {"kettle": 6, "iron": 6, "induction": 4, "heater": 3, "bulb": 5, "laptop": 30, "fault": 1}  # minutes
HEADER = "id,created_at,voltage,current,power,energy,pf,frequency\n"


def write_history(days):
    end = pd.Timestamp.now(tz=config.TIMEZONE).floor("s")
    start = (end - pd.Timedelta(days=days)).normalize()
    df = generate(start.date(), (end.normalize() - start).days + 1, seed=7)
    df = df[df["time"] < end]
    out = to_hardware_format(df).drop(columns=["heating_on", "heating_type"])
    out.to_csv(config.LIVE_CSV, index=False)
    print(f"Wrote {len(out)} rows of history ({start.date()} -> now) to {config.LIVE_CSV}")
    return int(out["id"].iloc[-1]), float(out["energy"].iloc[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", type=int, default=0, help="days of past data to write first")
    ap.add_argument("--every", type=float, default=config.SAMPLE_SECONDS, help="seconds between rows")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(config.LIVE_CSV) or ".", exist_ok=True)

    if a.history:
        row_id, energy = write_history(a.history)
    elif os.path.exists(config.LIVE_CSV) and os.path.getsize(config.LIVE_CSV) > len(HEADER):
        last = pd.read_csv(config.LIVE_CSV).iloc[-1]          # continue the existing file
        row_id, energy = int(last["id"]), float(last["energy"])
    else:
        with open(config.LIVE_CSV, "w") as f:
            f.write(HEADER)
        row_id, energy = 0, 0.0

    extra, fault_until = [], None                          # appliances switched on by typed commands
    lock = threading.Lock()

    def keyboard():
        nonlocal fault_until
        while True:
            cmd = input().strip().lower()
            if cmd == "quit":
                os._exit(0)
            if cmd in DEMO_LOADS:
                start = pd.Timestamp.now(tz=config.TIMEZONE)
                end = start + pd.Timedelta(minutes=DEMO_LOADS[cmd])
                with lock:
                    if cmd == "fault":
                        fault_until = end
                    else:
                        extra.append((cmd, start, end, {"start": 75, "end": 30} if cmd == "laptop" else {}))
                print(f"  -> {cmd} ON for {DEMO_LOADS[cmd]} min")
            elif cmd:
                print("  commands: " + " | ".join(DEMO_LOADS) + " | quit")

    threading.Thread(target=keyboard, daemon=True).start()
    print(f"Writing a row every {a.every:.0f} s to {config.LIVE_CSV}.  Type: "
          + " | ".join(DEMO_LOADS) + " | quit")

    plans, last_t = {}, None
    while True:
        now = pd.Timestamp.now(tz=config.TIMEZONE).floor("ms")
        for d in (now.normalize() - pd.Timedelta(days=1), now.normalize()):          # normal daily routine
            if d not in plans:
                plans[d] = plan_day(d.date(), np.random.default_rng(int(d.strftime("%Y%m%d"))))
        with lock:
            intervals = plans[now.normalize() - pd.Timedelta(days=1)] + plans[now.normalize()] + extra
            faulty = fault_until is not None and now < fault_until
        r = make_readings(intervals, pd.DatetimeIndex([now]).as_unit("ns"), np.random.default_rng()).iloc[0]
        if last_t is not None:
            energy += r["power"] * (now - last_t).total_seconds() / 3.6e6
        last_t = now
        row_id += 1
        v, i, p, pf, e = (0, 0, 0, 0, 0) if faulty else (r["voltage"], r["current"], r["power"], r["pf"], energy)
        stamp = now.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S.%f+00")
        line = f"{row_id},{stamp},{v:.1f},{i:.3f},{p:.1f},{e:.3f},{pf:.2f},{r['frequency']:.1f}\n"
        with open(config.LIVE_CSV, "a") as f:
            f.write(line)
        print(f"  {now:%H:%M:%S}  {p:7.1f} W  {v:6.1f} V  pf {pf:.2f}")
        time.sleep(max(1.0, a.every + np.random.uniform(-2, 2)))


if __name__ == "__main__":
    main()
