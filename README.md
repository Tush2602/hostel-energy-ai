# AI Smart Hostel Energy Monitor — AI/ML part

A PZEM-004T + ESP32 meter in a hostel room writes one reading every ~20 s to a CSV file.
This software watches that file **live** and:

1. **Trips** the room (prints the TRIP signal) when power is dangerous: **> 1500 W instantly**, or **> 1000 W for 30 s**.
2. **AI anomaly detection**: recognises a heating appliance from its electrical signature (kettle / iron / induction / heater) and alerts if it is not allowed.
3. **AI forecast**: every midnight it predicts the **next 24 hours, hour by hour** (the day-ahead request that would go to the substation) and the **electricity bill** (₹7/unit).
4. Shows everything on a **live dashboard**, including **forecast vs actual** for every hour (auto-updates as each hour completes).

---

## Files (only 8 Python files)

| File | What it does |
|---|---|
| `config.py` | **All settings**: live CSV path, trip limits, tariff, exam dates … change things here only |
| `data_cleaning.py` | Cleans the raw PZEM data (UTC → IST, sensor faults, energy). Run it to clean our real recorded day |
| `generate_data.py` | Makes 180 days of realistic training data from the appliances we measured on our real day |
| `models.py` | The two AI models (Random Forest) and the features they use |
| `train.py` | Trains + tests both models, saves them, makes charts for the report |
| `monitor.py` | The live brain: reads new rows, trip rules, AI alerts, midnight forecast |
| `app.py` | **Start the live system** + dashboard |
| `simulate_live.py` | Fake hardware logger for testing / demo without the ESP32 |
| `dashboard/` | The web page (HTML + CSS + JS, charts work offline) |

Folders: `data/` (raw log, cleaned data, generated data, live CSV), `models/` (trained models + scores), `reports/` (charts for the report/PPT).

---

## Setup (once)

```
python -m venv venv
venv\Scripts\activate            (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

The trained models are already included, so you can go straight to **Run live**.
To rebuild everything from scratch (takes ~3 minutes):

```
python data_cleaning.py     # cleans our real day  -> data/clean_real_day.csv, reports/real_day.png
python generate_data.py     # training data        -> data/training_data.csv.gz, data/hostel_hourly.csv.gz
python train.py             # trains + tests       -> models/*.joblib, models/metrics.json, reports/*.png
```

## Run live (with the hardware)

1. The hardware script keeps appending rows to `data/live_data.csv`
   (columns `id,created_at,voltage,current,power,energy,pf,frequency` — exactly like our recorded log).
   If the file is somewhere else, change `LIVE_CSV` in `config.py` (it can also be a folder: the newest CSV inside is used).
2. `python app.py` → the dashboard opens at **http://127.0.0.1:8000**. TRIP signals and alerts are printed in this window.

## Run a demo without hardware

Terminal 1: `python simulate_live.py --history 7`  (writes 7 days of past data, then a new row every 20 s)
Terminal 2: `python app.py`

In terminal 1 type `heater` + Enter → AI says "HEATER detected – not allowed", and after 30 s above 1000 W the **TRIP** is printed and shown in red on the dashboard.
Other commands: `kettle`, `iron`, `induction` (instant trip at 2000 W), `bulb`, `laptop`, `fault`.

(`simulate_live.py --history` **overwrites** `data/live_data.csv` — don't use it on the real hardware file.)

---

## How it works (for the presentation)

```
ESP32 + PZEM  ->  live CSV  ->  cleaning  ->  1) TRIP rules (fixed limits, safety)          -> "TRIP SIGNAL SENT"
                                          ->  2) AI anomaly model (Random Forest classifier) -> "IRON detected"
                                          ->  3) AI forecast model (Random Forest regression)-> next 24 h + bill
                                          ->  dashboard (Flask + HTML/JS)
```

**Data.** We recorded one real day (4–5 Nov, 22 116 readings) and wrote a load diary (which appliance was on when).
One day is far too little to train AI, so `generate_data.py` builds 180 days of data from the **appliances we measured**
(LED 9 W, chargers, laptop, 100 W bulb, 600 W kettle – their real power, power factor and behaviour) plus rated values for
appliances we could not safely test (iron 1000 W, induction 1200–2000 W, heater 1–2 kW). A student timetable
(classes, sleep, sports, exam weeks) decides when things are used, and the measured voltage pattern of our hostel is reused.
For the hostel-level forecast, 40 rooms with different habits are generated.

**Cleaning.** Times UTC → IST; sensor faults (voltage 0) are marked; the meter's energy counter resets to 0 during a fault,
so energy is calculated as **power × time** instead. Check on the real day: 519 Wh by power × time vs 515 Wh by the meter counter.

**Forecast model.** At midnight, for each of the 24 hours of the new day it uses only past data: same hour yesterday,
same hour last week, 7-day average, yesterday's total, weekday, hour, exam period (college calendar), and the room's typical week.
Banned appliances are not counted as normal demand (they get tripped / alerted).

**Anomaly model.** For each reading: real power, power corrected to 230 V (P × (230/V)²), reactive power √(S² − P²),
power factor, change from the last reading, 1–2 minute rolling mean/max/min/std. A heater or kettle is resistive (pf ≈ 1,
no reactive power) and large; a laptop charger has pf ≈ 0.55; an iron switches on/off by its thermostat; induction changes power levels.

**Why the TRIP is not done by AI:** a safety decision must be 100 % predictable and explainable, so it is a fixed rule.
AI gives the early warning and tells *which* appliance it is.

## Results (tested on days the models never saw — chronological split, last 25 % of days)

| Forecast accuracy (100 % − error) | AI model | "same as yesterday" | "7-day average" |
|---|---|---|---|
| Whole hostel (40 rooms), per day | **94.5 %** | 87.0 % | 83.7 % |
| Whole hostel, per hour | **85.6 %** | 75.3 % | 73.7 % |
| One room, per day | **77.2 %** | 67.7 % | 72.1 % |

One student is random (one extra cup of tea changes the day); 40 rooms together are very predictable — that is why
the day-ahead request to the substation is made for the hostel total.

| Anomaly model | Score |
|---|---|
| Precision (alarms that were real) | **99.4 %** |
| Recall (real ones caught) | **100 %** |
| Correct appliance type | **98.8 %** |
| **Our REAL recorded day** | both kettle uses caught, named "kettle" 100 %, **0 false alarms** in 22 116 readings (laptop, chargers, 100 W bulb correctly ignored) |

Replaying the real day through the live monitor: no trip (max was 598 W ✔), both kettle uses alerted, both sensor faults
(3:15 pm, 9:31 am) alerted, over-voltage (up to 274 V) alerted.

Charts for the report: `reports/dashboard_demo.png`, `reports/real_day.png`, `reports/forecast_vs_actual.png`, `reports/anomaly_real_day.png`, `reports/anomaly_confusion.png`.

## Honest limitations
- Training data is generated from one real day; scores on months of real data are unknown until it is collected.
  `train.py` can be re-run any time on more real data.
- Iron / induction / heater signatures use rated values (not measured by us).
- Earth-leakage faults cannot be seen from V, I, P, pf — that needs an RCCB / residual-current sensor (hardware).
- This is supervisory software; the MCB/RCCB remain the real protection.
