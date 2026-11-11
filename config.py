"""
config.py - ALL settings of the project in one place.
Change values here; no other file needs to be edited.
"""

# ---------------------------------------------------------------- files
LIVE_CSV = "data/live_data.csv"          # the CSV the hardware script keeps appending to
                                         # (can also be a FOLDER: then the newest .csv inside is used)
RAW_REAL_LOG = "data/raw/pzem_logs_rows.csv"   # the real day we recorded (used for cleaning + testing)
CLEAN_REAL_DAY = "data/clean_real_day.csv"     # output of data_cleaning.py
TRAINING_DATA = "data/training_data.csv.gz"    # output of generate_data.py: our room, every 20 s
HOSTEL_HOURLY = "data/hostel_hourly.csv.gz"    # output of generate_data.py: hourly energy of 40 rooms
FORECAST_LOG = "data/forecasts.csv"      # every daily forecast is saved here (never changed later)
EVENT_LOG = "data/events.csv"            # every alert / trip is saved here

FORECAST_MODEL = "models/forecast_model.joblib"
ANOMALY_MODEL = "models/anomaly_model.joblib"
METRICS_FILE = "models/metrics.json"     # model scores, shown on the dashboard
REPORT_DIR = "reports"                   # training charts (PNG) for the project report

# ---------------------------------------------------------------- time
TIMEZONE = "Asia/Kolkata"                # all charts / days / hours are in Indian time
SAMPLE_SECONDS = 20                      # hardware writes one row about every 20 s
MAX_GAP_SECONDS = 60                     # longer silence = data missing (no energy is added for it)
NO_DATA_SECONDS = 90                     # no new row for this long -> dashboard shows "NO DATA"

# ---------------------------------------------------------------- college calendar (known in advance)
# Students study late during exams -> more energy at night. The forecast model uses this.
EXAM_PERIODS = [("2026-05-18", "2026-05-31"),     # end-sem (even semester)
                ("2026-09-21", "2026-10-04"),     # mid-sem
                ("2026-11-16", "2026-11-29")]     # end-sem (odd semester)  <- update for your college

# ---------------------------------------------------------------- safety rules (TRIP)
TRIP_INSTANT_W = 1500                    # any single reading above this -> TRIP immediately
TRIP_SUSTAINED_W = 1000                  # above this ...
TRIP_SUSTAINED_SECONDS = 30              # ... continuously for this long -> TRIP
TRIP_HOLD_SECONDS = 120                  # after a trip, do not send another trip for this long

# ---------------------------------------------------------------- other alerts
OVER_VOLTAGE_V = 253                     # 230 V + 10 %
UNDER_VOLTAGE_V = 195                    # 230 V - 15 %
ANOMALY_PROBABILITY = 0.5                # ML model: above this = "heating appliance detected"
ANOMALY_CONFIRM_READINGS = 2             # ... for this many readings in a row (avoids false alarms)
ALERT_END_SECONDS = 60                   # an alert ends only after its cause is gone for this long
BANNED_APPLIANCES = ["iron", "induction", "heater/rod"]   # not allowed in rooms -> ALERT
                                         # (kettle is detected and shown, but treated as allowed; add
                                         #  "kettle" here if your hostel bans it)

# ---------------------------------------------------------------- bill
RATE_PER_KWH = 7.0                       # Rs per unit (flat rate)

# ---------------------------------------------------------------- live loop / dashboard
POLL_SECONDS = 2                         # how often the CSV is checked for new rows
DASHBOARD_PORT = 8000
