# =====================================================================
#  LIVE DATA for the flood app (fully automatic)
#  - weather    : Open-Meteo (free, no API key)        https://open-meteo.com
#  - water level: CWC Flood Forecast System portal     https://ffs.india-water.gov.in
#                 (same API the portal's own website uses; works from inside India)
#  Builds exactly the same features as notebook 02_feature_engineering.
# =====================================================================

import json
import numpy as np
import pandas as pd
import requests

STATION_INFO = {   # station_id: (latitude, longitude, danger level in m)
    "ASM_GW01": (26.1859, 91.7477, 49.68),
    "ASM_DB02": (27.4728, 94.9120, 105.70),
    "ASM_SL03": (24.8333, 92.7789, 19.83),
    "ASM_TZ04": (26.6338, 92.7926, 65.23),
    "ASM_JR05": (26.8606, 94.2405, 85.54),
}
UPSTREAM_OF = {"ASM_JR05": "ASM_DB02", "ASM_TZ04": "ASM_JR05", "ASM_GW01": "ASM_TZ04"}

# CWC gauge code for each of our stations (Jorhat = Neamatighat gauge, Silchar = Annapurnaghat gauge)
CWC_CODES = {"ASM_GW01": ["001-MBDGHY"], "ASM_DB02": ["010-UBDDIB"], "ASM_JR05": ["019-UBDDIB"],
             "ASM_TZ04": ["031-UBDDIB"], "ASM_SL03": ["010-MBDGHY", "01-11-01-007"]}   # Silchar has 2 codes
CWC_BASE = "https://ffs.india-water.gov.in"
CWC_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/126 Safari/537.36",
               "Accept": "application/json, text/plain, */*",
               "Referer": CWC_BASE + "/", "Origin": CWC_BASE}
WATER_LEVEL_CODES = {"HHS", "HHA", "HHX", "HHT", "HZS", "HZF"}

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
WEATHER_VARS = {"precipitation": "rainfall_mm", "temperature_2m": "temperature_c",
                "relative_humidity_2m": "humidity_pct", "wind_speed_10m": "wind_speed_kmh",
                "wind_direction_10m": "wind_direction_deg", "shortwave_radiation": "solar_radiation_wm2",
                "pressure_msl": "pressure_hpa"}


# ---------------------------------------------------------------------
# 1. Weather: last 9 days, hourly, for one station
# ---------------------------------------------------------------------
def fetch_weather(station_id):
    lat, lon, _ = STATION_INFO[station_id]
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(WEATHER_VARS),
              "past_days": 9, "forecast_days": 1, "timezone": "Asia/Kolkata"}
    r = requests.get(OPEN_METEO, params=params, timeout=20)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    w = pd.DataFrame(hourly).rename(columns=WEATHER_VARS)
    w["timestamp"] = pd.to_datetime(w.pop("time"))
    w["station_id"] = station_id
    w["satellite_rainfall_mm"] = w["rainfall_mm"]      # same source for both rain columns
    return w


# ---------------------------------------------------------------------
# 2. Water level: readings from CWC for the last 9 days
# ---------------------------------------------------------------------
def _expr(field, operator, value):
    return {"expression": {"valueIsRelationField": False, "fieldName": field,
                           "operator": operator, "value": value}}

def fetch_cwc_levels(station_id, since):
    """Try each CWC code of the station; return the first one that has readings."""
    lv = pd.DataFrame(columns=["timestamp", "water_level_m"])
    for code in CWC_CODES[station_id]:
        lv = _fetch_one_code(station_id, code, since)
        if not lv.empty:
            break
    return lv

def _fetch_one_code(station_id, code, since):
    spec = {"where": _expr("id.stationCode", "eq", code),
            "and": _expr("id.dataTime", "gt", since.strftime("%Y-%m-%dT%H:%M:%S"))}
    r = requests.get(CWC_BASE + "/iam/api/new-entry-data/specification/",
                     params={"specification": json.dumps(spec, separators=(",", ":"))},
                     headers=CWC_HEADERS, timeout=40)
    r.raise_for_status()
    danger = STATION_INFO[station_id][2]
    rows = []
    for rec in r.json():
        i = rec.get("id", {})
        if i.get("datatypeCode") not in WATER_LEVEL_CODES:
            continue
        value = rec.get("dataValidatedValue")
        if value is None:
            value = rec.get("dataValue")
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not (danger - 40 < value < danger + 15):      # drops -9990 sentinels and typos
            continue
        rows.append((pd.to_datetime(i.get("dataTime"), errors="coerce"), value))
    lv = pd.DataFrame(rows, columns=["timestamp", "water_level_m"]).dropna()
    if lv.empty:
        return lv
    lv["timestamp"] = lv["timestamp"].dt.floor("h")
    return lv.groupby("timestamp", as_index=False)["water_level_m"].median()

def level_on_hours(hours, readings):
    """Put CWC readings on the hourly time line: straight lines between readings,
    last reading carried forward to now."""
    s = readings.set_index("timestamp")["water_level_m"]
    full = s.reindex(s.index.union(pd.DatetimeIndex(hours))).sort_index()
    full = full.interpolate(method="time").ffill().bfill()
    return full.reindex(pd.DatetimeIndex(hours)).values


# ---------------------------------------------------------------------
# 3. Feature engineering (same steps as notebook 02)
#    raw: timestamp, station_id, weather columns, water_level_m, danger_level_m
# ---------------------------------------------------------------------
def build_features(raw):
    df = raw.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    df["month"] = df["timestamp"].dt.month
    df["hour"] = df["timestamp"].dt.hour
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_monsoon"] = df["month"].between(5, 9).astype(int)
    df["wind_sin"] = np.sin(np.radians(df["wind_direction_deg"]))
    df["wind_cos"] = np.cos(np.radians(df["wind_direction_deg"]))

    df["level_gap"] = df["water_level_m"] - df["danger_level_m"]

    df["upstream_station"] = df["station_id"].map(UPSTREAM_OF)
    up = df[["timestamp", "station_id", "level_gap"]].rename(
        columns={"station_id": "upstream_station", "level_gap": "upstream_gap"})
    df = df.merge(up, on=["timestamp", "upstream_station"], how="left")
    df["upstream_gap"] = df["upstream_gap"].fillna(0)
    df["has_upstream"] = df["upstream_station"].notna().astype(int)
    df = df.drop(columns=["upstream_station"])
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    station = df.groupby("station_id")
    for h in [1, 3, 6, 12, 24, 48]:
        df[f"level_lag_{h}h"] = station["level_gap"].shift(h)
        df[f"level_rise_{h}h"] = df["level_gap"] - df[f"level_lag_{h}h"]
    df["upstream_rise_24h"] = df["upstream_gap"] - station["upstream_gap"].shift(24)

    for w in [3, 6, 12, 24, 48, 72, 168]:
        df[f"rain_sum_{w}h"] = station["rainfall_mm"].transform(lambda x: x.rolling(w, min_periods=1).sum())
        df[f"satrain_sum_{w}h"] = station["satellite_rainfall_mm"].transform(lambda x: x.rolling(w, min_periods=1).sum())
    df["rain_max_24h"] = station["rainfall_mm"].transform(lambda x: x.rolling(24, min_periods=1).max())
    df["humidity_mean_24h"] = station["humidity_pct"].transform(lambda x: x.rolling(24, min_periods=1).mean())
    df["pressure_change_24h"] = df["pressure_hpa"] - station["pressure_hpa"].shift(24)

    df["rain_mm"] = df["rainfall_mm"]                      # raw copy for display
    rain_cols = [c for c in df.columns if c.startswith(("rain", "satrain", "satellite_rain")) and c != "rain_mm"]
    for col in rain_cols:
        df[col] = np.log1p(df[col])

    for sid in STATION_INFO:
        df[f"stn_{sid}"] = (df["station_id"] == sid).astype(int)
    return df


# ---------------------------------------------------------------------
# 4. Everything together: live frame for all stations
#    returns (frame, info)  info[station] = last reading time/value or error
# ---------------------------------------------------------------------
def build_live_frame(now):
    parts, info = [], {}
    since = now - pd.Timedelta(days=9)
    for sid, (_, _, danger) in STATION_INFO.items():
        try:
            readings = fetch_cwc_levels(sid, since)
        except Exception as e:
            info[sid] = {"error": f"CWC not reachable ({type(e).__name__})"}
            continue
        readings = readings[readings["timestamp"] <= now]
        if readings.empty:
            info[sid] = {"error": "No water level reading from CWC in the last 9 days"}
            continue
        last = readings.iloc[-1]
        info[sid] = {"last_time": last["timestamp"], "last_level": last["water_level_m"],
                     "n_readings": len(readings)}

        w = fetch_weather(sid)
        w = w[w["timestamp"] <= now].copy()
        w["water_level_m"] = level_on_hours(w["timestamp"], readings)
        w["danger_level_m"] = danger
        parts.append(w)

    if not parts:
        return pd.DataFrame(), info
    raw = pd.concat(parts, ignore_index=True)
    cols = list(WEATHER_VARS.values()) + ["satellite_rainfall_mm"]
    raw[cols] = raw.groupby("station_id")[cols].transform(lambda x: x.interpolate(limit_direction="both"))
    feats = build_features(raw)
    # upstream station has no CWC data -> use the station's own level instead of a fake 0
    got = set(raw["station_id"])
    for down, up in UPSTREAM_OF.items():
        if down in got and up not in got:
            m = feats["station_id"] == down
            feats.loc[m, "upstream_gap"] = feats.loc[m, "level_gap"]
            feats.loc[m, "upstream_rise_24h"] = feats.loc[m, "level_rise_24h"]
    first = feats.groupby("station_id")["timestamp"].transform("min")
    return feats[feats["timestamp"] >= first + pd.Timedelta(hours=48)].reset_index(drop=True), info
