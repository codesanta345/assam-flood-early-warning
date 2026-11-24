# =====================================================================
#  ASSAM FLOOD EARLY WARNING  -  Streamlit app (laptop layout)
#  Run from project folder:   streamlit run app/app.py
# =====================================================================

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import live

PROJECT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT / "data" / "flood_features.csv"
MODEL_FILE = PROJECT / "models" / "xgb_final_onset_24h.pkl"

STATIONS = {"ASM_GW01": "Guwahati", "ASM_DB02": "Dibrugarh", "ASM_JR05": "Jorhat",
            "ASM_TZ04": "Tezpur", "ASM_SL03": "Silchar"}

# colours taken from the design
TEAL, TEAL_DARK = "#4A8796", "#3A6F7D"
YELLOW, YELLOW_EDGE = "#FBD46D", "#E9A23B"
CORAL = "#EE6B5A"
SKY = "#A8D8F0"
SLATE = "#3E4C59"
NICE_NAMES = {"humidity_mean_24h": "Humidity (24h avg)", "level_gap": "Gap from danger level",
              "level_lag_1h": "Level 1h ago", "level_lag_3h": "Level 3h ago", "level_lag_6h": "Level 6h ago",
              "level_lag_24h": "Level 24h ago", "level_lag_48h": "Level 48h ago",
              "rain_sum_168h": "Rain last 7 days", "rain_sum_72h": "Rain last 3 days",
              "satrain_sum_168h": "Satellite rain 7 days", "satrain_sum_72h": "Satellite rain 3 days",
              "upstream_gap": "Upstream station level", "upstream_rise_24h": "Upstream rise 24h",
              "has_upstream": "Has upstream station", "month_sin": "Season (month)", "month_cos": "Season (month)",
              "humidity_pct": "Humidity now", "level_rise_24h": "Level rise 24h",
              "stn_ASM_DB02": "Station: Dibrugarh", "stn_ASM_JR05": "Station: Jorhat",
              "stn_ASM_SL03": "Station: Silchar", "stn_ASM_GW01": "Station: Guwahati",
              "stn_ASM_TZ04": "Station: Tezpur", "is_monsoon": "Monsoon season"}
STATUS_COLOR = {"Flood ongoing": "#B23A2E", "High": CORAL, "Medium": YELLOW_EDGE, "Low": TEAL}

st.set_page_config(page_title="Assam Flood Early Warning", page_icon="🌊", layout="wide")

# ---------------------------------------------------------------------
#  STYLE
# ---------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, label {{ font-family: 'Nunito', sans-serif !important; }}
header[data-testid="stHeader"], #MainMenu, footer {{ display: none; }}
.stApp {{ background: #EEF2F4; }}
.block-container {{ max-width: 1150px; padding-top: 1.2rem; padding-bottom: 2rem; }}

/* top bar */
.st-key-topbar {{ background: {TEAL}; border-radius: 14px; padding: 12px 22px; min-height: 64px; justify-content: center; }}
.st-key-topbar .stMarkdown {{ margin-bottom: 0 !important; }}
.st-key-topbar p {{ color: white; margin: 0; }}
.topbar-title {{ color: white; font-size: 24px; font-weight: 800; line-height: 40px; }}
.bell {{ text-align: right; color: white; font-size: 24px; position: relative; padding-top: 2px; }}
.bell .badge {{ position: absolute; top: -2px; right: -8px; background: {YELLOW}; color: {SLATE};
               font-size: 12px; font-weight: 800; border-radius: 10px; padding: 0 6px; }}
.st-key-back button {{ background: transparent; border: none; padding: 0; min-height: 0; }}
.st-key-back button p {{ color: white; font-size: 30px; font-weight: 800; line-height: 40px; }}
.st-key-back button:hover p {{ color: {YELLOW}; }}
.st-key-go_status button p, .st-key-go_rain button p, .st-key-go_water button p {{ font-size: 19px; font-weight: 800; }}
.st-key-next button p {{ font-size: 16px; font-weight: 800; }}

/* white panels */
.st-key-panel, .st-key-panel2, .st-key-panel3 {{ background: white; border-radius: 16px; padding: 20px 26px; }}

/* "Today" pill */
.today {{ display: inline-block; background: {YELLOW}; border: 2px solid {YELLOW_EDGE}; color: {SLATE};
          font-weight: 800; border-radius: 10px; padding: 4px 18px; font-size: 16px; }}
.when {{ color: {SLATE}; font-weight: 700; margin-left: 12px; font-size: 15px; white-space: nowrap; }}

/* buttons */
.st-key-go_status button, .st-key-go_rain button, .st-key-go_water button, .st-key-next button {{
    border-radius: 12px; font-weight: 800; font-size: 18px; padding: 12px 0; border: 2px solid; }}
.st-key-go_status button {{ background: {YELLOW}; border-color: {YELLOW_EDGE}; color: {SLATE}; }}
.st-key-go_rain   button {{ background: {CORAL};  border-color: #C9503F; color: white; }}
.st-key-go_water  button {{ background: {TEAL};   border-color: {TEAL_DARK}; color: white; }}
.st-key-next      button {{ background: {YELLOW}; border-color: {YELLOW_EDGE}; color: {SLATE}; font-size: 16px; }}
.st-key-go_status button:hover, .st-key-go_rain button:hover,
.st-key-go_water button:hover, .st-key-next button:hover {{ filter: brightness(0.95); }}

/* labels, value boxes, bars */
.label {{ color: {SLATE}; font-weight: 800; font-size: 15px; margin: 14px 0 6px 0; }}
.track {{ background: {SLATE}; border-radius: 20px; height: 26px; width: 100%; }}
.fill  {{ border-radius: 20px; height: 26px; }}
.vbox  {{ display: inline-block; border: 2px solid {SLATE}; border-radius: 6px; padding: 2px 12px;
          min-width: 90px; font-weight: 800; color: {SLATE}; margin-top: 8px; }}
.unit  {{ color: {SLATE}; font-weight: 700; margin-left: 8px; font-size: 14px; }}
.status-chip {{ display: inline-block; color: white; font-weight: 800; border-radius: 10px;
                padding: 6px 16px; font-size: 17px; white-space: nowrap; }}
.post-title {{ color: {SLATE}; font-weight: 800; font-size: 22px; text-align: center; margin-top: 6px; }}
.note {{ color: {SLATE}; font-size: 15px; margin-top: 10px; }}

/* always readable widgets (even if the laptop is in dark mode) */
[data-testid="stWidgetLabel"] p, .stRadio label p, .stRadio div[role="radiogroup"] p {{ color: {SLATE} !important; font-weight: 700; }}
[data-baseweb="select"] > div, .stDateInput input {{ background: white !important; color: {SLATE} !important; border-color: #C9D3DA !important; }}
[data-baseweb="select"] span, [data-baseweb="select"] svg {{ color: {SLATE} !important; fill: {SLATE} !important; }}
.src-row {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #E3E8EC; color: {SLATE}; font-size: 15px; }}
.src-ok {{ color: {TEAL_DARK}; font-weight: 800; }}
.src-old {{ color: {YELLOW_EDGE}; font-weight: 800; }}
.src-bad {{ color: #B23A2E; font-weight: 800; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
#  DATA + MODEL
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    return df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

saved = load_model()
model, threshold, features = saved["model"], saved["threshold"], saved["features"]

@st.cache_data
def load_history():
    df = load_data()
    df["probability"] = model.predict_proba(df[features])[:, 1]
    df["rain_mm"] = np.expm1(df["rainfall_mm"])          # undo log transform for display
    return df

@st.cache_data(ttl=1800, show_spinner="Getting live river levels (CWC) and weather (Open-Meteo)...")
def load_live(now):
    frame, info = live.build_live_frame(now)
    if not frame.empty:
        frame["probability"] = model.predict_proba(frame[features])[:, 1]
    return frame, info

history = load_history()


def get_status(level_gap, prob):
    if level_gap >= 0:
        return "Flood ongoing"
    if prob >= threshold:
        return "High"
    if prob >= 0.3:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------
#  SMALL HTML HELPERS
# ---------------------------------------------------------------------
def bar(value_0_to_1, color):
    w = max(0, min(1, value_0_to_1)) * 100
    return f"<div class='track'><div class='fill' style='width:{w:.0f}%;background:{color}'></div></div>"

def value_box(value, unit):
    return f"<span class='vbox'>{value}</span><span class='unit'>{unit}</span>"

def chip(status):
    return f"<span class='status-chip' style='background:{STATUS_COLOR[status]}'>{status}</span>"

HOUSE_SVG = f"""
<svg viewBox="0 0 200 200" width="210" height="210">
  <circle cx="100" cy="100" r="92" fill="{SKY}"/>
  <g transform="rotate(-12 100 105)">
    <rect x="62" y="88" width="76" height="56" fill="#F6C46B"/>
    <polygon points="52,92 100,52 148,92" fill="{CORAL}"/>
    <rect x="90" y="108" width="20" height="36" fill="#8C5A3C"/>
    <rect x="70" y="100" width="14" height="14" fill="white"/>
    <rect x="116" y="100" width="14" height="14" fill="white"/>
  </g>
  <path d="M8 138 Q30 126 52 138 T96 138 T140 138 T192 138 L192 160 Q100 215 8 160 Z" fill="#3D8FC4"/>
  <path d="M20 152 Q40 142 60 152 T100 152 T140 152 T180 152" stroke="white" stroke-width="3" fill="none"/>
</svg>"""

WATER_TILE = f"""
<div style="display:flex;justify-content:center">
<svg viewBox="0 0 120 120" width="120" height="120">
  <rect width="120" height="120" rx="10" fill="{TEAL}"/>
  <rect x="10" y="62" width="100" height="48" rx="4" fill="#4FB3E8"/>
  <path d="M10 64 Q25 52 40 64 T70 64 T110 64" stroke="white" stroke-width="4" fill="none"/>
  <path d="M10 80 Q25 70 40 80 T70 80 T110 80" stroke="#BFE7FA" stroke-width="3" fill="none"/>
</svg></div>"""


# ---------------------------------------------------------------------
#  NAVIGATION STATE
# ---------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "station" not in st.session_state:
    st.session_state.station = "ASM_GW01"

def go(page):
    st.session_state.page = page

def next_station():
    keys = list(STATIONS.keys())
    i = keys.index(st.session_state.station)
    st.session_state.station = keys[(i + 1) % len(keys)]

page = st.session_state.page


# ---------------------------------------------------------------------
#  TOP BAR
# ---------------------------------------------------------------------
def top_bar(title, alerts):
    with st.container(key="topbar"):
        c1, c2, c3 = st.columns([0.6, 10, 1], vertical_alignment="center")
        with c1:
            if title != "Home":
                st.button("←", key="back", on_click=go, args=("Home",))
        with c2:
            st.markdown(f"<div class='topbar-title'>{title}</div>", unsafe_allow_html=True)
        with c3:
            badge = f"<span class='badge'>{alerts}</span>" if alerts else ""
            st.markdown(f"<div class='bell'>🔔{badge}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------
#  CONTROLS (station + date + hour) - shown on every page
# ---------------------------------------------------------------------
dates = sorted(history["timestamp"].dt.date.unique())
if "mode" not in st.session_state:
    st.session_state.mode = "Live (now)"

def controls():
    c0, c1, c2, c3 = st.columns([1.6, 1.6, 1.6, 2.4])
    c0.radio("Mode", ["Live (now)", "Past data"], key="mode", horizontal=True)
    c1.selectbox("Station", list(STATIONS.keys()), key="station", format_func=lambda s: STATIONS[s])
    if st.session_state.mode == "Past data":
        c2.date_input("Date", value=pd.Timestamp("2024-08-29").date(),
                      min_value=dates[0], max_value=dates[-1], key="day")
        c3.slider("Hour", 0, 23, 22, key="hour")
    else:
        c2.markdown(f"<div class='label' style='margin-top:30px'>Now: {live_now:%d %b %Y, %H:00} IST</div>",
                    unsafe_allow_html=True)


TITLES = {"Home": "Home", "Status": "Flood Status", "Rain": "Rainfall Data", "Water": "Water Level Data"}

live_now = pd.Timestamp.now(tz="Asia/Kolkata").floor("h").tz_localize(None)
live_info = {}

def pick_data():
    global live_info
    if st.session_state.mode == "Live (now)":
        try:
            frame, live_info = load_live(live_now)
        except Exception as e:
            st.error(f"Could not get live data ({e}). Check your internet, or switch Mode to 'Past data'.")
            st.stop()
        if frame.empty:
            frame = history.iloc[0:0]
        return frame, live_now
    day_ = st.session_state.get("day", pd.Timestamp("2024-08-29").date())
    hour_ = st.session_state.get("hour", 22)
    return history, pd.Timestamp(day_) + pd.Timedelta(hours=hour_)

df, now = pick_data()
snap = df[df["timestamp"] == now]
alerts = sum(get_status(g, p) in ("High", "Flood ongoing") for g, p in zip(snap["level_gap"], snap["probability"]))

top_bar(TITLES[page], alerts)
st.write("")
controls()
df, now = pick_data()
is_live = st.session_state.mode == "Live (now)"

station = st.session_state.station
snap = df[df["timestamp"] == now]
row = df[(df["station_id"] == station) & (df["timestamp"] == now)]
if row.empty and page != "Home":
    if is_live:
        err = live_info.get(station, {}).get("error", "No live data")
        st.warning(f"{STATIONS[station]}: {err}. Try another station, or switch Mode to 'Past data'.")
    else:
        st.warning("No data for this time. Please choose another date or hour.")
    st.stop()
row = row.iloc[0] if not row.empty else None
status = get_status(row["level_gap"], row["probability"]) if row is not None else None
hist = df[(df["station_id"] == station) & (df["timestamp"] > now - pd.Timedelta(days=7)) & (df["timestamp"] <= now)]

PILL = "Today" if is_live else "History"
today_html = (f"<span class='today'>{PILL}</span>"
              f"<span class='when'>{STATIONS[station]} &nbsp;·&nbsp; {now:%d %b %Y, %H:00}</span>")


# =====================================================================
#  PAGE: HOME
# =====================================================================
if page == "Home":
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        with st.container(key="panel"):
            st.markdown(f"<div style='display:flex;justify-content:center'>{HOUSE_SVG}</div>",
                        unsafe_allow_html=True)
            st.write("")
            st.button("Flood Status", key="go_status", on_click=go, args=("Status",), width="stretch")
            st.button("Rainfall Data", key="go_rain", on_click=go, args=("Rain",), width="stretch")
            st.button("Water Level Data", key="go_water", on_click=go, args=("Water",), width="stretch")

    with right:
        with st.container(key="panel2"):
            st.markdown(f"<span class='today'>{PILL}</span><span class='when'>{now:%d %b %Y, %H:00}</span>",
                        unsafe_allow_html=True)
            st.markdown("<div class='label' style='font-size:18px'>All stations</div>", unsafe_allow_html=True)
            for sid, name in STATIONS.items():
                r = snap[snap["station_id"] == sid]
                if r.empty:
                    a, b, c = st.columns([1.2, 2, 1.7], vertical_alignment="center")
                    a.markdown(f"<div class='label' style='margin:6px 0'>{name}</div>", unsafe_allow_html=True)
                    b.markdown(bar(0, CORAL), unsafe_allow_html=True)
                    c.markdown("<span class='status-chip' style='background:#9AA5AE'>No data</span>",
                               unsafe_allow_html=True)
                    continue
                r = r.iloc[0]
                s = get_status(r["level_gap"], r["probability"])
                a, b, c = st.columns([1.2, 2, 1.7], vertical_alignment="center")
                a.markdown(f"<div class='label' style='margin:6px 0'>{name}</div>", unsafe_allow_html=True)
                b.markdown(bar(r["probability"], CORAL), unsafe_allow_html=True)
                c.markdown(chip(s), unsafe_allow_html=True)
            st.markdown(f"<div class='note'>Bar = chance that a new flood starts in the next 24 hours "
                        f"(High above {threshold*100:.0f}%).</div>", unsafe_allow_html=True)

        if is_live:
            st.write("")
            with st.container(key="panel3"):
                st.markdown("<div class='label' style='font-size:18px;margin-top:0'>River levels</div>",
                            unsafe_allow_html=True)
                rows_html = ""
                for sid, name in STATIONS.items():
                    inf = live_info.get(sid, {})
                    if "error" in inf:
                        right_txt = "<span class='src-bad'>No data</span>"
                    else:
                        right_txt = (f"<span class='src-ok'>{inf['last_level']:.2f} m</span> "
                                     f"&nbsp;{inf['last_time']:%d %b, %H:%M}")
                    rows_html += f"<div class='src-row'><b>{name}</b><span>{right_txt}</span></div>"
                st.markdown(rows_html, unsafe_allow_html=True)


# =====================================================================
#  PAGE: FLOOD STATUS
# =====================================================================
elif page == "Status":
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        with st.container(key="panel"):
            st.markdown(today_html, unsafe_allow_html=True)
            st.write("")
            st.markdown(WATER_TILE, unsafe_allow_html=True)
            st.markdown(f"<div class='post-title'>{STATIONS[station]}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center;margin-top:8px'>{chip(status)}</div>",
                        unsafe_allow_html=True)

            st.markdown("<div class='label'>Flood start chance (next 24h)</div>", unsafe_allow_html=True)
            st.markdown(bar(row["probability"], CORAL), unsafe_allow_html=True)
            st.markdown(value_box(f"{row['probability']*100:.1f}", "%"), unsafe_allow_html=True)

            st.markdown("<div class='label'>Water level vs danger level (full bar = danger)</div>",
                        unsafe_allow_html=True)
            st.markdown(bar((row["level_gap"] + 3) / 3, SKY), unsafe_allow_html=True)
            st.markdown(value_box(f"{row['water_level_m']:.2f}", f"m &nbsp;(danger {row['danger_level_m']:.2f} m, "
                                  f"gap {row['level_gap']:+.2f} m)"), unsafe_allow_html=True)

            msg = {"Flood ongoing": "Water is already above the danger level.",
                   "High": "High chance that a flood will start within the next 24 hours.",
                   "Medium": "Moderate risk. Keep monitoring water level and rainfall.",
                   "Low": "Low risk of a new flood in the next 24 hours."}[status]
            st.markdown(f"<div class='note'><b>{msg}</b></div>", unsafe_allow_html=True)
            st.write("")
            _, nb = st.columns([2, 1])
            nb.button("NEXT ➜", key="next", on_click=next_station, width="stretch")

    with right:
        with st.container(key="panel2"):
            st.markdown("<div class='label' style='font-size:18px;margin-top:0'>Flood start chance - last 7 days</div>",
                        unsafe_allow_html=True)
            chart = alt.Chart(hist).mark_area(color=CORAL, opacity=0.85, line={"color": "#C9503F"}).encode(
                x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d %b", grid=False)),
                y=alt.Y("probability:Q", title="Probability", scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format="%")))
            rule = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(
                color=SLATE, strokeDash=[6, 4]).encode(y="y:Q")
            st.altair_chart((chart + rule).properties(height=300), width="stretch")
            st.markdown(f"<div class='note'>Dashed line = High risk ({threshold*100:.0f}%)</div>",
                        unsafe_allow_html=True)

        st.write("")
        with st.container(key="panel3"):
            st.markdown("<div class='label' style='font-size:18px;margin-top:0'>What the model looks at most</div>",
                        unsafe_allow_html=True)
            imp = (pd.Series(model.feature_importances_, index=features)
                   .sort_values(ascending=False).head(8).reset_index())
            imp.columns = ["feature", "importance"]
            imp["feature"] = imp["feature"].map(lambda f: NICE_NAMES.get(f, f))
            st.altair_chart(alt.Chart(imp).mark_bar(color=TEAL, cornerRadiusEnd=4).encode(
                x=alt.X("importance:Q", title=None, axis=alt.Axis(grid=False, labels=False, ticks=False)),
                y=alt.Y("feature:N", sort="-x", title=None)).properties(height=230), width="stretch")


# =====================================================================
#  PAGE: RAINFALL DATA
# =====================================================================
elif page == "Rain":
    last24 = hist[hist["timestamp"] > now - pd.Timedelta(hours=24)]
    daily = (hist.assign(day=hist["timestamp"].dt.strftime("%d %b"))
                 .groupby("day", sort=False)["rain_mm"].sum().reset_index())

    left, right = st.columns([1.15, 1.6], gap="large")
    with left:
        with st.container(key="panel"):
            st.markdown(today_html, unsafe_allow_html=True)
            st.markdown("<div class='label'>Rainfall amount (last 24h)</div>", unsafe_allow_html=True)
            st.markdown(value_box(f"{last24['rain_mm'].sum():.0f}", "millimeter (mm)"), unsafe_allow_html=True)
            st.markdown("<div class='label'>Duration (last 24h)</div>", unsafe_allow_html=True)
            st.markdown(value_box(f"{(last24['rain_mm'] > 0).sum()}", "hours of rain"), unsafe_allow_html=True)
            st.markdown("<div class='label'>Rainfall amount (last 72h)</div>", unsafe_allow_html=True)
            st.markdown(value_box(f"{np.expm1(row['rain_sum_72h']):.0f}", "millimeter (mm)"), unsafe_allow_html=True)
            st.markdown("<div class='label'>Rainfall intensity (now)</div>", unsafe_allow_html=True)
            st.markdown(bar(row["rain_mm"] / 30, SKY), unsafe_allow_html=True)
            st.markdown(value_box(f"{row['rain_mm']:.1f}", "mm/h"), unsafe_allow_html=True)

    with right:
        with st.container(key="panel2"):
            st.markdown("<div class='label' style='font-size:18px;margin-top:0;text-align:center'>"
                        "Daily rainfall (mm) - last 7 days</div>", unsafe_allow_html=True)
            st.altair_chart(alt.Chart(daily).mark_bar(color=SKY, stroke="#6FB6DE", strokeWidth=1).encode(
                x=alt.X("day:N", sort=None, title=None, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("rain_mm:Q", title="mm", axis=alt.Axis(grid=False))).properties(height=380),
                width="stretch")


# =====================================================================
#  PAGE: WATER LEVEL DATA
# =====================================================================
elif page == "Water":
    daily = (hist.assign(day=hist["timestamp"].dt.strftime("%d %b"))
                 .groupby("day", sort=False)["water_level_m"].max().reset_index())
    lo = min(daily["water_level_m"].min(), row["danger_level_m"]) - 0.5
    hi = max(daily["water_level_m"].max(), row["danger_level_m"]) + 0.5

    left, right = st.columns([1.15, 1.6], gap="large")
    with left:
        with st.container(key="panel"):
            st.markdown(today_html, unsafe_allow_html=True)
            st.markdown("<div class='label'>Water level</div>", unsafe_allow_html=True)
            st.markdown(value_box(f"{row['water_level_m']:.2f}", "m"), unsafe_allow_html=True)
            st.markdown("<div class='label'>Danger level</div>", unsafe_allow_html=True)
            st.markdown(value_box(f"{row['danger_level_m']:.2f}", "m"), unsafe_allow_html=True)
            st.markdown("<div class='label'>Change in last 24h</div>", unsafe_allow_html=True)
            st.markdown(value_box(f"{row['level_rise_24h']:+.2f}", "m"), unsafe_allow_html=True)
            if row["has_upstream"] == 1:
                up = live.UPSTREAM_OF[station]
                st.markdown(f"<div class='label'>Upstream station {STATIONS[up]} (gap from its danger level)</div>",
                            unsafe_allow_html=True)
                if is_live and "error" in live_info.get(up, {}):
                    st.markdown(value_box("No data", ""), unsafe_allow_html=True)
                else:
                    st.markdown(value_box(f"{row['upstream_gap']:+.2f}", "m"), unsafe_allow_html=True)

    with right:
        with st.container(key="panel2"):
            st.markdown("<div class='label' style='font-size:18px;margin-top:0;text-align:center'>"
                        "Daily highest water level (m) - last 7 days</div>", unsafe_allow_html=True)
            bars = alt.Chart(daily).mark_bar(color=CORAL, stroke="#C9503F", strokeWidth=1, clip=True).encode(
                x=alt.X("day:N", sort=None, title=None, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("water_level_m:Q", title="m", scale=alt.Scale(domain=[lo, hi]), axis=alt.Axis(grid=False)))
            danger = alt.Chart(pd.DataFrame({"y": [row["danger_level_m"]]})).mark_rule(
                color=SLATE, strokeDash=[6, 4], size=2).encode(y="y:Q")
            st.altair_chart((bars + danger).properties(height=380), width="stretch")
            st.markdown("<div class='note' style='text-align:center'>Dashed line = danger level</div>",
                        unsafe_allow_html=True)
            if is_live:
                st.markdown("<div class='note' style='text-align:center'>Live mode: water levels from CWC "
                            "readings (joined with straight lines between readings).</div>",
                            unsafe_allow_html=True)
