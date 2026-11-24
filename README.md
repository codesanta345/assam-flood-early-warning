# Assam Flood Early Warning System

Predicts the chance of a flood starting in the next 24 hours at 5 river stations in Assam
(Guwahati, Dibrugarh, Jorhat, Tezpur, Silchar) using XGBoost.

- Live river levels: CWC Flood Forecast System (ffs.india-water.gov.in)
- Live weather: Open-Meteo
- Dashboard: Streamlit

## Run
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
1. Run notebooks 01 → 04 in `notebooks/` (they create `data/flood_features.csv` and the model in `models/`).
2. `streamlit run app/app.py`

Live mode needs internet from inside India (the CWC portal does not answer from abroad).