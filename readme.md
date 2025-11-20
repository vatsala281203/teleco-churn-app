# Telecom Churn Predictor (Streamlit)

A simple Streamlit app to predict telecom customer churn per-customer and visualize results.

## What this repo contains
- `app.py` - Streamlit application (prediction + per-customer visuals)
- `Models/cols.json` - feature column list used by the model
- `Models/scaler.pkl` - scaler for numeric features
- `Models/xgb_churn_model.pkl` - XGBoost model
- `requirements.txt` - Python dependencies
- `Dataset/sample_data.csv` - (optional) sample/historical data for cohort visuals

> ⚠️ Do not commit large datasets or secrets to this repository.

## Run locally

1. Create Python environment (recommended)
```bash
conda create -n teleco_churn python=3.10 -y
conda activate teleco_churn

2. Install dependencies
pip install -r requirements.txt

3. Run Streamlit
streamlit run app.py


streamlit app - https://churninsightai.streamlit.app/
