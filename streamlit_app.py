import json, joblib, pandas as pd, streamlit as st
from pathlib import Path

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "model_linreg_pipeline.pkl"
CATS_PATH  = BASE / "model" / "categories.json"

st.set_page_config(page_title="Food Delivery ETA", page_icon="⏱")
st.title("Food Delivery Time Predictor")
st.caption("Model: Linear Regression (Pipeline dengan OneHot + Scaling)")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(CATS_PATH) as f:
        cats = json.load(f)
    return model, cats

model, cats = load_artifacts()

# kolom sesuai training
num_cols = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
cat_cols = list(cats.keys())

with st.form("form"):
    c1, c2 = st.columns(2)
    with c1:
        distance = st.number_input("Distance (km)", 0.1, 50.0, 5.0, 0.1)
        prep     = st.number_input("Preparation Time (min)", 1, 120, 20, 1)
        exp      = st.number_input("Courier Experience (yrs)", 0, 50, 3, 1)
    with c2:
        weather  = st.selectbox("Weather", cats["Weather"])
        traffic  = st.selectbox("Traffic Level", cats["Traffic_Level"])
        tod      = st.selectbox("Time of Day", cats["Time_of_Day"])
        vehicle  = st.selectbox("Vehicle Type", cats["Vehicle_Type"])

    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([{
        "Distance_km": distance,
        "Preparation_Time_min": prep,
        "Courier_Experience_yrs": exp,
        "Weather": weather,
        "Traffic_Level": traffic,
        "Time_of_Day": tod,
        "Vehicle_Type": vehicle,
    }], columns=num_cols + cat_cols)

    pred = model.predict(row)[0]
    st.success(f"Estimated delivery time: **{pred:.1f} minutes**")
    st.caption("Catatan: estimasi rata-rata error historis ± ~6 menit (MAE).")

