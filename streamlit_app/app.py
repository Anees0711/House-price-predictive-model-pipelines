import glob
import json
import os

import numpy as np
import pandas as pd
import psycopg2
import streamlit as st
from joblib import load

from house_prices.preprocess import transform_data

MODELS_DIR = "/opt/project/models"

DB_HOST = "postgres"
DB_NAME = "airflow"
DB_USER = "airflow"
DB_PASS = "airflow"

st.title("House Price Intelligence (MLOps Demo)")

def latest_file(pattern: str) -> str:
    files = sorted(glob.glob(os.path.join(MODELS_DIR, pattern)))
    return files[-1] if files else ""

model_path = latest_file("model_*.joblib")
preproc_path = latest_file("preprocessors_*.joblib")

if not model_path or not preproc_path:
    st.error("No model found yet. Run the Airflow DAG first.")
    st.stop()

st.write("Using:", os.path.basename(model_path))

model = load(model_path)
pre = load(preproc_path)

# minimal input form (add more fields later)
overall_qual = st.slider("OverallQual", 1, 10, 5)
gr_liv_area = st.number_input("GrLivArea", min_value=200, max_value=8000, value=1500)
neighborhood = st.text_input("Neighborhood", "NAmes")
ms_zoning = st.text_input("MSZoning", "RL")

row = {
    "Id": 999999,
    "OverallQual": overall_qual,
    "GrLivArea": gr_liv_area,
    "Neighborhood": neighborhood,
    "MSZoning": ms_zoning,
}

input_df = pd.DataFrame([row])

if st.button("Predict"):
    X = transform_data(input_df, pre)
    pred_log = model.predict(X)
    pred = float(np.expm1(pred_log)[0])

    st.success(f"Predicted SalePrice: ${pred:,.0f}")

    # log into Postgres
    conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO app.inference_requests (request_payload, predicted_price)
        VALUES (%s::jsonb, %s)
        """,
        (json.dumps(row), pred),
    )
    conn.commit()
    cur.close()
    conn.close()
    st.write("Logged inference request.")
