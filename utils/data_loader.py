"""
Cached data and model artifact loaders for the Panama Load Telemetry dashboard.
"""

import json
import os
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART_DIR = os.path.join(BASE_DIR, "artifacts")


def _load_table(stem):
    pq_path = os.path.join(ART_DIR, f"{stem}.parquet")
    gz_path = os.path.join(ART_DIR, f"{stem}.csv.gz")
    if os.path.exists(pq_path):
        df = pd.read_parquet(pq_path)
    elif os.path.exists(gz_path):
        df = pd.read_csv(gz_path)
    else:
        raise FileNotFoundError(f"Artefak tabel '{stem}' tidak ditemukan di {ART_DIR}. Jalankan scripts/build_artifacts.py terlebih dahulu.")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@st.cache_data(show_spinner=False)
def load_master_hourly():
    return _load_table("master_hourly")


@st.cache_data(show_spinner=False)
def load_forecast_st():
    return _load_table("forecast_st_168h")


@st.cache_data(show_spinner=False)
def load_forecast_lt():
    return _load_table("forecast_lt_12m")


@st.cache_data(show_spinner=False)
def load_audit_stats():
    json_path = os.path.join(ART_DIR, "audit_stats.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_gam_pre():
    return joblib.load(os.path.join(ART_DIR, "model_gam_pre.pkl"))


@st.cache_resource(show_spinner=False)
def load_gam_covid():
    return joblib.load(os.path.join(ART_DIR, "model_gam_covid.pkl"))


@st.cache_resource(show_spinner=False)
def load_gbm_full():
    return joblib.load(os.path.join(ART_DIR, "model_gbm_full.pkl"))


@st.cache_resource(show_spinner=False)
def load_preprocessors():
    return joblib.load(os.path.join(ART_DIR, "preprocessors.pkl"))
