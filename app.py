"""
Main Streamlit entry point and st.navigation router.
Clean, restrained SaaS navigation with concise natural titles.
"""

import streamlit as st
from utils.ui_components import inject_styles

st.set_page_config(
    page_title="Analisis & Forecasting Beban Listrik Panama",
    page_icon=":material/bolt:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

pages = {
    "Analisis Data & Evaluasi": [
        st.Page("pages/0_Executive_Overview.py", title="Ringkasan", icon=":material/dashboard:", default=True),
        st.Page("pages/1_Eksplorasi_&_Audit_Statistik.py", title="Eksplorasi & Audit Data", icon=":material/query_stats:"),
        st.Page("pages/2_Evaluasi_Model_&_CV.py", title="Evaluasi Model & CV", icon=":material/compare_arrows:"),
        st.Page("pages/3_Sensitivitas_Cuaca_&_COVID19.py", title="Cuaca, Libur, & COVID-19", icon=":material/thermostat:"),
    ],
    "Proyeksi & Simulasi": [
        st.Page("pages/4_Forecast_Jangka_Pendek_7Hari.py", title="Forecast 7 Hari", icon=":material/bolt:"),
        st.Page("pages/5_Forecast_Jangka_Panjang_12Bulan.py", title="Forecast 12 Bulan", icon=":material/calendar_month:"),
        st.Page("pages/6_Simulator_WhatIf.py", title="Simulator What-If", icon=":material/tune:"),
    ],
}

pg = st.navigation(pages)
st.sidebar.caption("Dataset: Beban Listrik Panama (2015–2020 · 48.048 jam)")
pg.run()
