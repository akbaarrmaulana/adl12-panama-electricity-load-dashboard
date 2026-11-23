"""
Main Streamlit entry point and st.navigation router.
Soft UI / Neumorphism design with a 5-page layperson-first story architecture.
"""

import streamlit as st
from utils.ui_components import inject_styles

st.set_page_config(
    page_title="Listrik Nasional Panama (2015–2020)",
    page_icon=":material/bolt:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

pages = {
    "Cerita Utama": [
        st.Page("pages/1_Beranda.py", title="Beranda & Ringkasan", icon=":material/home:", default=True),
        st.Page("pages/2_Pola_dan_Faktor_Beban.py", title="Pola & Faktor Pemicu", icon=":material/wb_sunny:"),
        st.Page("pages/3_Akurasi_Model.py", title="Perbandingan Model", icon=":material/verified:"),
    ],
    "Prediksi & Simulasi": [
        st.Page("pages/4_Prediksi_Kebutuhan.py", title="Prediksi 7 Hari & 12 Bulan", icon=":material/trending_up:"),
        st.Page("pages/5_Simulasi_Interaktif.py", title="Simulasi Coba Sendiri", icon=":material/tune:"),
    ],
}

pg = st.navigation(pages)
st.sidebar.markdown(
    '<div class="neu-sidebar-note"><b>Panduan Singkat</b><br>1 GWh = 1.000 MWh<br>Data Nasional Panama 2015–2020 (48.048 jam observasi).</div>',
    unsafe_allow_html=True,
)
pg.run()
