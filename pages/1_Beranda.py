import pandas as pd
import streamlit as st
from utils.charts import make_panama_station_map, make_regime_timeline
from utils.data_loader import load_audit_stats, load_master_hourly
from utils.ui_components import (
    inject_styles,
    render_insight_box,
    render_neu_kpis,
    render_page_header,
    render_story_cards,
)

inject_styles()

df = load_master_hourly()
audit = load_audit_stats()

render_page_header(
    "Pemakaian & Prediksi Listrik Nasional Panama",
    "Ringkasan 5,5 tahun data pemakaian listrik per jam (2015–2020), dampak cuaca tropis & pandemi COVID-19, serta hasil prediksi masa depan dalam bahasa yang mudah dipahami.",
    kicker="Ringkasan Utama",
)

meta = audit["dataset_meta"]
stlf_row = next(r for r in audit["backtest_summary"] if r["col_key"] == "pred_gbm")
covid_sum = audit["covid_summary"]

render_neu_kpis([
    {
        "label": "Rata-rata Pemakaian",
        "value": f"{meta['nat_demand_mean']:,.0f} MWh/jam",
        "hint": f"Dari {meta['rows']:,} jam pencatatan (2015–2020)",
        "badge": "Skala Nasional",
        "tone": "blue",
    },
    {
        "label": "Rekor Beban Tertinggi",
        "value": f"{meta['nat_demand_max']:,.0f} MWh",
        "hint": "Terjadi pada siang hari kerja yang terik",
        "badge": "Puncak Siang",
        "tone": "amber",
    },
    {
        "label": "Penurunan Saat COVID-19",
        "value": f"{covid_sum['mean_delta_pct']:+.2f}%",
        "hint": f"Berkurang {abs(covid_sum['total_lost_gwh']):,.1f} GWh (Mar–Jun 2020)",
        "badge": "Dampak Lockdown",
        "tone": "rose",
    },
    {
        "label": "Akurasi Model Terbaik",
        "value": f"{100.0 - stlf_row['MAPE_2019_NoBO_%']:.2f}%",
        "hint": f"Rata-rata meleset hanya {stlf_row['MAPE_2019_NoBO_%']:.2f}% ({stlf_row['RMSE_2019_NoBO']:.1f} MWh)",
        "badge": "Prediksi 7 Hari",
        "tone": "teal",
    },
])

render_story_cards([
    {
        "step": "1",
        "title": "Cuaca Panas Siang Hari = Puncak Listrik",
        "body": "Aktivitas perkantoran dan pendingin ruangan (AC) membuat pemakaian listrik selalu memuncak pukul 11:00–15:00 dan turun saat malam serta akhir pekan.",
        "tone": "blue",
    },
    {
        "step": "2",
        "title": "Lockdown 2020 Mengubah Pola Konsumsi",
        "body": f"Sejak Maret 2020, penutupan kantor dan sekolah memangkas pemakaian listrik rata-rata {abs(covid_sum['mean_delta_mwh']):,.0f} MWh/jam dibanding kondisi normal.",
        "tone": "rose",
    },
    {
        "step": "3",
        "title": "Dua Model untuk Dua Kebutuhan",
        "body": "Model Machine Learning Boosting paling jitu untuk jadwal operasional 7 hari ke depan, sedangkan model GAM paling andal untuk perencanaan 12 bulan ke depan.",
        "tone": "teal",
    },
])

with st.container(border=True):
    c_left, c_right = st.columns([8, 4], gap="large")
    with c_left:
        regime_filter = st.radio(
            "Tampilkan Periode",
            ["Semua Tahun (2015–2020)", "Sebelum Pandemi (2015–2019)", "Masa Pandemi (2020)"],
            horizontal=True,
        )
        st.plotly_chart(
            make_regime_timeline(df, regime_filter=regime_filter),
            width="stretch",
            theme=None,
            config={"displaylogo": False},
        )
    with c_right:
        map_metric = st.selectbox(
            "Informasi Peta 3 Kota",
            ["Rata-rata Suhu (°C)", "Korelasi Beban (r)"],
            index=0,
        )
        st.plotly_chart(
            make_panama_station_map(df, audit["vif_table"], metric_mode=map_metric),
            width="stretch",
            theme=None,
            config={"displaylogo": False},
        )
        st.caption("Tocumen (Kota Panama) merupakan pusat populasi & beban listrik utama; suhu ketiganya sangat selaras.")

render_insight_box(
    "Cara Membaca Dashboard Ini",
    "Gunakan menu di sebelah kiri secara berurutan: lihat pola cuaca & pandemi di 'Pola & Faktor Pemicu', bandingkan ketepatan 5 model di 'Perbandingan Model', cek proyeksi ke depan di 'Prediksi 7 Hari & 12 Bulan', atau uji skenario sendiri di 'Simulasi Coba Sendiri'.",
    tone="blue",
)

with st.expander("Lihat Detail Teknis: Pembagian Data Latih & Uji (Train–Test Split)"):
    st.caption("Urutan waktu dipertahankan secara ketat agar model tidak 'mengintip' data masa depan saat diuji.")
    st.dataframe(pd.DataFrame(audit["split_summary"]), width="stretch", hide_index=True)
    st.markdown(
        "- **Evaluasi Konsistensi (Cross-Validation):** `TimeSeriesSplit` 3-fold pada `2015–2018` menghasilkan rata-rata RMSE `78,23 MWh` (GAM) dan `47,75 MWh` (HistGradientBoosting).\n"
        "- **Pemisahan Jangka Waktu:** Jangka pendek (`7` hari / `168` jam) memakai HistGradientBoosting dengan riwayat minggu lalu; jangka panjang (`12` bulan / `8.760` jam) memakai GAM murni cuaca & kalender."
    )
