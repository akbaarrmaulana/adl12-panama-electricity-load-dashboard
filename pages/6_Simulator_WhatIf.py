import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from utils.charts import make_whatif_24h_chart, plot_mpl_whatif_delta_and_temp
from utils.data_loader import (
    load_audit_stats,
    load_gam_covid,
    load_gam_pre,
    load_gbm_full,
    load_preprocessors,
)
from utils.inference import HOLIDAY_LABELS, simulate_whatif_24h
from utils.ui_components import render_methodology_drawer, render_page_header, render_shadcn_kpis

audit = load_audit_stats()
gam_pre = load_gam_pre()
gam_cov = load_gam_covid()
gbm_pkg = load_gbm_full()
preproc = load_preprocessors()

render_page_header(
    "Simulator What-If Beban Listrik",
    "Simulasi prediksi beban per jam dan profil 24 jam secara real-time berdasarkan perubahan cuaca 3 kota, kategori hari libur, serta rezim COVID-19.",
)

col_ctrl, col_out = st.columns([4, 8], gap="large")

month_names = {1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November", 12: "Desember"}
dow_names = {0: "Senin", 1: "Selasa", 2: "Rabu", 3: "Kamis", 4: "Jumat", 5: "Sabtu", 6: "Minggu"}

with col_ctrl:
    st.subheader("Parameter Skenario")
    model_choice = st.radio(
        "Model Inferensi",
        ["GAM (Set A Eksogen Murni)", f"{gbm_pkg['engine']} (Set B + Lag Mingguan)"],
    )
    c_m, c_d = st.columns(2)
    month = c_m.selectbox("Bulan", list(range(1, 13)), index=3, format_func=lambda m: month_names[m])
    day_of_week = c_d.selectbox("Hari", list(range(7)), index=2, format_func=lambda d: dow_names[d])
    hour_focus = st.slider("Jam Fokus (00:00–23:00)", min_value=0, max_value=23, value=14)

    holiday_id = st.selectbox("Kategori Hari Libur (Holiday_ID)", list(range(23)), index=0, format_func=lambda h: HOLIDAY_LABELS[h])
    c_sch, c_cov = st.columns(2)
    school = 1 if c_sch.toggle("Masa Sekolah (school=1)", value=True) else 0
    covid_active = c_cov.toggle("Rezim COVID-19 Aktif", value=False)

    lag_regime = "Minggu Normal (Mei 2019)"
    if not model_choice.startswith("GAM"):
        lag_regime = st.selectbox(
            "Referensi Lag Beban Mingguan (Set B)",
            ["Minggu Normal (Mei 2019)", "Lockdown COVID-19 (Apr 2020)"],
            index=1 if covid_active else 0,
        )

    st.markdown("**Meteorologi 3 Kota**")
    sync_cities = st.checkbox("Sinkronkan Santiago & David terhadap Tocumen", value=True)
    t2m_toc = st.slider("Suhu Tocumen T2M_toc (°C)", min_value=22.0, max_value=36.0, value=29.5, step=0.25)
    if sync_cities:
        t2m_san = float(np.clip(t2m_toc - 0.5, 20.0, 39.0))
        t2m_dav = float(np.clip(t2m_toc - 2.7, 19.5, 35.0))
        st.caption(f"Tersinkron otomatis → Santiago: {t2m_san:.2f}°C · David: {t2m_dav:.2f}°C")
    else:
        t2m_san = st.slider("Suhu Santiago T2M_san (°C)", min_value=20.0, max_value=39.0, value=29.0, step=0.25)
        t2m_dav = st.slider("Suhu David T2M_dav (°C)", min_value=19.5, max_value=35.0, value=26.8, step=0.25)

    qv2m_toc = st.slider("Kelembapan Spesifik QV2M_toc (kg/kg)", min_value=0.012, max_value=0.023, value=0.0185, step=0.0005, format="%.4f")

sim_df, interval_label = simulate_whatif_24h(
    model_choice=model_choice,
    month=month,
    day_of_week=day_of_week,
    holiday_id=holiday_id,
    school=school,
    t2m_toc=t2m_toc,
    t2m_san=t2m_san,
    t2m_dav=t2m_dav,
    qv2m_toc=qv2m_toc,
    hour_focus=hour_focus,
    covid_active=covid_active,
    lag_regime=lag_regime,
    gam_pre_pkg=gam_pre,
    gam_cov_pkg=gam_cov,
    gbm_pkg=gbm_pkg,
    preproc=preproc,
)

focus_row = sim_df[sim_df["hour"] == float(hour_focus)].iloc[0]
baseline_h = audit["baseline_hourly_2019"]
baseline_int = {int(k): float(v) for k, v in baseline_h.items()}
base_val = float(baseline_int.get(int(hour_focus), 1250.0))
delta_base = float(focus_row["pred_mean"]) - base_val
delta_base_pct = delta_base / base_val * 100.0
daily_energy_mwh = float(sim_df["pred_mean"].sum())

with col_out:
    render_shadcn_kpis([
        {
            "label": f"Prediksi Jam {int(hour_focus):02d}:00",
            "value": f"{focus_row['pred_mean']:,.2f} MWh",
            "description": f"{dow_names[day_of_week]}, {month_names[month]}",
        },
        {
            "label": interval_label,
            "value": f"{focus_row['pred_lo']:,.0f}–{focus_row['pred_hi']:,.0f} MWh",
            "description": f"Lebar interval {focus_row['pred_hi'] - focus_row['pred_lo']:,.1f} MWh",
        },
        {
            "label": "Delta vs Normal 2019",
            "value": f"{delta_base:+.2f} MWh",
            "description": f"Baseline {base_val:,.1f} MWh",
            "delta": f"{delta_base_pct:+.2f}%",
        },
        {
            "label": "Total Energi Harian (24j)",
            "value": f"{daily_energy_mwh / 1000.0:.2f} GWh",
            "description": f"Peak {sim_df['pred_mean'].max():,.1f} · Min {sim_df['pred_mean'].min():,.1f} MWh",
        },
    ], key_prefix="wi_kpi")

    st.plotly_chart(
        make_whatif_24h_chart(sim_df, interval_label=interval_label, baseline_hourly_2019=baseline_int, hour_focus=hour_focus),
        width="stretch", theme=None, config={"displaylogo": False},
    )

    fig_comp = plot_mpl_whatif_delta_and_temp(sim_df, baseline_int, hour_focus=hour_focus)
    st.pyplot(fig_comp)
    plt.close(fig_comp)

    sim_disp = sim_df.copy()
    sim_disp["hour"] = sim_disp["hour"].astype(int).map(lambda h: f"{h:02d}:00")
    for c in ["T2M_toc", "T2M_san", "T2M_dav", "pred_mean", "pred_lo", "pred_hi"]:
        sim_disp[c] = sim_disp[c].round(2)
    st.dataframe(sim_disp, width="stretch", hide_index=True)

    st.download_button(
        "Unduh Hasil Simulasi What-If 24 Jam (CSV)",
        data=sim_disp.to_csv(index=False).encode("utf-8"),
        file_name="panama_simulasi_whatif_24h.csv",
        mime="text/csv",
    )

render_methodology_drawer(
    "Cara Kerja Mesin Inferensi Simulator What-If",
    """
- **GAM (Set A Eksogen):** Memprediksi langsung dari fitur kalender, suhu terproyeksi PCA (`T2M_pca1`), kelembapan `QV2M_toc`, dan status rezim `is_covid_regime` tanpa memerlukan *lag* beban historis. Ideal untuk uji sensitivitas murni cuaca & hari libur.
- **HistGradientBoosting (Set B):** Menggabungkan kurva suhu 24 jam yang berpusat pada suhu pilihan pengguna dengan *lag* beban historis (`t-168`, `t-336`, `t-504`) dari minggu referensi yang dipilih (`Mei 2019` normal vs `April 2020` lockdown).
""",
)
