import matplotlib.pyplot as plt
import numpy as np
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
from utils.ui_components import inject_styles, render_insight_box, render_neu_kpis, render_page_header

inject_styles()

audit = load_audit_stats()
gam_pre = load_gam_pre()
gam_cov = load_gam_covid()
gbm_pkg = load_gbm_full()
preproc = load_preprocessors()

render_page_header(
    "Simulasi Kondisi Listrik (Coba Sendiri)",
    "Uji bagaimana pemakaian listrik berubah sepanjang 24 jam jika cuaca tiba-tiba panas terik, bertepatan dengan hari besar nasional, atau saat ada pembatasan aktivitas.",
    kicker="Simulasi What-If",
)

PRESETS = {
    "Hari Kerja Normal": {
        "month": 5, "dow": 2, "temp": 29.5, "hum": "Normal Tropis", "hol": 0, "school": True, "covid": False,
    },
    "Cuaca Panas Terik": {
        "month": 4, "dow": 2, "temp": 34.5, "hum": "Sangat Lembap (Gerah)", "hol": 0, "school": True, "covid": False,
    },
    "Akhir Pekan Santai": {
        "month": 5, "dow": 6, "temp": 29.5, "hum": "Normal Tropis", "hol": 0, "school": True, "covid": False,
    },
    "Libur Nasional (Hari Natal)": {
        "month": 12, "dow": 4, "temp": 28.5, "hum": "Normal Tropis", "hol": 21, "school": False, "covid": False,
    },
    "Pembatasan / Pandemi": {
        "month": 4, "dow": 2, "temp": 30.0, "hum": "Normal Tropis", "hol": 0, "school": False, "covid": True,
    },
}

HUMIDITY_MAP = {
    "Kering (Nyaman)": 0.0140,
    "Normal Tropis": 0.0185,
    "Sangat Lembap (Gerah)": 0.0215,
}

with st.container(border=True):
    preset = st.radio(
        "Pilih Contoh Skenario Cepat (lalu ubah pengatur di bawah sesuai kebutuhan):",
        list(PRESETS.keys()),
        horizontal=True,
        key="sim_preset",
    )

# Populate widget keys BEFORE instantiating widgets when preset changes or on first load
if "last_preset" not in st.session_state or st.session_state["last_preset"] != preset:
    cfg = PRESETS[preset]
    st.session_state["sim_month"] = cfg["month"]
    st.session_state["sim_dow"] = cfg["dow"]
    st.session_state["sim_temp"] = cfg["temp"]
    st.session_state["sim_hum"] = cfg["hum"]
    st.session_state["sim_hol"] = cfg["hol"]
    st.session_state["sim_school"] = cfg["school"]
    st.session_state["sim_covid"] = cfg["covid"]
    st.session_state["last_preset"] = preset

month_names = {1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November", 12: "Desember"}
dow_names = {0: "Senin", 1: "Selasa", 2: "Rabu", 3: "Kamis", 4: "Jumat", 5: "Sabtu", 6: "Minggu"}

col_ctrl, col_out = st.columns([4.2, 7.8], gap="large")

with col_ctrl:
    with st.container(border=True):
        st.markdown("**Pengaturan Skenario**")
        c_m, c_d = st.columns(2)
        month = c_m.selectbox("Bulan", list(range(1, 13)), format_func=lambda m: month_names[m], key="sim_month")
        day_of_week = c_d.selectbox("Hari", list(range(7)), format_func=lambda d: dow_names[d], key="sim_dow")

        hour_focus = st.slider("Jam yang Ingin Dilihat Detailnya", min_value=0, max_value=23, value=14, format="%02d:00")
        t2m_toc = st.slider("Suhu Siang Kota Panama (°C)", min_value=22.0, max_value=36.0, step=0.5, key="sim_temp")
        hum_label = st.radio("Kondisi Kelembapan Udara", list(HUMIDITY_MAP.keys()), horizontal=True, key="sim_hum")
        qv2m_toc = HUMIDITY_MAP[hum_label]

        holiday_id = st.selectbox(
            "Status Hari Libur Nasional",
            list(range(23)),
            format_func=lambda h: HOLIDAY_LABELS[h].split(" - ", 1)[-1],
            key="sim_hol",
        )
        c_sch, c_cov = st.columns(2)
        school = 1 if c_sch.toggle("Kegiatan Sekolah Aktif", key="sim_school") else 0
        covid_active = c_cov.toggle("Pembatasan Pandemi Aktif", key="sim_covid")

        with st.expander("Pengaturan Lanjutan (Model & Suhu Kota Lain)"):
            model_choice = st.radio(
                "Mesin Prediksi",
                ["GAM (Respons Murni Cuaca & Kalender)", f"{gbm_pkg['engine']} (Dengan Pola Minggu Lalu)"],
            )
            lag_regime = "Minggu Normal (Mei 2019)"
            if not model_choice.startswith("GAM"):
                lag_regime = st.selectbox(
                    "Patokan Pola Minggu Lalu",
                    ["Minggu Normal (Mei 2019)", "Lockdown COVID-19 (Apr 2020)"],
                    index=1 if covid_active else 0,
                )
            sync_cities = st.checkbox("Sesuaikan Suhu Santiago & David Otomatis", value=True)
            if sync_cities:
                t2m_san = float(np.clip(t2m_toc - 0.5, 20.0, 39.0))
                t2m_dav = float(np.clip(t2m_toc - 2.7, 19.5, 35.0))
                st.caption(f"Santiago: {t2m_san:.1f}°C · David: {t2m_dav:.1f}°C")
            else:
                t2m_san = st.slider("Suhu Santiago (°C)", min_value=20.0, max_value=39.0, value=29.0, step=0.5)
                t2m_dav = st.slider("Suhu David (°C)", min_value=19.5, max_value=35.0, value=26.8, step=0.5)

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
daily_energy_gwh = float(sim_df["pred_mean"].sum()) / 1000.0

with col_out:
    tone_summary = "rose" if delta_base_pct < -5.0 else ("amber" if delta_base_pct > 5.0 else "teal")
    arah = "lebih tinggi" if delta_base >= 0 else "lebih rendah"
    render_insight_box(
        f"Ringkasan Hasil Skenario ({preset})",
        f"Pada kondisi ini, perkiraan kebutuhan listrik pukul {int(hour_focus):02d}:00 mencapai {focus_row['pred_mean']:,.0f} MWh ({abs(delta_base_pct):.1f}% {arah} dari rata-rata hari normal 2019 sebesar {base_val:,.0f} MWh).",
        tone=tone_summary,
    )

    render_neu_kpis([
        {
            "label": f"Perkiraan Pukul {int(hour_focus):02d}:00",
            "value": f"{focus_row['pred_mean']:,.0f} MWh",
            "hint": f"{dow_names[day_of_week]}, {month_names[month]} (rentang {focus_row['pred_lo']:,.0f}–{focus_row['pred_hi']:,.0f})",
            "badge": "Titik Fokus",
            "tone": "blue",
        },
        {
            "label": "Selisih vs Hari Normal 2019",
            "value": f"{delta_base:+,.0f} MWh",
            "hint": f"Patokan normal jam {int(hour_focus):02d}:00 = {base_val:,.0f} MWh",
            "badge": f"{delta_base_pct:+.1f}%",
            "tone": tone_summary,
        },
        {
            "label": "Total Listrik Sehari Penuh",
            "value": f"{daily_energy_gwh:.2f} GWh",
            "hint": f"Tertinggi {sim_df['pred_mean'].max():,.0f} MWh · Terendah {sim_df['pred_mean'].min():,.0f} MWh",
            "badge": "Total 24 Jam",
            "tone": "teal",
        },
    ])

    with st.container(border=True):
        st.plotly_chart(
            make_whatif_24h_chart(sim_df, interval_label=interval_label, baseline_hourly_2019=baseline_int, hour_focus=hour_focus),
            width="stretch",
            theme=None,
            config={"displaylogo": False},
        )

    with st.expander("Lihat Grafik Selisih per Jam, Suhu 3 Kota, & Unduh Data 24 Jam (CSV)"):
        fig_comp = plot_mpl_whatif_delta_and_temp(sim_df, baseline_int, hour_focus=hour_focus)
        st.pyplot(fig_comp)
        plt.close(fig_comp)

        sim_disp = sim_df.copy()
        sim_disp["hour"] = sim_disp["hour"].astype(int).map(lambda h: f"{h:02d}:00")
        for c in ["T2M_toc", "T2M_san", "T2M_dav", "pred_mean", "pred_lo", "pred_hi"]:
            sim_disp[c] = sim_disp[c].round(2)
        st.dataframe(sim_disp, width="stretch", hide_index=True)

        st.download_button(
            "Unduh Hasil Simulasi 24 Jam (CSV)",
            data=sim_disp.to_csv(index=False).encode("utf-8"),
            file_name="panama_simulasi_whatif_24h.csv",
            mime="text/csv",
        )
