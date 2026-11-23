import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from utils.charts import (
    make_dual_axis_explorer,
    plot_mpl_covid_counterfactual_2panel,
    plot_mpl_fft_and_acf_stem,
    plot_mpl_intraday_profiles_and_heatmap,
    plot_mpl_vif_bars,
    plot_mpl_weather_and_holiday,
    plot_mpl_xai_importance,
)
from utils.data_loader import load_audit_stats, load_master_hourly
from utils.inference import HOLIDAY_LABELS
from utils.ui_components import inject_styles, render_insight_box, render_neu_kpis, render_page_header

inject_styles()

df = load_master_hourly()
audit = load_audit_stats()

render_page_header(
    "Pola Pemakaian & Faktor Pemicu Listrik",
    "Kenali jam-jam paling sibuk, pengaruh suhu panas tropis & hari libur nasional, hingga dampak nyata pembatasan COVID-19 pada 2020.",
    kicker="Pola & Faktor Utama",
)

het = audit["heteroskedasticity"]
hol_diff_pct = (het["holiday1_mean"] - het["holiday0_mean"]) / het["holiday0_mean"] * 100.0
pl = audit["placebo_2019"]
cov = audit["covid_summary"]
cf_tbl = pd.DataFrame(audit["covid_monthly_cf"])
apr_row = cf_tbl[cf_tbl["month"] == 4].iloc[0]

tab1, tab2, tab3 = st.tabs(["Pola Jam & Hari", "Cuaca & Hari Libur", "Dampak Pandemi COVID-19"])

with tab1:
    render_insight_box(
        "Pola Harian yang Konsisten",
        "Listrik Panama paling tinggi pada pukul 11:00–15:00 di hari Senin–Jumat karena perkantoran dan pendingin ruangan (AC), lebih rendah pada Sabtu–Minggu, dan turun tajam di siang hari selama pandemi 2020.",
        tone="blue",
    )
    with st.container(border=True):
        period_opt = st.radio(
            "Pilih Periode Peta Panas",
            ["Semua Tahun (2015–2020)", "Tahun Normal (2019)", "Masa Pandemi (2020)"],
            horizontal=True,
        )
        if period_opt == "Tahun Normal (2019)":
            df_sub = df[df["year"] == 2019]
        elif period_opt == "Masa Pandemi (2020)":
            df_sub = df[df["year"] == 2020]
        else:
            df_sub = df
        fig_intra = plot_mpl_intraday_profiles_and_heatmap(df, df_sub)
        st.pyplot(fig_intra)
        plt.close(fig_intra)

with tab2:
    render_neu_kpis([
        {
            "label": "Suhu Rata-rata Kota Panama",
            "value": f"{df['T2M_toc'].mean():.1f} °C",
            "hint": "Iklim tropis panas sepanjang tahun",
            "badge": "Tocumen",
            "tone": "amber",
        },
        {
            "label": "Rata-rata Hari Biasa",
            "value": f"{het['holiday0_mean']:,.1f} MWh",
            "hint": "Saat aktivitas sekolah & kantor normal",
            "badge": "Hari Aktif",
            "tone": "blue",
        },
        {
            "label": "Rata-rata Hari Libur Nasional",
            "value": f"{het['holiday1_mean']:,.1f} MWh",
            "hint": f"Lebih rendah {abs(hol_diff_pct):.1f}% dibanding hari biasa",
            "badge": f"{hol_diff_pct:+.1f}%",
            "tone": "teal",
        },
    ])

    render_insight_box(
        "Mengapa Suhu & Hari Libur Sangat Berpengaruh?",
        "Begitu suhu udara melewati ~27°C, beban listrik melonjak tajam karena penggunaan AC. Sebaliknya, hari besar nasional seperti Jumat Agung dan Hari Natal menurunkan pemakaian listrik lebih dari 150 MWh.",
        tone="amber",
    )

    with st.container(border=True):
        c1, c2, c3 = st.columns([1.4, 1.4, 1.2])
        city_map = {
            "Kota Panama (Tocumen)": ("toc", "Kota Panama"),
            "Santiago": ("san", "Santiago"),
            "David": ("dav", "David"),
        }
        var_map = {
            "Suhu Udara (°C)": "T2M",
            "Kelembapan Udara": "QV2M",
            "Curah Hujan": "TQL",
            "Kecepatan Angin (m/s)": "W2M",
        }
        city_choice = c1.selectbox("Kota Pengamatan", list(city_map.keys()))
        var_choice = c2.selectbox("Faktor Cuaca", list(var_map.keys()))
        res_choice = c3.selectbox("Tampilan Waktu", ["Rata-rata Mingguan", "Rata-rata Harian"], index=0)

        c_suf, c_lbl = city_map[city_choice]
        w_pre = var_map[var_choice]
        w_col = f"{w_pre}_{c_suf}"
        rule = "W" if res_choice == "Rata-rata Mingguan" else "D"
        df_plot = (
            df.set_index("datetime")[["nat_demand", w_col, "is_blackout"]]
            .resample(rule)
            .agg({"nat_demand": "mean", w_col: "mean", "is_blackout": "max"})
            .reset_index()
        )
        st.plotly_chart(
            make_dual_axis_explorer(df_plot, w_col, f"{var_choice} ({c_lbl})", show_blackout=True, show_lockdown=True),
            width="stretch",
            theme=None,
            config={"displaylogo": False},
        )

    with st.container(border=True):
        top_n_hol = st.slider("Jumlah Hari Libur yang Ditampilkan", min_value=6, max_value=22, value=10)
        fig_wh = plot_mpl_weather_and_holiday(audit["gam_partial_curves"], audit["xai_gbm"]["pdp_t2m_toc"], HOLIDAY_LABELS, top_n=top_n_hol)
        st.pyplot(fig_wh)
        plt.close(fig_wh)

with tab3:
    render_neu_kpis([
        {
            "label": "Total Listrik Berkurang (Mar–Jun 2020)",
            "value": f"{cov['total_lost_gwh']:,.1f} GWh",
            "hint": f"Rata-rata turun {abs(cov['mean_delta_mwh']):,.1f} MWh per jam",
            "badge": f"{cov['mean_delta_pct']:+.2f}%",
            "tone": "rose",
        },
        {
            "label": "Penurunan Terdalam (April 2020)",
            "value": f"{apr_row['Delta_Persen']:+.2f}%",
            "hint": f"Selisih {apr_row['Delta_Mean_MWh']:+.1f} MWh/jam selama puncak lockdown",
            "badge": "Puncak Kontraksi",
            "tone": "rose",
        },
        {
            "label": "Uji Kewajaran Model (2019)",
            "value": f"{pl['bias_pct']:+.2f}%",
            "hint": f"Uji pada Mar–Jun 2019 hanya meleset {pl['bias_mwh']:+.2f} MWh (Placebo)",
            "badge": "Tervalidasi",
            "tone": "teal",
        },
    ])

    render_insight_box(
        "Cara Menghitung Dampak Pandemi",
        "Garis biru putus-putus menunjukkan perkiraan pemakaian listrik seandainya tidak ada COVID-19 (dihitung dari suhu & kalender 2020). Selisihnya dengan garis hitam aktual adalah penurunan nyata akibat pembatasan aktivitas.",
        tone="rose",
    )

    month_map = {1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni"}
    df_20 = df[df["year"] == 2020].copy()
    with st.container(border=True):
        c_m, c_d = st.columns([2, 2])
        month_focus = c_m.selectbox(
            "Bulan Fokus Profil 24 Jam",
            [1, 2, 3, 4, 5, 6],
            index=3,
            format_func=lambda m: f"{month_map[m]} 2020",
        )
        day_type = c_d.radio("Tipe Hari", ["Hari Kerja", "Akhir Pekan"], horizontal=True)
        fig_cf = plot_mpl_covid_counterfactual_2panel(df_20, df_20, month_focus=month_focus, day_type=day_type)
        st.pyplot(fig_cf)
        plt.close(fig_cf)

    with st.expander("Lihat Tabel Ringkasan Bulanan COVID-19 & Unduh CSV"):
        cf_disp = cf_tbl.copy()
        cf_disp["Bulan"] = cf_disp["month"].map(month_map)
        st.dataframe(
            cf_disp[["Bulan", "Aktual_Mean_MWh", "Counterfactual_Mean_MWh", "Delta_Mean_MWh", "Delta_Persen", "Total_Energi_Hilang_GWh"]],
            width="stretch",
            hide_index=True,
        )
        export_cf = df_20[["datetime", "month", "hour", "is_weekend", "nat_demand", "cf_mean", "cf_ci_lo", "cf_ci_hi"]].copy()
        export_cf["delta_mwh"] = export_cf["nat_demand"] - export_cf["cf_mean"]
        st.download_button(
            "Unduh Data Dampak COVID-19 Per Jam (CSV)",
            data=export_cf.to_csv(index=False).encode("utf-8"),
            file_name="panama_covid19_counterfactual_2020.csv",
            mime="text/csv",
        )

with st.expander("Lihat Audit Statistik Lanjutan (Spektrum FFT, ACF, VIF, Stasioneritas ADF/KPSS, XAI, & 8 Titik Blackout)"):
    fig_fft = plot_mpl_fft_and_acf_stem(audit["fft_spectrum"], audit["acf_demand"])
    st.pyplot(fig_fft)
    plt.close(fig_fft)

    c_vif, c_stat = st.columns([5.5, 6.5], gap="large")
    with c_vif:
        fig_vif = plot_mpl_vif_bars(audit["vif_table"])
        st.pyplot(fig_vif)
        plt.close(fig_vif)
    with c_stat:
        st.markdown("**Uji Stasioneritas ADF & KPSS (Pra-Pandemi 2015–2019)**")
        st.dataframe(pd.DataFrame(audit["stationarity"]), width="stretch", hide_index=True)
        st.caption(
            f"Breusch-Pagan LM = {het['bp_lm']:,.2f} (p = {het['bp_p']:.2e}) · "
            f"ARCH-LM(24) = {het['arch_lm']:,.2f} (p = {het['arch_p']:.2e})"
        )
        st.markdown("**8 Kejadian Gangguan Listrik / Blackout (< 600 MWh)**")
        bo_df = pd.DataFrame(audit["blackouts"])
        st.dataframe(bo_df, width="stretch", hide_index=True)
        st.download_button(
            "Unduh Data 8 Titik Blackout (CSV)",
            data=bo_df.to_csv(index=False).encode("utf-8"),
            file_name="panama_blackouts_8pts.csv",
            mime="text/csv",
        )

    fig_xai = plot_mpl_xai_importance(audit["xai_gbm"]["grouped"], audit["xai_gbm"]["top10"], audit["xai_gbm"]["method_label"])
    st.pyplot(fig_xai)
    plt.close(fig_xai)
