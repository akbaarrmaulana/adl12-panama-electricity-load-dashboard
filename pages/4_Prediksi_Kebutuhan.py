import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from utils.charts import (
    make_lt_forecast_chart,
    make_st_forecast_chart,
    plot_mpl_lt_energy_bars,
    plot_mpl_st_daily_bars,
)
from utils.data_loader import (
    load_audit_stats,
    load_forecast_lt,
    load_forecast_st,
    load_gam_covid,
    load_gam_pre,
    load_master_hourly,
    load_preprocessors,
)
from utils.inference import compute_custom_lt_forecast
from utils.ui_components import inject_styles, render_insight_box, render_neu_kpis, render_page_header

inject_styles()

df = load_master_hourly()
future_st = load_forecast_st()
lt_df = load_forecast_lt()
audit = load_audit_stats()
gam_pre = load_gam_pre()
gam_cov = load_gam_covid()
preproc = load_preprocessors()

st_sum = audit["st_summary"]
st_daily = pd.DataFrame(audit["st_daily"])
lt_sum = audit["lt_summary"]
lt_monthly = pd.DataFrame(audit["lt_monthly"])

render_page_header(
    "Prediksi Kebutuhan Listrik ke Depan",
    "Proyeksi operasional jangka pendek untuk 7 hari ke depan (Boosting) dan proyeksi perencanaan 12 bulan ke depan (GAM).",
    kicker="Proyeksi Masa Depan",
)

tab1, tab2 = st.tabs([
    "Jangka Pendek: 7 Hari ke Depan (27 Jun – 4 Jul 2020)",
    "Jangka Panjang: 12 Bulan ke Depan (Jul 2020 – Jun 2021)",
])

with tab1:
    render_neu_kpis([
        {
            "label": "Rata-rata Kebutuhan",
            "value": f"{st_sum['mean_mwh']:,.0f} MWh/jam",
            "hint": "Selama periode 27 Juni – 4 Juli 2020 (168 jam)",
            "badge": "7 Hari",
            "tone": "blue",
        },
        {
            "label": "Perkiraan Puncak Tertinggi",
            "value": f"{st_sum['max_mwh']:,.0f} MWh",
            "hint": f"Terendah malam hari {st_sum['min_mwh']:,.0f} MWh",
            "badge": "Beban Puncak",
            "tone": "amber",
        },
        {
            "label": "Total Energi Seminggu",
            "value": f"{st_sum['total_gwh']:.2f} GWh",
            "hint": f"Lebar rentang 90% rata-rata {st_sum['mean_pi_width']:.1f} MWh",
            "badge": "Akumulasi 7H",
            "tone": "teal",
        },
    ])

    with st.container(border=True):
        c1, c2, c3 = st.columns([3.0, 1.8, 1.8])
        threshold_mwh = c1.slider("Batas Kapasitas Siaga (MWh)", min_value=1050, max_value=1450, value=1250, step=10)
        show_pi = c2.checkbox("Tampilkan Rentang Kemungkinan 90%", value=True)
        show_q50 = c3.checkbox("Tampilkan Garis Median (Q50)", value=False)

        exceed_df = future_st[future_st["pred_st_mean"] > threshold_mwh]
        n_exceed = len(exceed_df)
        if n_exceed == 0:
            render_insight_box(
                "Cadangan Kapasitas Aman",
                f"Seluruh 168 jam ke depan diperkirakan berada di bawah batas siaga {threshold_mwh:,.0f} MWh (puncak tertinggi {st_sum['max_mwh']:,.2f} MWh).",
                tone="teal",
            )
        else:
            render_insight_box(
                f"Perhatian: {n_exceed} Jam Melebihi Batas Siaga {threshold_mwh:,.0f} MWh",
                f"Beban puncak diperkirakan mencapai {st_sum['max_mwh']:,.2f} MWh (+{st_sum['max_mwh'] - threshold_mwh:,.1f} MWh di atas batas siaga Anda, terutama saat siang hari kerja).",
                tone="rose",
            )

        hist_7d = df.iloc[-168:].copy()
        st.plotly_chart(
            make_st_forecast_chart(hist_7d, future_st, threshold_mwh=threshold_mwh, show_pi=show_pi, show_q50=show_q50),
            width="stretch",
            theme=None,
            config={"displaylogo": False},
        )

    with st.container(border=True):
        c_bars, c_tbl = st.columns([6.5, 5.5], gap="large")
        with c_bars:
            fig_st_bars = plot_mpl_st_daily_bars(st_daily)
            st.pyplot(fig_st_bars)
            plt.close(fig_st_bars)
        with c_tbl:
            st.markdown("**Ringkasan per Hari (27 Jun – 4 Jul 2020)**")
            daily_simple = st_daily[["Tanggal", "Hari", "Mean_MWh", "Peak_MWh", "Total_GWh"]].rename(
                columns={
                    "Mean_MWh": "Rata-rata (MWh)",
                    "Peak_MWh": "Puncak (MWh)",
                    "Total_GWh": "Total (GWh)",
                }
            )
            st.dataframe(daily_simple, width="stretch", hide_index=True)

    with st.expander("Lihat & Unduh Tabel Rincian Per Jam (168 Jam)"):
        date_opts = ["Semua 7 Hari (168 Jam)"] + st_daily["Tanggal"].tolist()
        date_filter = st.selectbox("Filter Tanggal", date_opts)
        tbl_st = future_st[["datetime", "Tanggal", "Hari", "pred_st_mean", "pred_st_q05", "pred_st_q50", "pred_st_q95", "pi_width_90"]].copy()
        if date_filter != "Semua 7 Hari (168 Jam)":
            tbl_st = tbl_st[tbl_st["Tanggal"] == date_filter]
        for c in ["pred_st_mean", "pred_st_q05", "pred_st_q50", "pred_st_q95", "pi_width_90"]:
            tbl_st[c] = tbl_st[c].round(2)
        st.dataframe(tbl_st, width="stretch", hide_index=True)
        st.download_button(
            "Unduh Forecast Jangka Pendek 168 Jam (CSV)",
            data=future_st.to_csv(index=False).encode("utf-8"),
            file_name="panama_forecast_jangka_pendek_168h.csv",
            mime="text/csv",
        )
        st.caption(f"Catatan Teknis: Menggunakan {audit['dataset_meta']['gbm_engine']} Set B yang dilatih penuh pada {st_sum['n_train_rows']:,} jam bersih dengan cuaca klimatologi 2015–2019.")

with tab2:
    render_insight_box(
        "Dua Skenario Masa Depan + Simulasi Anda",
        f"Karena pemulihan ekonomi pasca-pandemi belum pasti, prediksi 1 tahun disajikan antara Skenario Pulih Normal ({lt_sum['normal_total_twh']:.2f} TWh) hingga Skenario Pandemi Berlanjut ({lt_sum['covid_total_twh']:.2f} TWh). Geser pengatur di bawah untuk menguji skenario Anda sendiri.",
        tone="blue",
    )

    with st.container(border=True):
        s1, s2, s3, s4 = st.columns([2.8, 2.6, 1.6, 1.6])
        recovery_pct = s1.slider("Asumsi Pemulihan Aktivitas Ekonomi (%)", min_value=0, max_value=100, value=50, step=5)
        delta_t2m = s2.slider("Perubahan Suhu Rata-rata Tahunan (°C)", min_value=-2.0, max_value=3.0, value=0.0, step=0.25)
        show_norm = s3.checkbox("Tampilkan Garis Normal", value=True)
        show_cov = s4.checkbox("Tampilkan Garis COVID-19", value=True)

        _, custom_monthly, custom_sum = compute_custom_lt_forecast(
            lt_df, gam_pre, gam_cov, preproc, recovery_pct=recovery_pct, delta_t2m=delta_t2m
        )

        render_neu_kpis([
            {
                "label": "Skenario Pulih Normal",
                "value": f"{lt_sum['normal_total_twh']:.3f} TWh",
                "hint": f"{lt_sum['normal_total_gwh']:,.1f} GWh · Rata-rata {lt_sum['normal_mean_mwh']:,.0f} MWh/jam",
                "badge": "Kondisi Normal",
                "tone": "blue",
            },
            {
                "label": "Skenario Pandemi Berlanjut",
                "value": f"{lt_sum['covid_total_twh']:.3f} TWh",
                "hint": f"{lt_sum['covid_total_gwh']:,.1f} GWh ({lt_sum['diff_pct']:+.2f}% vs Normal)",
                "badge": "Tekanan COVID-19",
                "tone": "rose",
            },
            {
                "label": "Hasil Simulasi Pilihan Anda",
                "value": f"{custom_sum['total_twh']:.3f} TWh",
                "hint": f"Pemulihan {recovery_pct}% · Suhu {delta_t2m:+.2f}°C ({custom_sum['mean_mwh']:,.0f} MWh/jam)",
                "badge": "Kustom Anda",
                "tone": "teal",
            },
        ])

        hist_m = df.set_index("datetime")["nat_demand"].resample("MS").mean().reset_index()
        lt_m = lt_df.set_index("datetime")[["lt_base_mean", "lt_base_lo", "lt_base_hi", "lt_cov_mean", "lt_cov_lo", "lt_cov_hi"]].resample("MS").mean().reset_index()
        st.plotly_chart(
            make_lt_forecast_chart(hist_m, lt_m, show_normal=show_norm, show_covid=show_cov, custom_monthly=custom_monthly),
            width="stretch",
            theme=None,
            config={"displaylogo": False},
        )

    with st.container(border=True):
        fig_lt_bars = plot_mpl_lt_energy_bars(audit["lt_monthly"], custom_monthly=custom_monthly)
        st.pyplot(fig_lt_bars)
        plt.close(fig_lt_bars)

    with st.expander("Lihat & Unduh Tabel Proyeksi Bulanan (12 Bulan)"):
        merged_lt = lt_monthly[
            ["Bulan", "Jam", "Ref_2019_GWh", "Normal_Mean_MWh", "Normal_CI95_MWh", "Normal_Energi_GWh", "COVID_Mean_MWh", "COVID_Energi_GWh", "Selisih_GWh"]
        ].merge(custom_monthly, on="Bulan", how="left")
        st.dataframe(merged_lt, width="stretch", hide_index=True)
        st.download_button(
            "Unduh Proyeksi 12 Bulan + Skenario Kustom (CSV)",
            data=merged_lt.to_csv(index=False).encode("utf-8"),
            file_name="panama_forecast_jangka_panjang_12m.csv",
            mime="text/csv",
        )
        st.caption("Catatan Teknis: Menggunakan GAM Set A eksogen murni (tanpa lag beban) dengan proyeksi PCA suhu 3 kota dan bobot pemulihan ekonomi.")
