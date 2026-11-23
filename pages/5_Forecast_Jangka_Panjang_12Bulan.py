import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from utils.charts import make_lt_forecast_chart, plot_mpl_lt_energy_bars
from utils.data_loader import (
    load_audit_stats,
    load_forecast_lt,
    load_gam_covid,
    load_gam_pre,
    load_master_hourly,
    load_preprocessors,
)
from utils.inference import compute_custom_lt_forecast
from utils.ui_components import render_methodology_drawer, render_page_header, render_shadcn_kpis

df = load_master_hourly()
lt_df = load_forecast_lt()
audit = load_audit_stats()
gam_pre = load_gam_pre()
gam_cov = load_gam_covid()
preproc = load_preprocessors()

lt_sum = audit["lt_summary"]
lt_monthly = pd.DataFrame(audit["lt_monthly"])

render_page_header(
    "Forecast Jangka Panjang (12 Bulan: Jul 2020 – Jun 2021)",
    "Proyeksi eksogen murni 8.760 jam ke depan menggunakan GAM Set A pada skenario pemulihan normal, tekanan COVID-19 berkelanjutan, dan simulasi kustom.",
)

c1, c2, c3, c4 = st.columns([1.8, 1.8, 2.8, 2.6])
show_norm = c1.checkbox("Skenario A: Pemulihan Normal", value=True)
show_cov = c2.checkbox("Skenario B: Tekanan COVID-19", value=True)
recovery_pct = c3.slider("Simulasi Pemulihan Ekonomi/Mobilitas (%)", min_value=0, max_value=100, value=50, step=5)
delta_t2m = c4.slider("Anomali Suhu Regional ΔT2M (°C)", min_value=-2.0, max_value=3.0, value=0.0, step=0.25)

custom_hourly, custom_monthly, custom_sum = compute_custom_lt_forecast(
    lt_df, gam_pre, gam_cov, preproc, recovery_pct=recovery_pct, delta_t2m=delta_t2m
)

render_shadcn_kpis([
    {
        "label": "Skenario A (Pemulihan Normal)",
        "value": f"{lt_sum['normal_total_twh']:.3f} TWh",
        "description": f"{lt_sum['normal_total_gwh']:,.2f} GWh · Mean {lt_sum['normal_mean_mwh']:,.2f} MWh",
    },
    {
        "label": "Skenario B (Tekanan COVID-19)",
        "value": f"{lt_sum['covid_total_twh']:.3f} TWh",
        "description": f"{lt_sum['covid_total_gwh']:,.2f} GWh (Koef {lt_sum['cov_coef_mwh']:+.2f} MWh)",
        "delta": f"{lt_sum['diff_pct']:+.2f}% vs Normal",
    },
    {
        "label": "Skenario Kustom (Slider)",
        "value": f"{custom_sum['total_twh']:.3f} TWh",
        "description": f"Pemulihan {recovery_pct}% · ΔT2M {delta_t2m:+.2f}°C ({custom_sum['mean_mwh']:,.1f} MWh)",
    },
    {
        "label": "Referensi Aktual 2019",
        "value": f"{lt_sum['ref_2019_total_gwh'] / 1000.0:.3f} TWh",
        "description": f"{lt_sum['ref_2019_total_gwh']:,.2f} GWh · Mean {lt_sum['ref_2019_mean_mwh']:,.2f} MWh",
    },
], key_prefix="lt_kpi")

st.divider()

hist_m = df.set_index("datetime")["nat_demand"].resample("MS").mean().reset_index()
lt_m = lt_df.set_index("datetime")[["lt_base_mean", "lt_base_lo", "lt_base_hi", "lt_cov_mean", "lt_cov_lo", "lt_cov_hi"]].resample("MS").mean().reset_index()

st.plotly_chart(
    make_lt_forecast_chart(hist_m, lt_m, show_normal=show_norm, show_covid=show_cov, custom_monthly=custom_monthly),
    width="stretch", theme=None, config={"displaylogo": False},
)

fig_lt_bars = plot_mpl_lt_energy_bars(audit["lt_monthly"], custom_monthly=custom_monthly)
st.pyplot(fig_lt_bars)
plt.close(fig_lt_bars)

st.subheader("Tabel Proyeksi Bulanan (Juli 2020 – Juni 2021)")
merged_lt = lt_monthly[["Bulan", "Jam", "Ref_2019_GWh", "Normal_Mean_MWh", "Normal_CI95_MWh", "Normal_Energi_GWh", "COVID_Mean_MWh", "COVID_Energi_GWh", "Selisih_GWh"]].merge(
    custom_monthly, on="Bulan", how="left"
)
st.dataframe(merged_lt, width="stretch", hide_index=True)

st.download_button(
    "Unduh Ringkasan Proyeksi 12 Bulan + Skenario Kustom (CSV)",
    data=merged_lt.to_csv(index=False).encode("utf-8"),
    file_name="panama_forecast_jangka_panjang_12m.csv",
    mime="text/csv",
)

render_methodology_drawer(
    "Metodologi Proyeksi 12 Bulan & Asumsi Skenario (Tahap 8B)",
    """
- **Mengapa GAM Set A?** Pada horizon `12` bulan (`8.760` jam), *lag* beban target tidak tersedia di masa depan. GAM Set A menggunakan tren makro (`time_index_years`), siklus harian/tahunan, kalender penuh dari templat tahun normal `2019`, dan klimatologi cuaca `2015–2019`.
- **Skenario A vs B:** Skenario A memakai `res_gam_pre` (`Jan 2015 – Feb 2020`), sedangkan Skenario B memakai `res_gam_covid` (`Jan 2015 – Jun 2020`) dengan indikator rezim `is_covid_regime` (`-177,44` MWh, $p < 10^{-16}$).
- **Simulasi Kustom:** Menggeser suhu ketiga kota sebesar $\\Delta T2M$, menghitung ulang proyeksi PCA suhu `T2M_pca1` dan interaksinya, mengevaluasi kedua model GAM, lalu memadukan bobot pemulihan ekonomi (`0–100%`).
""",
)
