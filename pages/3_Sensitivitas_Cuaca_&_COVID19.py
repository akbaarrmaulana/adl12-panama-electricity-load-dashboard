import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from utils.charts import (
    plot_mpl_covid_counterfactual_2panel,
    plot_mpl_weather_and_holiday,
    plot_mpl_xai_importance,
)
from utils.data_loader import load_audit_stats, load_master_hourly
from utils.inference import HOLIDAY_LABELS
from utils.ui_components import render_methodology_drawer, render_page_header, render_shadcn_kpis

df = load_master_hourly()
audit = load_audit_stats()

render_page_header(
    "Sensitivitas Cuaca, Hari Libur, & Dampak Kausal COVID-19",
    "Analisis kurva non-linear GAM (Penalized B-Splines), uji placebo 2019, counterfactual 2020, efek 22 kategori Holiday_ID, serta interpretabilitas fitur Boosting.",
)

month_map = {1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni"}
f1, f2, f3, f4 = st.columns([3.0, 1.8, 1.8, 2.0])
selected_months = f1.multiselect(
    "Filter Bulan Counterfactual",
    options=[1, 2, 3, 4, 5, 6],
    default=[1, 2, 3, 4, 5, 6],
    format_func=lambda m: f"{month_map[m]} 2020",
)
month_focus = f2.selectbox("Bulan Profil 24 Jam", [1, 2, 3, 4, 5, 6], index=3, format_func=lambda m: f"{month_map[m]} 2020")
day_type = f3.radio("Tipe Hari", ["Hari Kerja", "Akhir Pekan"], horizontal=True)
top_n_hol = f4.slider("Top Kategori Holiday_ID", min_value=6, max_value=22, value=14)

pl = audit["placebo_2019"]
cov = audit["covid_summary"]
gam_meta = audit["gam_meta"]
cf_tbl = pd.DataFrame(audit["covid_monthly_cf"])
apr_row = cf_tbl[cf_tbl["month"] == 4].iloc[0]

render_shadcn_kpis([
    {
        "label": "Uji Placebo (Mar–Jun 2019)",
        "value": f"{pl['bias_mwh']:+.2f} MWh",
        "description": f"MAE {pl['mae_mwh']:.2f} MWh",
        "delta": f"{pl['bias_pct']:+.2f}% bias",
    },
    {
        "label": "Total Energi Hilang (Mar–Jun 2020)",
        "value": f"{cov['total_lost_gwh']:,.2f} GWh",
        "description": f"Rata-rata {cov['mean_delta_mwh']:+.2f} MWh/jam",
        "delta": f"{cov['mean_delta_pct']:+.2f}%",
    },
    {
        "label": "Kontraksi Puncak (April 2020)",
        "value": f"{apr_row['Delta_Persen']:+.2f}%",
        "description": f"{apr_row['Delta_Mean_MWh']:+.2f} MWh/jam · {apr_row['Total_Energi_Hilang_GWh']:.2f} GWh",
    },
    {
        "label": "Fit GAM (Set A Eksogen)",
        "value": f"{gam_meta['pseudo_r2']:.4f}",
        "description": f"EDF {gam_meta['edf']:.2f} · PCA suhu {gam_meta['pca_t2m_var_pct']:.2f}%",
    },
], key_prefix="cov_kpi")

st.divider()

df_20 = df[df["year"] == 2020].copy()
df_20_sub = df_20[df_20["month"].isin(selected_months)].copy() if selected_months else df_20.copy()

st.subheader("Estimasi Counterfactual & Pergeseran Profil Harian COVID-19")
fig_cf = plot_mpl_covid_counterfactual_2panel(df_20_sub, df_20, month_focus=month_focus, day_type=day_type)
st.pyplot(fig_cf)
plt.close(fig_cf)

st.subheader("Kurva Partial Effect GAM & Boosting PDP")
fig_wh = plot_mpl_weather_and_holiday(audit["gam_partial_curves"], audit["xai_gbm"]["pdp_t2m_toc"], HOLIDAY_LABELS, top_n=top_n_hol)
st.pyplot(fig_wh)
plt.close(fig_wh)

st.subheader("Kontribusi Kelompok Fitur & Top 10 Prediktor (XAI)")
fig_xai = plot_mpl_xai_importance(audit["xai_gbm"]["grouped"], audit["xai_gbm"]["top10"], audit["xai_gbm"]["method_label"])
st.pyplot(fig_xai)
plt.close(fig_xai)

st.subheader("Ringkasan Bulanan Kausal COVID-19 (Jan–Jun 2020)")
cf_disp = cf_tbl.copy()
cf_disp["Bulan"] = cf_disp["month"].map(month_map)
st.dataframe(cf_disp[["Bulan", "Aktual_Mean_MWh", "Counterfactual_Mean_MWh", "Delta_Mean_MWh", "Delta_Persen", "Total_Energi_Hilang_GWh"]], width="stretch", hide_index=True)

export_cf = df_20[["datetime", "month", "hour", "is_weekend", "nat_demand", "cf_mean", "cf_ci_lo", "cf_ci_hi"]].copy()
export_cf["delta_mwh"] = export_cf["nat_demand"] - export_cf["cf_mean"]
st.download_button(
    "Unduh Deret Counterfactual Per Jam 2020 (CSV)",
    data=export_cf.to_csv(index=False).encode("utf-8"),
    file_name="panama_covid19_counterfactual_2020.csv",
    mime="text/csv",
)

render_methodology_drawer(
    "Kerangka Kausal Potential Outcomes & Penalized B-Splines GAM",
    """
- **Potential Outcomes Rubin:** Dampak kausal diestimasi sebagai $\\hat{\\Delta}_t = Y_t(1) - \\hat{Y}_t(0)$ di mana $\\hat{Y}_t(0) = \\hat{f}_{\\text{GAM, pra-COVID}}(\\mathbf{X}_t^{\\text{eksogen}})$ dilatih pada `Jan 2015 – Feb 2020` **tanpa lag beban** agar tidak terkontaminasi penurunan lockdown.
- **Uji Validasi Placebo (`Mar–Jun 2019`):** Bias rata-rata hanya `+6,35` MWh (`+0,66%`), membuktikan model tidak salah mengatribusikan fluktuasi musiman sebagai dampak kebijakan.
""",
)
