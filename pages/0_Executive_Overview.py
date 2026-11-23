import pandas as pd
import streamlit as st
from utils.charts import make_panama_station_map, make_regime_timeline
from utils.data_loader import load_audit_stats, load_master_hourly
from utils.ui_components import render_methodology_drawer, render_page_header, render_shadcn_kpis

df = load_master_hourly()
audit = load_audit_stats()

render_page_header(
    "Ringkasan Beban Listrik Nasional Panama (2015–2020)",
    "Gambaran umum dataset per jam, pembagian rezim kronologis, evaluasi model backtesting, dan proyeksi masa depan.",
)

stlf_row = next(r for r in audit["backtest_summary"] if r["col_key"] == "pred_gbm")
covid_sum = audit["covid_summary"]
st_sum = audit["st_summary"]
lt_sum = audit["lt_summary"]

render_shadcn_kpis([
    {
        "label": "Total Observasi",
        "value": f"{audit['dataset_meta']['rows']:,} jam",
        "description": f"Rata-rata historis {audit['dataset_meta']['nat_demand_mean']:,.2f} MWh",
    },
    {
        "label": "STLF Terbaik (2019 No Blackout)",
        "value": f"{stlf_row['RMSE_2019_NoBO']:.2f} MWh",
        "description": f"HistGradientBoosting · MAPE {stlf_row['MAPE_2019_NoBO_%']:.2f}%",
        "delta": stlf_row["DM_vs_SARIMAX"],
    },
    {
        "label": "Defisit COVID-19 (Mar–Jun 2020)",
        "value": f"{covid_sum['total_lost_gwh']:,.2f} GWh",
        "description": f"Rata-rata {covid_sum['mean_delta_mwh']:+.2f} MWh/jam",
        "delta": f"{covid_sum['mean_delta_pct']:+.2f}% vs counterfactual",
    },
    {
        "label": "Proyeksi 7 Hari & 12 Bulan",
        "value": f"{st_sum['total_gwh']:.2f} GWh",
        "description": f"7H (168j) · 12B: {lt_sum['covid_total_twh']:.3f}–{lt_sum['normal_total_twh']:.3f} TWh",
    },
], key_prefix="ov_kpi")

st.divider()

c_f1, c_f2 = st.columns([8, 4], gap="large")
with c_f1:
    regime_filter = st.radio(
        "Filter Periode Linimasa",
        ["Semua (2015–2020)", "Training (2015–2018)", "Test Normal (2019)", "Test Disrupsi (2020)"],
        horizontal=True,
    )
    st.plotly_chart(make_regime_timeline(df, regime_filter=regime_filter), width="stretch", theme=None, config={"displaylogo": False})

with c_f2:
    map_metric = st.selectbox(
        "Metrik Peta Stasiun Cuaca",
        ["Korelasi Beban (r)", "VIF Multikolinearitas", "Rata-rata Suhu (°C)"],
        index=0,
    )
    st.plotly_chart(make_panama_station_map(df, audit["vif_table"], metric_mode=map_metric), width="stretch", theme=None, config={"displaylogo": False})

st.subheader("Pembagian Data Kronologis (3 Rezim)")
st.caption("Urutan waktu dipertahankan secara ketat untuk menghindari kebocoran data akibat autokorelasi per jam.")
st.dataframe(pd.DataFrame(audit["split_summary"]), width="stretch", hide_index=True)

render_methodology_drawer(
    "Catatan Metodologi: Train–Test Split, Cross-Validation, & Kategori Forecast",
    """
- **Train–Test Split Kronologis:** Blok latih `2015–2018` (`35.015` jam, `72,88%`) dengan `3` titik *blackout* diimputasi; Blok uji normal `2019` (`8.760` jam, `18,23%`) dievaluasi dengan & tanpa `5` titik *blackout* (`2019-01-20`); Blok uji disrupsi `2020` (`4.273` jam, `8,89%`) memuat kontrol `Jan–Feb` dan lockdown `Mar–Jun` (`2.833` jam, `5,90%`).
- **Cross-Validation Temporal:** `TimeSeriesSplit` `3`-fold (*expanding window*) dijalankan pada `2015–2018` untuk **GAM** (mean RMSE `78,23` MWh) dan **HistGradientBoosting** (mean RMSE `47,75` MWh), disertai *Placebo Test* `Mar–Jun 2019` (`+6,35` MWh).
- **Pemisahan Horizon Forecasting:** *Jangka Pendek (STLF, `h=168` jam)* oleh `HistGradientBoosting` Set B; *Jangka Panjang (LTLF, `1` tahun)* oleh `GAM` Set A eksogen murni tanpa *lag* beban.
""",
)
