import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from utils.charts import make_st_forecast_chart, plot_mpl_st_daily_bars
from utils.data_loader import load_audit_stats, load_forecast_st, load_master_hourly
from utils.ui_components import (
    render_methodology_drawer,
    render_page_header,
    render_shadcn_alert,
    render_shadcn_kpis,
)

df = load_master_hourly()
future_st = load_forecast_st()
audit = load_audit_stats()
st_sum = audit["st_summary"]
st_daily = pd.DataFrame(audit["st_daily"])

render_page_header(
    "Forecast Jangka Pendek (7 Hari / 168 Jam)",
    f"Proyeksi operasional 2020-06-27 01:00 s.d. 2020-07-04 00:00 menggunakan {audit['dataset_meta']['gbm_engine']} Set B (refit penuh {st_sum['n_train_rows']:,} jam).",
)

r1, r2, r3, r4 = st.columns([2.8, 1.8, 1.8, 2.2])
threshold_mwh = r1.slider("Ambang Batas Kapasitas Aman (MWh)", min_value=1050, max_value=1450, value=1250, step=10)
show_pi = r2.checkbox("Tampilkan Interval 90% (Q05–Q95)", value=True)
show_q50 = r3.checkbox("Tampilkan Median (Q50)", value=True)
date_opts = ["Semua 7 Hari (168 Jam)"] + st_daily["Tanggal"].tolist()
date_filter = r4.selectbox("Filter Tabel Per Jam", date_opts)

exceed_df = future_st[future_st["pred_st_mean"] > threshold_mwh]
n_exceed = len(exceed_df)
if n_exceed == 0:
    render_shadcn_alert(
        "Cadangan Operasi Aman",
        f"Seluruh 168 jam berada di bawah ambang {threshold_mwh:,.0f} MWh (puncak prediksi {st_sum['max_mwh']:,.2f} MWh).",
        destructive=False,
        key="st_alert_safe",
    )
else:
    render_shadcn_alert(
        f"Peringatan Beban Puncak ({n_exceed} jam di atas ambang {threshold_mwh:,.0f} MWh)",
        f"Puncak prediksi mencapai {st_sum['max_mwh']:,.2f} MWh (+{st_sum['max_mwh'] - threshold_mwh:,.2f} MWh di atas ambang; batas Q95 hingga {future_st['pred_st_q95'].max():,.2f} MWh).",
        destructive=True,
        key="st_alert_warn",
    )

render_shadcn_kpis([
    {
        "label": "Rata-rata Beban 7 Hari",
        "value": f"{st_sum['mean_mwh']:,.2f} MWh",
        "description": "Horizon operasional h = 168 jam",
    },
    {
        "label": "Puncak – Minimum",
        "value": f"{st_sum['max_mwh']:,.2f} MWh",
        "description": f"Minimum {st_sum['min_mwh']:,.2f} MWh",
    },
    {
        "label": "Total Energi 7 Hari",
        "value": f"{st_sum['total_gwh']:.2f} GWh",
        "description": f"Refit {st_sum['n_train_rows']:,} jam bersih",
    },
    {
        "label": "Lebar Rata-rata PI 90%",
        "value": f"{st_sum['mean_pi_width']:.2f} MWh",
        "description": "Interval kuantil terurut Q05–Q95",
    },
], key_prefix="st_kpi")

st.divider()

hist_7d = df.iloc[-168:].copy()
st.plotly_chart(
    make_st_forecast_chart(hist_7d, future_st, threshold_mwh=threshold_mwh, show_pi=show_pi, show_q50=show_q50),
    width="stretch", theme=None, config={"displaylogo": False},
)

c_bars, c_tbl = st.columns([6, 6], gap="large")
with c_bars:
    fig_st_bars = plot_mpl_st_daily_bars(st_daily)
    st.pyplot(fig_st_bars)
    plt.close(fig_st_bars)
with c_tbl:
    st.subheader("Ringkasan Harian (27 Jun – 4 Jul 2020)")
    st.dataframe(st_daily, width="stretch", hide_index=True)

st.subheader("Tabel Detail Per Jam (168 Jam)")
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

render_methodology_drawer(
    "Asumsi Klimatologi & Fitur Lag Non-Rekursif (Tahap 8A)",
    """
- **Lag Beban 100% Historis:** Karena horizon prediksi adalah `168` jam ke depan (`27 Jun – 4 Jul 2020`), seluruh fitur `load_lag_168`, `load_lag_336`, `load_lag_504`, serta statistik bergerak `168` jam merujuk langsung pada observasi historis bersih (`nat_demand_clean_full`, dengan ke-8 titik *blackout* terimputasi) tanpa rekursi berantai.
- **Klimatologi Cuaca & Kalender:** Cuaca 168 jam ke depan memakai rata-rata historis `2015–2019` per `(month, day, hour)` dengan transisi lag suhu mulus dari 24 jam aktual terakhir; kalender mengikuti pola normal akhir Juni–awal Juli (`holiday=0`, `Holiday_ID=0`, `school=1`).
- **Monotonisitas Kuantil:** Pengurutan ulang (`np.sort`) menjamin $\\hat{Q}_{0.05} \\le \\hat{Q}_{0.50} \\le \\hat{Q}_{0.95}$ di setiap jam.
""",
)
