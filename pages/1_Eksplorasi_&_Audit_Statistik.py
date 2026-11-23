import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from utils.charts import (
    make_dual_axis_explorer,
    make_panama_station_map,
    plot_mpl_fft_and_acf_stem,
    plot_mpl_intraday_profiles_and_heatmap,
    plot_mpl_vif_bars,
)
from utils.data_loader import load_audit_stats, load_master_hourly
from utils.ui_components import render_methodology_drawer, render_page_header, render_shadcn_kpis

df = load_master_hourly()
audit = load_audit_stats()

render_page_header(
    "Eksplorasi & Audit Statistik Data",
    "Inspeksi deret waktu beban listrik, cuaca tiga kota (Tocumen, Santiago, David), spektrum musiman ganda, multikolinearitas, dan kejadian blackout.",
)

c1, c2, c3, c4, c5 = st.columns([2.2, 1.4, 1.4, 1.4, 1.6])
min_d = df["datetime"].min().date()
max_d = df["datetime"].max().date()
date_range = c1.date_input("Rentang Tanggal", value=(min_d, max_d), min_value=min_d, max_value=max_d)
agg_mode = c2.selectbox("Resolusi", ["Rata-rata Harian", "Per Jam", "Rata-rata Mingguan"], index=0)
weather_var = c3.selectbox("Variabel Cuaca", ["T2M (Suhu °C)", "QV2M (Kelembapan)", "TQL (Presipitasi)", "W2M (Angin m/s)"])
city_choice = c4.selectbox("Kota Stasiun", ["Tocumen (toc)", "Santiago (san)", "David (dav)"])
with c5:
    show_bo = st.toggle("Sorot Blackout (<600)", value=True)
    show_cov = st.toggle("Zona Lockdown 2020", value=True)

start_d, end_d = (date_range[0], date_range[1]) if isinstance(date_range, (list, tuple)) and len(date_range) == 2 else (min_d, max_d)
mask = (df["datetime"].dt.date >= start_d) & (df["datetime"].dt.date <= end_d)
df_sub = df.loc[mask].copy()

w_prefix = weather_var.split(" ")[0]
c_suffix = city_choice.split("(")[1].replace(")", "")
weather_col = f"{w_prefix}_{c_suffix}"

het = audit["heteroskedasticity"]
hol_diff_pct = (het["holiday1_mean"] - het["holiday0_mean"]) / het["holiday0_mean"] * 100.0

render_shadcn_kpis([
    {
        "label": "Rata-rata Beban Periode",
        "value": f"{df_sub['nat_demand'].mean():,.2f} MWh",
        "description": f"{len(df_sub):,} jam observasi terpilih",
    },
    {
        "label": "Beban Maksimum",
        "value": f"{df_sub['nat_demand'].max():,.2f} MWh",
        "description": f"Maksimum global {audit['dataset_meta']['nat_demand_max']:,.2f} MWh",
    },
    {
        "label": "Beban Minimum",
        "value": f"{df_sub['nat_demand'].min():,.2f} MWh",
        "description": f"{int(df_sub['is_blackout'].sum())} titik blackout (<600 MWh)",
    },
    {
        "label": "Efek Hari Libur (Global)",
        "value": f"{het['holiday1_mean']:,.1f} MWh",
        "description": f"Hari biasa {het['holiday0_mean']:,.1f} MWh",
        "delta": f"{hol_diff_pct:+.1f}%",
    },
], key_prefix="exp_kpi")

st.divider()

if agg_mode == "Per Jam":
    df_plot = df_sub
elif agg_mode == "Rata-rata Mingguan":
    df_plot = df_sub.set_index("datetime")[["nat_demand", weather_col, "is_blackout"]].resample("W").agg({"nat_demand": "mean", weather_col: "mean", "is_blackout": "max"}).reset_index()
else:
    df_plot = df_sub.set_index("datetime")[["nat_demand", weather_col, "is_blackout"]].resample("D").agg({"nat_demand": "mean", weather_col: "mean", "is_blackout": "max"}).reset_index()

col_ts, col_map = st.columns([8, 4], gap="large")
with col_ts:
    st.plotly_chart(
        make_dual_axis_explorer(df_plot, weather_col, f"{w_prefix} ({city_choice})", show_blackout=show_bo, show_lockdown=show_cov),
        width="stretch", theme=None, config={"displaylogo": False},
    )
with col_map:
    map_metric = st.selectbox("Pewarnaan Peta Panama", ["VIF Multikolinearitas", "Korelasi Beban (r)", "Rata-rata Suhu (°C)"])
    st.plotly_chart(make_panama_station_map(df, audit["vif_table"], metric_mode=map_metric), width="stretch", theme=None, config={"displaylogo": False})

st.subheader("Profil Beban Intraharian & Spektrum Musiman Ganda")
fig_intra = plot_mpl_intraday_profiles_and_heatmap(df, df_sub)
st.pyplot(fig_intra)
plt.close(fig_intra)

fig_fft = plot_mpl_fft_and_acf_stem(audit["fft_spectrum"], audit["acf_demand"])
st.pyplot(fig_fft)
plt.close(fig_fft)

st.divider()
st.subheader("Audit Multikolinearitas Spasial, Uji Stasioneritas, & Daftar Blackout")
c_vif, c_tables = st.columns([5.5, 6.5], gap="large")
with c_vif:
    fig_vif = plot_mpl_vif_bars(audit["vif_table"])
    st.pyplot(fig_vif)
    plt.close(fig_vif)
with c_tables:
    st.markdown("**Uji Stasioneritas ADF & KPSS (Pra-Pandemi 2015–2019)**")
    st.dataframe(pd.DataFrame(audit["stationarity"]), width="stretch", hide_index=True)
    st.caption(
        f"Heteroskedastisitas OLS Pendahuluan: Breusch-Pagan LM = {het['bp_lm']:,.2f} (p = {het['bp_p']:.2e}) · "
        f"ARCH-LM(24) = {het['arch_lm']:,.2f} (p = {het['arch_p']:.2e}) · "
        f"Std hari biasa {het['holiday0_std']:.2f} vs libur {het['holiday1_std']:.2f} MWh."
    )
    st.markdown("**8 Observasi Blackout (< 600 MWh)**")
    bo_df = pd.DataFrame(audit["blackouts"])
    st.dataframe(bo_df, width="stretch", hide_index=True)
    st.download_button(
        "Unduh Data Blackout (CSV)",
        data=bo_df.to_csv(index=False).encode("utf-8"),
        file_name="panama_blackouts_8pts.csv",
        mime="text/csv",
    )

render_methodology_drawer(
    "Penjelasan Matematis Audit Statistik",
    """
- **ADF & KPSS:** ADF menguji $H_0$ akar unit, sedangkan KPSS menguji $H_0$ stasioneritas di sekitar tren. Penolakan KPSS pada level (`KPSS_p = 0.0112`) dan penerimaan setelah diferensiasi (`KPSS_p = 0.10`) membuktikan tren & komponen musiman deterministik.
- **Periodogram FFT:** Menghitung $P(f_k) = \\frac{1}{N}|\\sum (Y_t - \\bar{Y})e^{-i2\\pi kt/N}|^2$ yang menunjukkan puncak utama di periode `24` jam (harian) dan `168` jam (mingguan).
- **VIF Spasial:** $\\text{VIF}_j = (\\mathbf{R}^{-1})_{jj} > 10$ pada suhu ketiga kota (`T2M_san = 19,69`, `T2M_dav = 14,78`, `T2M_toc = 11,13`) memotivasi reduksi dimensi `PCA` untuk model GAM.
""",
)
