"""
Soft UI / Neumorphism Visualization Library (Hybrid Matplotlib + Plotly):
Blends seamlessly into #EAEFF5 / #EEF3F9 Soft UI cards while keeping plain-Indonesian labels:
- Slate Ink (#1E293B, #64748B, #94A3B8) for base series & structure
- Soft Indigo (#3B6EF6) for GAM & primary trend lines
- Soft Teal (#0D9488) for STLF Boosting solution
- Soft Rose (#E11D48) strictly for disruptions, COVID-19 contrast, or threshold alerts
- Warm Amber (#D97706) for weather/holiday highlights
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plt.rcParams.update({
    "figure.dpi": 135,
    "figure.facecolor": "#EEF3F9",
    "axes.facecolor": "#F7FAFD",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#E2E8F0",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": "#1E293B",
    "axes.labelcolor": "#334155",
    "xtick.color": "#64748B",
    "ytick.color": "#64748B",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "legend.frameon": False,
})

COLORS = {
    "actual": "#1E293B",
    "blue": "#3B6EF6",
    "blue_fill": "rgba(59, 110, 246, 0.14)",
    "emerald": "#0D9488",
    "emerald_fill": "rgba(13, 148, 136, 0.15)",
    "red": "#E11D48",
    "red_fill": "rgba(225, 29, 72, 0.12)",
    "amber": "#D97706",
    "muted": "#64748B",
    "light_gray": "#94A3B8",
    "grid": "#E2E8F0",
    "border": "#CBD5E1",
}

PANAMA_STATIONS = [
    {"city": "Tocumen (Kota Panama)", "code": "Kota Panama", "lat": 9.0714, "lon": -79.3835, "t2m_col": "T2M_toc"},
    {"city": "Santiago (Veraguas)", "code": "Santiago", "lat": 8.1000, "lon": -80.9833, "t2m_col": "T2M_san"},
    {"city": "David (Chiriquí)", "code": "David", "lat": 8.4273, "lon": -82.4308, "t2m_col": "T2M_dav"},
]


# ==============================================================================
# SOFT UI PLOTLY HELPERS (Interactive Time-Series & Panama Spatial Map)
# ==============================================================================
def apply_clean_plotly_layout(fig, title="", height=380, y_title="", x_title="", show_legend=True, margin=None):
    m = margin or dict(l=52, r=20, t=68 if title else 40, b=38)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Plus Jakarta Sans, Inter, sans-serif", size=14, color="#1E293B"),
            x=0,
            xanchor="left",
            y=0.96,
        ),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255, 255, 255, 0.50)",
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", size=12, color="#334155"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#F4F8FD", bordercolor="#CBD5E1", font=dict(size=11, color="#1E293B")),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            bgcolor="rgba(244,248,253,0.75)",
            font=dict(size=11, color="#475569"),
        ),
        margin=m,
    )
    fig.update_xaxes(
        title_text=x_title,
        showgrid=False,
        zeroline=False,
        linecolor="#CBD5E1",
        tickfont=dict(size=10, color="#64748B"),
    )
    fig.update_yaxes(
        title_text=y_title,
        showgrid=True,
        gridcolor="#DCE5F2",
        zeroline=False,
        linecolor="#CBD5E1",
        tickfont=dict(size=10, color="#64748B"),
    )
    return fig


def make_panama_station_map(df_master, vif_rows, metric_mode="Rata-rata Suhu (°C)"):
    vif_map = {r["Variabel_Cuaca"]: r for r in vif_rows}
    rows = []
    for stn in PANAMA_STATIONS:
        c = stn["t2m_col"]
        vinfo = vif_map.get(c, {})
        rows.append({
            "city": stn["city"],
            "code": stn["code"],
            "lat": stn["lat"],
            "lon": stn["lon"],
            "mean_t2m": float(df_master[c].mean()),
            "corr": float(vinfo.get("Corr_nat_demand", 0.0)),
            "vif": float(vinfo.get("VIF", 0.0)),
        })
    sdf = pd.DataFrame(rows)
    if metric_mode == "VIF Multikolinearitas":
        color_vals, ctitle = sdf["vif"], "VIF"
        cscale = [[0.0, "#93C5FD"], [1.0, "#3B6EF6"]]
    elif metric_mode == "Korelasi Beban (r)":
        color_vals, ctitle = sdf["corr"], "r"
        cscale = [[0.0, "#5EEAD4"], [1.0, "#0D9488"]]
    else:
        color_vals, ctitle = sdf["mean_t2m"], "°C"
        cscale = [[0.0, "#93C5FD"], [1.0, "#3B6EF6"]]

    hover_text = [
        f"<b>{r['city']}</b><br>"
        f"Koordinat: {r['lat']:.2f}°N, {abs(r['lon']):.2f}°W<br>"
        f"Rata-rata Suhu: {r['mean_t2m']:.2f} °C<br>"
        f"Hubungan dgn Beban Listrik: r = {r['corr']:.3f}<br>"
        f"Kemiripan Antar-Kota (VIF): {r['vif']:.2f}"
        for _, r in sdf.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=sdf["lon"], lat=sdf["lat"], mode="lines",
        line=dict(width=1.6, color="rgba(59, 110, 246, 0.35)", dash="dot"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scattergeo(
        lon=sdf["lon"], lat=sdf["lat"],
        mode="markers+text",
        text=[f"{r['code']} ({r['mean_t2m']:.1f}°C)" for _, r in sdf.iterrows()],
        textposition=["top right", "bottom center", "top left"],
        textfont=dict(size=10.5, color="#1E293B"),
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(
            size=[15, 13, 13],
            color=color_vals,
            colorscale=cscale,
            showscale=True,
            colorbar=dict(title=dict(text=ctitle, font=dict(size=10)), tickfont=dict(size=9), len=0.72, thickness=9),
            line=dict(width=1.8, color="#FFFFFF"),
        ),
    ))
    fig.update_geos(
        resolution=50,
        showcoastlines=True, coastlinecolor="#94A3B8",
        showcountries=True, countrycolor="#CBD5E1",
        showland=True, landcolor="#F4F7FC",
        showocean=True, oceancolor="#DDE7F5",
        lataxis=dict(range=[6.8, 10.1]),
        lonaxis=dict(range=[-83.4, -77.3]),
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title=dict(text="Peta 3 Stasiun Cuaca Panama", font=dict(family="Plus Jakarta Sans, Inter, sans-serif", size=14, color="#1E293B"), x=0),
        height=345,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=4, r=4, t=42, b=4),
        showlegend=False,
        hoverlabel=dict(bgcolor="#F4F8FD", bordercolor="#CBD5E1", font=dict(size=11, color="#1E293B")),
    )
    return fig


def make_regime_timeline(df_master, regime_filter="Semua Tahun (2015–2020)"):
    daily = df_master.set_index("datetime")["nat_demand"].resample("D").mean().reset_index()
    bo = df_master[df_master["is_blackout"] == 1]

    if regime_filter == "Sebelum Pandemi (2015–2019)":
        daily = daily[daily["datetime"] <= "2019-12-31 23:00:00"]
        bo = bo[bo["datetime"] <= "2019-12-31 23:00:00"]
    elif regime_filter == "Training (2015–2018)":
        daily = daily[daily["datetime"] <= "2018-12-31 23:00:00"]
        bo = bo[bo["datetime"] <= "2018-12-31 23:00:00"]
    elif regime_filter == "Test Normal (2019)":
        daily = daily[(daily["datetime"] >= "2019-01-01") & (daily["datetime"] <= "2019-12-31 23:00:00")]
        bo = bo[(bo["datetime"] >= "2019-01-01") & (bo["datetime"] <= "2019-12-31 23:00:00")]
    elif regime_filter in ("Masa Pandemi (2020)", "Test Disrupsi (2020)"):
        daily = daily[daily["datetime"] >= "2020-01-01"]
        bo = bo[bo["datetime"] >= "2020-01-01"]

    fig = go.Figure()
    if regime_filter in ("Semua Tahun (2015–2020)", "Semua (2015–2020)"):
        fig.add_vrect(
            x0="2015-01-03", x1="2018-12-31",
            fillcolor="rgba(148, 163, 184, 0.10)", line_width=0,
            annotation_text="Data Latih (2015–2018)", annotation_position="top left",
            annotation_font=dict(size=10, color="#64748B"),
        )
        fig.add_vrect(
            x0="2019-01-01", x1="2019-12-31",
            fillcolor="rgba(59, 110, 246, 0.08)", line_width=0,
            annotation_text="Uji Normal (2019)", annotation_position="top left",
            annotation_font=dict(size=10, color="#3B6EF6"),
        )
        fig.add_vrect(
            x0="2020-03-01", x1="2020-06-27",
            fillcolor="rgba(225, 29, 72, 0.09)", line_width=0,
            annotation_text="Pandemi COVID-19", annotation_position="top left",
            annotation_font=dict(size=10, color="#E11D48"),
        )
    elif regime_filter == "Masa Pandemi (2020)":
        fig.add_vrect(
            x0="2020-03-01", x1="2020-06-27",
            fillcolor="rgba(225, 29, 72, 0.09)", line_width=0,
            annotation_text="Pembatasan COVID-19", annotation_position="top left",
            annotation_font=dict(size=10, color="#E11D48"),
        )

    fig.add_trace(go.Scatter(
        x=daily["datetime"], y=daily["nat_demand"], mode="lines",
        line=dict(color=COLORS["blue"], width=1.6),
        name="Rata-rata Harian (MWh)",
    ))
    if len(bo) > 0:
        fig.add_trace(go.Scatter(
            x=bo["datetime"], y=bo["nat_demand"], mode="markers",
            marker=dict(color=COLORS["red"], size=7, symbol="diamond", line=dict(width=1, color="#FFFFFF")),
            name="Gangguan Listrik / Blackout",
        ))
    return apply_clean_plotly_layout(fig, title="Tren Pemakaian Listrik Harian Panama (2015–2020)", height=345, y_title="Beban (MWh)")


def make_dual_axis_explorer(df_sub, weather_col, weather_label, show_blackout=True, show_lockdown=True):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df_sub["datetime"], y=df_sub["nat_demand"], name="Pemakaian Listrik (MWh)", line=dict(color=COLORS["blue"], width=1.6)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df_sub["datetime"], y=df_sub[weather_col], name=weather_label, line=dict(color=COLORS["amber"], width=1.3), opacity=0.85),
        secondary_y=True,
    )
    if show_blackout:
        bo = df_sub[df_sub["is_blackout"] == 1]
        if len(bo) > 0:
            fig.add_trace(
                go.Scatter(
                    x=bo["datetime"], y=bo["nat_demand"], mode="markers",
                    marker=dict(color=COLORS["red"], size=7, symbol="x"),
                    name="Gangguan Listrik (<600 MWh)",
                ),
                secondary_y=False,
            )
    if show_lockdown and df_sub["datetime"].max() >= pd.Timestamp("2020-03-01"):
        x0 = max(df_sub["datetime"].min(), pd.Timestamp("2020-03-01"))
        fig.add_vrect(x0=x0, x1=df_sub["datetime"].max(), fillcolor=COLORS["red_fill"], line_width=0)

    apply_clean_plotly_layout(fig, title=f"Perbandingan Listrik Nasional vs {weather_label}", height=385, margin=dict(l=52, r=52, t=68, b=38))
    fig.update_yaxes(title_text="Beban Listrik (MWh)", secondary_y=False, gridcolor="#DCE5F2")
    fig.update_yaxes(title_text=weather_label, secondary_y=True, showgrid=False)
    return fig


def make_weekly_zoom_chart(df_week, selected_models, show_pi=True):
    col_style = {
        "pred_naive": ("Patokan Minggu Lalu (Naive)", COLORS["light_gray"], "dot"),
        "pred_sarima": ("Statistik Klasik (SARIMA)", "#94A3B8", "dash"),
        "pred_sarimax": ("Statistik Linear + Cuaca (SARIMAX)", COLORS["red"], "dash"),
        "pred_gam": ("Model Kurva GAM (Solusi 12 Bulan)", COLORS["blue"], "solid"),
        "pred_gbm": ("Machine Learning Boosting (Terbaik 7 Hari)", COLORS["emerald"], "solid"),
    }
    fig = go.Figure()
    if show_pi and "pred_gbm" in selected_models:
        fig.add_trace(go.Scatter(
            x=pd.concat([df_week["datetime"], df_week["datetime"][::-1]]),
            y=pd.concat([df_week["pred_gbm_q95"], df_week["pred_gbm_q05"][::-1]]),
            fill="toself", fillcolor=COLORS["emerald_fill"], line=dict(color="rgba(0,0,0,0)"),
            name="Rentang Kemungkinan 90% (Boosting)", hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(x=df_week["datetime"], y=df_week["nat_demand"], mode="lines", line=dict(color=COLORS["actual"], width=2.3), name="Pemakaian Aktual"))
    for col in selected_models:
        lbl, clr, dash = col_style[col]
        fig.add_trace(go.Scatter(x=df_week["datetime"], y=df_week[col], mode="lines", line=dict(color=clr, width=1.8, dash=dash), name=lbl))
    return apply_clean_plotly_layout(fig, title="Zoom Mingguan: Pemakaian Aktual vs Prediksi Model", height=390, y_title="Beban (MWh)")


def make_st_forecast_chart(hist_7d, future_st, threshold_mwh=1250.0, show_pi=True, show_q50=False):
    fig = go.Figure()
    if show_pi:
        fig.add_trace(go.Scatter(
            x=pd.concat([future_st["datetime"], future_st["datetime"][::-1]]),
            y=pd.concat([future_st["pred_st_q95"], future_st["pred_st_q05"][::-1]]),
            fill="toself", fillcolor=COLORS["emerald_fill"], line=dict(color="rgba(0,0,0,0)"),
            name="Rentang Kemungkinan 90%", hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(x=hist_7d["datetime"], y=hist_7d["nat_demand"], name="Aktual 7 Hari Terakhir", line=dict(color=COLORS["actual"], width=2.0)))
    fig.add_trace(go.Scatter(x=future_st["datetime"], y=future_st["pred_st_mean"], name="Prediksi 7 Hari ke Depan", line=dict(color=COLORS["emerald"], width=2.3)))
    if show_q50:
        fig.add_trace(go.Scatter(x=future_st["datetime"], y=future_st["pred_st_q50"], name="Nilai Tengah (Median)", line=dict(color=COLORS["blue"], width=1.3, dash="dot")))

    exceed = future_st[future_st["pred_st_mean"] > threshold_mwh]
    if len(exceed) > 0:
        fig.add_trace(go.Scatter(
            x=exceed["datetime"], y=exceed["pred_st_mean"], mode="markers",
            marker=dict(color=COLORS["red"], size=6, line=dict(width=1, color="#FFFFFF")),
            name=f"Melebihi Batas Siaga ({threshold_mwh:,.0f} MWh)",
        ))

    cutoff_ms = hist_7d["datetime"].max().timestamp() * 1000
    fig.add_vline(x=cutoff_ms, line_dash="dash", line_color="#64748B", annotation_text="Batas Data Terakhir", annotation_font=dict(color="#64748B", size=10))
    fig.add_hline(y=threshold_mwh, line_dash="dot", line_color=COLORS["red"], annotation_text=f"Batas Siaga {threshold_mwh:,.0f} MWh", annotation_font=dict(color=COLORS["red"], size=10))
    return apply_clean_plotly_layout(fig, title="Prediksi Kebutuhan Listrik 7 Hari (27 Jun – 4 Jul 2020)", height=405, y_title="Beban (MWh)")


def make_lt_forecast_chart(hist_monthly, lt_monthly, show_normal=True, show_covid=True, custom_monthly=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_monthly["datetime"], y=hist_monthly["nat_demand"], name="Aktual Bulanan (2015–Jun 2020)", line=dict(color=COLORS["actual"], width=2.0)))

    if show_normal:
        fig.add_trace(go.Scatter(
            x=pd.concat([lt_monthly["datetime"], lt_monthly["datetime"][::-1]]),
            y=pd.concat([lt_monthly["lt_base_hi"], lt_monthly["lt_base_lo"][::-1]]),
            fill="toself", fillcolor=COLORS["blue_fill"], line=dict(color="rgba(0,0,0,0)"),
            name="Rentang Ketidakpastian Normal", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(x=lt_monthly["datetime"], y=lt_monthly["lt_base_mean"], mode="lines+markers", name="Skenario Pulih Normal", line=dict(color=COLORS["blue"], width=2.0, dash="dash"), marker=dict(size=5)))

    if show_covid:
        fig.add_trace(go.Scatter(
            x=pd.concat([lt_monthly["datetime"], lt_monthly["datetime"][::-1]]),
            y=pd.concat([lt_monthly["lt_cov_hi"], lt_monthly["lt_cov_lo"][::-1]]),
            fill="toself", fillcolor=COLORS["red_fill"], line=dict(color="rgba(0,0,0,0)"),
            name="Rentang Ketidakpastian Pandemi", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(x=lt_monthly["datetime"], y=lt_monthly["lt_cov_mean"], mode="lines+markers", name="Skenario Pandemi Berlanjut", line=dict(color=COLORS["red"], width=2.0, dash="dashdot"), marker=dict(size=5)))

    if custom_monthly is not None:
        fig.add_trace(go.Scatter(
            x=lt_monthly["datetime"], y=custom_monthly["Custom_Mean_MWh"], mode="lines+markers",
            name="Simulasi Pilihan Anda", line=dict(color=COLORS["emerald"], width=2.6), marker=dict(size=6, symbol="diamond"),
        ))

    fig.add_vline(x=pd.Timestamp("2020-06-27").timestamp() * 1000, line_dash="dot", line_color="#64748B")
    apply_clean_plotly_layout(fig, title="Proyeksi Rata-rata Beban Bulanan 12 Bulan (Jul 2020 – Jun 2021)", height=395, y_title="Rata-rata Bulanan (MWh)")
    fig.update_layout(uirevision="lt_forecast")
    return fig


def make_whatif_24h_chart(sim_df, interval_label, baseline_hourly_2019, hour_focus):
    base_y = [baseline_hourly_2019.get(int(h), np.nan) for h in sim_df["hour"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([sim_df["hour"], sim_df["hour"][::-1]]),
        y=pd.concat([sim_df["pred_hi"], sim_df["pred_lo"][::-1]]),
        fill="toself", fillcolor=COLORS["blue_fill"], line=dict(color="rgba(0,0,0,0)"),
        name=interval_label, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=sim_df["hour"], y=base_y, name="Rata-rata Hari Normal 2019", line=dict(color="#94A3B8", width=1.8, dash="dot")))
    fig.add_trace(go.Scatter(x=sim_df["hour"], y=sim_df["pred_mean"], mode="lines+markers", name="Hasil Simulasi Anda", line=dict(color=COLORS["blue"], width=2.5), marker=dict(size=5)))

    focus_row = sim_df[sim_df["hour"] == float(hour_focus)].iloc[0]
    fig.add_trace(go.Scatter(
        x=[focus_row["hour"]], y=[focus_row["pred_mean"]], mode="markers",
        marker=dict(size=11, color=COLORS["emerald"], line=dict(width=2, color="#FFFFFF")),
        name=f"Jam Dipilih ({int(hour_focus):02d}:00)",
    ))
    apply_clean_plotly_layout(fig, title="Perkiraan Pemakaian Listrik Sepanjang 24 Jam", height=380, x_title="Jam (00:00–23:00)", y_title="Prediksi Beban (MWh)")
    fig.update_xaxes(dtick=2)
    fig.update_layout(uirevision="whatif_24h")
    return fig


# ==============================================================================
# SOFT UI MATPLOTLIB FIGURES
# ==============================================================================
def plot_mpl_intraday_profiles_and_heatmap(df_master, df_sub):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.3))

    slices = [
        ("2015", (df_master["year"] == 2015), "#94A3B8", "--", 1.4),
        ("2017", (df_master["year"] == 2017), "#64748B", "-.", 1.4),
        ("2019 (Tahun Normal)", (df_master["year"] == 2019), "#3B6EF6", "-", 2.2),
        ("Mar–Jun 2020 (COVID-19)", (df_master["datetime"] >= "2020-03-01"), "#E11D48", "-", 2.2),
    ]
    for lbl, msk, clr, ls, lw in slices:
        prof = df_master.loc[msk].groupby("hour")["nat_demand"].mean()
        axes[0].plot(prof.index, prof.values, label=lbl, color=clr, linestyle=ls, linewidth=lw)
    axes[0].set_title("A. Pola Pemakaian Listrik dari Pagi hingga Malam")
    axes[0].set_xlabel("Jam (00:00–23:00)")
    axes[0].set_ylabel("Rata-rata Beban (MWh)")
    axes[0].set_xticks(range(0, 24, 3))
    axes[0].legend(fontsize=8.5, loc="upper left")

    day_map = {0: "Sen", 1: "Sel", 2: "Rab", 3: "Kam", 4: "Jum", 5: "Sab", 6: "Min"}
    piv = df_sub.groupby(["day_of_week", "hour"])["nat_demand"].mean().unstack()
    im = axes[1].imshow(piv.values, aspect="auto", cmap="Blues", origin="upper")
    axes[1].set_yticks(range(len(piv.index)))
    axes[1].set_yticklabels([day_map[int(d)] for d in piv.index])
    axes[1].set_xticks(range(0, 24, 3))
    axes[1].set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
    axes[1].set_xlabel("Jam")
    axes[1].set_title("B. Peta Panas Pemakaian Listrik per Hari & Jam")
    axes[1].grid(False)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("MWh", fontsize=8.5)

    plt.tight_layout()
    return fig


def plot_mpl_model_accuracy_layman(summary_df, use_no_bo=True):
    mape_col = "MAPE_2019_NoBO_%" if use_no_bo else "MAPE_2019_All_%"
    rmse_col = "RMSE_2019_NoBO" if use_no_bo else "RMSE_2019_All"

    name_map = {
        "HistGradientBoosting (Set B h=168)": ("Machine Learning Boosting\n(Terbaik 7 Hari)", "#0D9488"),
        "GAM (Set A Eksogen)": ("Model Kurva Non-Linear GAM\n(Terbaik 12 Bulan)", "#3B6EF6"),
        "Seasonal Naive (Y_t-168)": ("Patokan Minggu Lalu\n(Seasonal Naive)", "#94A3B8"),
        "SARIMAX (Linear)": ("Statistik Linear + Cuaca\n(SARIMAX)", "#CBD5E1"),
        "SARIMA(1,0,1)(1,1,0)_24": ("Statistik Klasik Tanpa Cuaca\n(SARIMA)", "#CBD5E1"),
    }

    work = summary_df.copy()
    work["Label"] = work["Model"].map(lambda m: name_map.get(m, (m, "#94A3B8"))[0])
    work["BarColor"] = work["Model"].map(lambda m: name_map.get(m, (m, "#94A3B8"))[1])
    work = work.sort_values(mape_col, ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.1))
    y = np.arange(len(work))

    b1 = axes[0].barh(y, work[mape_col], color=work["BarColor"], height=0.58)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(work["Label"], fontsize=8.5)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(work[mape_col]) * 1.24)
    axes[0].set_xlabel("Rata-rata Meleset (%)")
    axes[0].set_title("A. Tingkat Meleset (% MAPE — Makin Kecil Makin Akurat)")
    axes[0].grid(True, axis="x", color="#E2E8F0")
    axes[0].grid(False, axis="y")
    for bar, v in zip(b1, work[mape_col]):
        axes[0].text(v + 0.14, bar.get_y() + bar.get_height() / 2, f"{v:.2f}%", va="center", fontsize=8.5, fontweight="bold", color="#1E293B")

    b2 = axes[1].barh(y, work[rmse_col], color=work["BarColor"], height=0.58)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(["" for _ in y])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(work[rmse_col]) * 1.24)
    axes[1].set_xlabel("Rata-rata Selisih Prediksi (MWh)")
    axes[1].set_title("B. Rata-rata Selisih Energi (MWh RMSE — Makin Kecil Makin Baik)")
    axes[1].grid(True, axis="x", color="#E2E8F0")
    axes[1].grid(False, axis="y")
    for bar, v in zip(b2, work[rmse_col]):
        axes[1].text(v + 2.0, bar.get_y() + bar.get_height() / 2, f"{v:.1f} MWh", va="center", fontsize=8.5, fontweight="bold", color="#1E293B")

    plt.tight_layout()
    return fig


def plot_mpl_fft_and_acf_stem(fft_spectrum, acf_vals):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.1))

    axes[0].plot(fft_spectrum["periods"], fft_spectrum["power"], color="#3B6EF6", lw=1.4)
    axes[0].axvline(24, color="#E11D48", ls="--", lw=1.1, label="S1 = 24 jam (Harian)")
    axes[0].axvline(168, color="#64748B", ls=":", lw=1.3, label="S2 = 168 jam (Mingguan)")
    axes[0].set_title("A. Periodogram FFT: Musiman Ganda")
    axes[0].set_xlabel("Periode (Jam)")
    axes[0].set_ylabel("Spektral Power")
    axes[0].legend(fontsize=8.5)

    lags = np.arange(len(acf_vals))
    axes[1].stem(lags, acf_vals, linefmt="#3B6EF6", markerfmt=" ", basefmt="#1E293B")
    axes[1].axvline(24, color="#E11D48", ls="--", lw=1.1, label="Lag 24")
    axes[1].axvline(168, color="#64748B", ls=":", lw=1.3, label="Lag 168")
    axes[1].set_title("B. ACF nat_demand hingga Lag 180 Jam")
    axes[1].set_xlabel("Lag (Jam)")
    axes[1].set_ylabel("Autokorelasi")
    axes[1].legend(fontsize=8.5)

    plt.tight_layout()
    return fig


def plot_mpl_vif_bars(vif_rows):
    vdf = pd.DataFrame(vif_rows).sort_values("VIF", ascending=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    clrs = ["#E11D48" if v > 10 else "#3B6EF6" for v in vdf["VIF"]]
    bars = ax.barh(vdf["Variabel_Cuaca"], vdf["VIF"], color=clrs, height=0.66)
    ax.axvline(10, color="#64748B", ls="--", lw=1.2, label="Ambang VIF = 10")
    for bar, v in zip(bars, vdf["VIF"]):
        ax.text(v + 0.35, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", fontsize=8, color="#1E293B")
    ax.set_xlim(0, max(vdf["VIF"]) * 1.15)
    ax.set_title("Audit Multikolinearitas Spasial 12 Fitur Cuaca (VIF)")
    ax.set_xlabel("Variance Inflation Factor (VIF)")
    ax.grid(True, axis="x", color="#E2E8F0")
    ax.grid(False, axis="y")
    ax.legend(fontsize=8.5, loc="lower right")
    plt.tight_layout()
    return fig


def plot_mpl_backtest_and_acf_comparison(summary_df, acf_err, use_no_bo=True):
    rmse_col = "RMSE_2019_NoBO" if use_no_bo else "RMSE_2019_All"
    mae_col = "MAE_2019_NoBO" if use_no_bo else "MAE_2019_All"
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.3))

    x = np.arange(len(summary_df))
    b1 = axes[0].bar(x - 0.18, summary_df[rmse_col], width=0.35, color="#3B6EF6", label="RMSE 2019 (MWh)")
    b2 = axes[0].bar(x + 0.18, summary_df[mae_col], width=0.35, color="#94A3B8", label="MAE 2019 (MWh)")
    for b in list(b1) + list(b2):
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.split(" (")[0] for m in summary_df["Model"]], rotation=14, ha="right", fontsize=8.5)
    axes[0].set_ylim(0, max(summary_df[rmse_col]) * 1.18)
    axes[0].set_ylabel("Error (MWh)")
    axes[0].set_title("A. Perbandingan RMSE & MAE pada Rezim Normal 2019")
    axes[0].legend(fontsize=8.5)

    lags = acf_err["lags"]
    axes[1].plot(lags, acf_err["sarimax"], label="SARIMAX", color="#E11D48", lw=1.3)
    axes[1].plot(lags, acf_err["gam"], label="GAM (Set A)", color="#3B6EF6", lw=1.3)
    axes[1].plot(lags, acf_err["gbm"], label="HistGradientBoosting (Set B)", color="#0D9488", lw=1.4)
    axes[1].axvline(168, color="#64748B", ls="--", lw=1.1, label="Lag 168")
    axes[1].axhline(0, color="#1E293B", lw=0.7)
    axes[1].set_title("B. Autokorelasi Error Test 2019 (Lag 1–174)")
    axes[1].set_xlabel("Lag (Jam)")
    axes[1].set_ylabel("ACF Error")
    axes[1].legend(fontsize=8.5)

    plt.tight_layout()
    return fig


def plot_mpl_tscv_3fold(cv_gam_rows, cv_gbm_rows):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.9))

    folds = [1, 2, 3]
    tr_years = [1, 2, 3]
    for i, f in enumerate(folds):
        axes[0].barh(f, tr_years[i], left=2015, height=0.44, color="#94A3B8", label="Blok Latih" if i == 0 else "")
        axes[0].barh(f, 1, left=2015 + tr_years[i], height=0.44, color="#3B6EF6", label="Blok Uji" if i == 0 else "")
        axes[0].text(2015 + tr_years[i] + 0.5, f, f"Uji {2015 + tr_years[i]}", va="center", ha="center", color="#FFFFFF", fontsize=8, fontweight="bold")
    axes[0].set_yticks(folds)
    axes[0].set_yticklabels([f"Tahap {f}" for f in folds])
    axes[0].invert_yaxis()
    axes[0].set_xlim(2014.9, 2019.2)
    axes[0].set_xticks([2015, 2016, 2017, 2018, 2019])
    axes[0].set_title("A. Skema Uji Konsistensi Bertahap (2015–2018)")
    axes[0].grid(True, axis="x", color="#E2E8F0")
    axes[0].grid(False, axis="y")
    axes[0].legend(fontsize=8.5, loc="lower left")

    gam_r = [r["RMSE"] for r in cv_gam_rows]
    gbm_r = [r["RMSE"] for r in cv_gbm_rows]
    x = np.arange(len(folds))
    b1 = axes[1].bar(x - 0.17, gam_r, width=0.33, color="#3B6EF6", label="GAM (12 Bulan)")
    b2 = axes[1].bar(x + 0.17, gbm_r, width=0.33, color="#0D9488", label="Boosting (7 Hari)")
    for b in list(b1) + list(b2):
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width() / 2, h + 1.2, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Uji {2015+f}" for f in folds])
    axes[1].set_ylim(0, max(gam_r) * 1.2)
    axes[1].set_ylabel("Selisih Prediksi RMSE (MWh)")
    axes[1].set_title("B. Konsistensi Akurasi Antar-Tahun (3-Fold CV)")
    axes[1].legend(fontsize=8.5)

    plt.tight_layout()
    return fig


def plot_mpl_sarima_inadequacy_3panel(df_19, df_apr, hourly_rmse):
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0))
    from statsmodels.tsa.stattools import acf as sm_acf

    err_19 = (df_19["nat_demand"] - df_19["pred_sarimax"]).values
    acf_err = sm_acf(err_19, nlags=180, fft=True)
    conf = 1.96 / np.sqrt(len(err_19))
    axes[0].stem(np.arange(len(acf_err)), acf_err, linefmt="#3B6EF6", markerfmt=" ", basefmt="#1E293B")
    axes[0].axhline(conf, color="#E11D48", ls="--", lw=0.9)
    axes[0].axhline(-conf, color="#E11D48", ls="--", lw=0.9)
    axes[0].axvline(168, color="#64748B", ls=":", lw=1.2, label="Lag 168")
    axes[0].set_title("1. ACF Error SARIMAX (2019)")
    axes[0].set_xlabel("Lag (Jam)")
    axes[0].legend(fontsize=8)

    axes[1].plot(df_apr["datetime"], df_apr["nat_demand"], label="Aktual Apr 2020", color="#1E293B", lw=1.3)
    axes[1].plot(df_apr["datetime"], df_apr["pred_sarimax"], label="Prediksi SARIMAX (168j)", color="#E11D48", ls="--", lw=1.2)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[1].tick_params(axis="x", rotation=20, labelsize=8)
    axes[1].set_title("2. Over-Forecast saat Lockdown Apr 2020")
    axes[1].set_ylabel("MWh")
    axes[1].legend(fontsize=8)

    axes[2].bar(range(24), hourly_rmse, color="#64748B", width=0.72)
    axes[2].set_title("3. RMSE SARIMAX per Jam (2019)")
    axes[2].set_xlabel("Jam (0–23)")
    axes[2].set_ylabel("RMSE (MWh)")
    axes[2].set_xticks(range(0, 24, 3))

    plt.tight_layout()
    return fig


def plot_mpl_covid_counterfactual_2panel(df_20_sub, df_20_full, month_focus=4, day_type="Hari Kerja"):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.3))

    d20 = df_20_sub.set_index("datetime")[["nat_demand", "cf_mean", "cf_ci_lo", "cf_ci_hi"]].resample("D").mean()
    axes[0].plot(d20.index, d20["nat_demand"], color="#1E293B", lw=1.7, label="Pemakaian Aktual 2020")
    axes[0].plot(d20.index, d20["cf_mean"], color="#3B6EF6", ls="--", lw=1.7, label="Perkiraan Tanpa COVID (Normal)")
    axes[0].fill_between(d20.index, d20["cf_ci_lo"], d20["cf_ci_hi"], color="#3B6EF6", alpha=0.13, label="Rentang 95%")
    axes[0].axvline(pd.Timestamp("2020-03-01"), color="#E11D48", ls=":", lw=1.4, label="Awal Pembatasan (Mar 2020)")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[0].tick_params(axis="x", rotation=18, labelsize=8.5)
    axes[0].set_title("A. Pemakaian Aktual vs Perkiraan Kondisi Normal (2020)")
    axes[0].set_ylabel("Rata-rata Harian (MWh)")
    axes[0].legend(fontsize=8, loc="lower left")

    is_wknd = 1 if day_type == "Akhir Pekan" else 0
    month_names = {1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni"}
    prof = df_20_full[(df_20_full["month"] == int(month_focus)) & (df_20_full["is_weekend"] == is_wknd)].groupby("hour")[["nat_demand", "cf_mean"]].mean()
    axes[1].plot(prof.index, prof["cf_mean"], color="#3B6EF6", ls="--", lw=2.0, label="Perkiraan Normal (Tanpa COVID)")
    axes[1].plot(prof.index, prof["nat_demand"], color="#1E293B", lw=2.0, label="Aktual 2020")
    axes[1].fill_between(prof.index, prof["nat_demand"], prof["cf_mean"], color="#E11D48", alpha=0.13, label="Listrik yang Berkurang")
    axes[1].set_title(f"B. Pergeseran Pola 24 Jam ({month_names[int(month_focus)]} 2020 · {day_type})")
    axes[1].set_xlabel("Jam (00:00–23:00)")
    axes[1].set_ylabel("Rata-rata Beban (MWh)")
    axes[1].set_xticks(range(0, 24, 3))
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_mpl_weather_and_holiday(gam_curves, pdp_t2m, holiday_labels, top_n=14):
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.0))

    t = gam_curves["temp"]
    axes[0, 0].plot(t["x"], t["mean"], color="#3B6EF6", lw=2.1)
    axes[0, 0].fill_between(t["x"], t["ci_lo"], t["ci_hi"], color="#3B6EF6", alpha=0.14)
    axes[0, 0].set_title("1. Kenaikan Beban Saat Cuaca Makin Panas (GAM)")
    axes[0, 0].set_xlabel("Indeks Suhu Gabungan 3 Kota")
    axes[0, 0].set_ylabel("Prediksi Beban (MWh)")

    axes[0, 1].plot(pdp_t2m["x"], pdp_t2m["y"], color="#0D9488", lw=2.1)
    axes[0, 1].set_title("2. Efek Suhu Kota Panama terhadap Listrik (°C)")
    axes[0, 1].set_xlabel("Suhu Kota Panama (°C)")
    axes[0, 1].set_ylabel("Rata-rata Efek Prediksi (MWh)")

    h = gam_curves["hour"]
    axes[1, 0].plot(h["x"], h["mean"], color="#1E293B", lw=2.1)
    axes[1, 0].fill_between(h["x"], h["ci_lo"], h["ci_hi"], color="#94A3B8", alpha=0.20)
    axes[1, 0].set_title("3. Pola Naik-Turun Listrik dari Pagi ke Malam")
    axes[1, 0].set_xlabel("Jam (00:00–23:00)")
    axes[1, 0].set_ylabel("Prediksi Beban (MWh)")
    axes[1, 0].set_xticks(range(0, 24, 3))

    hdf = pd.DataFrame(gam_curves["holiday_coefs"])
    hdf["Label"] = [holiday_labels.get(int(i), str(i)).split(" - ", 1)[-1][:34] for i in hdf["Holiday_ID"]]
    hdf = hdf.reindex(hdf["Efek_MWh"].abs().sort_values(ascending=True).index).tail(top_n)
    clrs = ["#3B6EF6" if v < 0 else "#0D9488" for v in hdf["Efek_MWh"]]
    axes[1, 1].barh(hdf["Label"], hdf["Efek_MWh"], color=clrs, height=0.68)
    axes[1, 1].axvline(0, color="#1E293B", lw=0.8)
    axes[1, 1].set_title(f"4. Hari Libur yang Paling Menurunkan Beban Listrik (Top {top_n})")
    axes[1, 1].set_xlabel("Selisih vs Hari Biasa (MWh)")
    axes[1, 1].tick_params(axis="y", labelsize=8)
    axes[1, 1].grid(True, axis="x", color="#E2E8F0")
    axes[1, 1].grid(False, axis="y")

    plt.tight_layout()
    return fig


def plot_mpl_xai_importance(grouped_rows, top10_rows, method_label):
    gdf = pd.DataFrame(grouped_rows)
    tdf = pd.DataFrame(top10_rows)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.0))

    axes[0].barh(gdf["Kelompok"], gdf["Importance"], color="#3B6EF6", height=0.56)
    axes[0].set_title(f"A. Kontribusi Kelompok Fitur\n({method_label})")
    axes[0].grid(True, axis="x", color="#E2E8F0")
    axes[0].grid(False, axis="y")

    axes[1].barh(tdf["Fitur"], tdf["Importance"], color="#0D9488", height=0.60)
    axes[1].set_title("B. Top 10 Fitur Individual (Test 2019)")
    axes[1].grid(True, axis="x", color="#E2E8F0")
    axes[1].grid(False, axis="y")

    plt.tight_layout()
    return fig


def plot_mpl_st_daily_bars(st_daily_df):
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(st_daily_df))
    b1 = ax.bar(x - 0.18, st_daily_df["Mean_MWh"], width=0.35, color="#3B6EF6", label="Rata-rata Harian (MWh)")
    b2 = ax.bar(x + 0.18, st_daily_df["Peak_MWh"], width=0.35, color="#1E293B", label="Beban Puncak Harian (MWh)")
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 12, f"{h:.0f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['Tanggal'][5:]}\n({r['Hari'][:3]})" for _, r in st_daily_df.iterrows()], fontsize=8)
    ax.set_ylim(0, max(st_daily_df["Peak_MWh"]) * 1.15)
    ax.set_ylabel("Beban (MWh)")
    ax.set_title("Perbandingan Beban Rata-rata & Puncak per Hari")
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    return fig


def plot_mpl_lt_energy_bars(lt_monthly_rows, custom_monthly=None):
    ldf = pd.DataFrame(lt_monthly_rows)
    fig, ax = plt.subplots(figsize=(13.5, 4.0))
    x = np.arange(len(ldf))
    if custom_monthly is not None:
        w = 0.20
        ax.bar(x - 1.5 * w, ldf["Ref_2019_GWh"], width=w, color="#94A3B8", label="Aktual Bulan Sama 2019")
        ax.bar(x - 0.5 * w, ldf["Normal_Energi_GWh"], width=w, color="#3B6EF6", label="Skenario Pulih Normal")
        ax.bar(x + 0.5 * w, ldf["COVID_Energi_GWh"], width=w, color="#E11D48", label="Skenario Pandemi Berlanjut")
        ax.bar(x + 1.5 * w, custom_monthly["Custom_Energi_GWh"], width=w, color="#0D9488", label="Simulasi Pilihan Anda")
    else:
        w = 0.26
        ax.bar(x - w, ldf["Ref_2019_GWh"], width=w, color="#94A3B8", label="Aktual Bulan Sama 2019")
        ax.bar(x, ldf["Normal_Energi_GWh"], width=w, color="#3B6EF6", label="Skenario Pulih Normal")
        ax.bar(x + w, ldf["COVID_Energi_GWh"], width=w, color="#E11D48", label="Skenario Pandemi Berlanjut")
    ax.set_xticks(x)
    ax.set_xticklabels(ldf["Bulan"], rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("Total Energi (GWh)")
    ax.set_title("Total Energi Bulanan (GWh): Proyeksi 12 Bulan vs Tahun Normal 2019")
    ax.legend(fontsize=8.5, ncol=4, loc="upper right")
    ax.set_ylim(0, max(ldf["Normal_Energi_GWh"].max(), ldf["Ref_2019_GWh"].max()) * 1.18)
    plt.tight_layout()
    return fig


def plot_mpl_whatif_delta_and_temp(sim_df, baseline_hourly_2019, hour_focus):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.5))

    base_y = np.array([baseline_hourly_2019.get(int(h), np.nan) for h in sim_df["hour"]])
    delta = sim_df["pred_mean"].values - base_y
    bar_clrs = ["#E11D48" if d < 0 else "#0D9488" for d in delta]
    axes[0].bar(sim_df["hour"], delta, color=bar_clrs, width=0.72, alpha=0.88)
    axes[0].axhline(0, color="#1E293B", lw=0.8)
    axes[0].axvline(hour_focus, color="#64748B", ls="--", lw=1.1, label=f"Jam Fokus {int(hour_focus):02d}:00")
    axes[0].set_title("A. Selisih Prediksi vs Hari Normal 2019 (MWh)")
    axes[0].set_xlabel("Jam (00:00–23:00)")
    axes[0].set_ylabel("Selisih (MWh)")
    axes[0].set_xticks(range(0, 24, 3))
    axes[0].legend(fontsize=8)

    axes[1].plot(sim_df["hour"], sim_df["T2M_toc"], label="Tocumen (Kota Panama)", color="#1E293B", lw=1.8)
    axes[1].plot(sim_df["hour"], sim_df["T2M_san"], label="Santiago", color="#3B6EF6", lw=1.5, ls="--")
    axes[1].plot(sim_df["hour"], sim_df["T2M_dav"], label="David", color="#64748B", lw=1.5, ls="-.")
    axes[1].axvline(hour_focus, color="#94A3B8", ls=":", lw=1.0)
    axes[1].set_title("B. Kurva Suhu Harian di 3 Kota pada Skenario Ini")
    axes[1].set_xlabel("Jam (00:00–23:00)")
    axes[1].set_ylabel("Suhu (°C)")
    axes[1].set_xticks(range(0, 24, 3))
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    return fig
