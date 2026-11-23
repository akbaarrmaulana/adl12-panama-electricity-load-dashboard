"""
Restrained Clean SaaS Visualization Library (Hybrid Matplotlib + Plotly):
Palette discipline:
- Slate/Ink (#0F172A, #64748B, #94A3B8) for base series & structure
- Primary Blue (#2563EB) for main model/GAM & reference trends
- Muted Emerald (#059669) for STLF Boosting solution
- Semantic Red (#DC2626) strictly for disruptions, anomalies, or threshold alerts
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plt.rcParams.update({
    "figure.dpi": 135,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#E2E8F0",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#F1F5F9",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": "#0F172A",
    "axes.labelcolor": "#334155",
    "xtick.color": "#64748B",
    "ytick.color": "#64748B",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "legend.frameon": False,
})

COLORS = {
    "actual": "#0F172A",
    "blue": "#2563EB",
    "blue_fill": "rgba(37, 99, 235, 0.12)",
    "emerald": "#059669",
    "emerald_fill": "rgba(5, 150, 105, 0.14)",
    "red": "#DC2626",
    "red_fill": "rgba(220, 38, 38, 0.10)",
    "muted": "#64748B",
    "light_gray": "#94A3B8",
    "grid": "#F1F5F9",
    "border": "#E2E8F0",
}

PANAMA_STATIONS = [
    {"city": "Tocumen (Kota Panama)", "code": "TOC", "lat": 9.0714, "lon": -79.3835, "t2m_col": "T2M_toc"},
    {"city": "Santiago (Veraguas)", "code": "SAN", "lat": 8.1000, "lon": -80.9833, "t2m_col": "T2M_san"},
    {"city": "David (Chiriquí)", "code": "DAV", "lat": 8.4273, "lon": -82.4308, "t2m_col": "T2M_dav"},
]


# ==============================================================================
# CLEAN PLOTLY HELPERS (Interactive Time-Series & Panama Spatial Map)
# ==============================================================================
def apply_clean_plotly_layout(fig, title="", height=380, y_title="", x_title="", show_legend=True, margin=None):
    m = margin or dict(l=52, r=20, t=68 if title else 40, b=38)
    fig.update_layout(
        title=dict(text=title, font=dict(family="Inter, sans-serif", size=14, color="#0F172A"), x=0, xanchor="left", y=0.96),
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=12, color="#334155"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#CBD5E1", font=dict(size=11, color="#0F172A")),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, bgcolor="rgba(255,255,255,0.9)", font=dict(size=11, color="#475569")),
        margin=m,
    )
    fig.update_xaxes(title_text=x_title, showgrid=False, zeroline=False, linecolor="#E2E8F0", tickfont=dict(size=10, color="#64748B"))
    fig.update_yaxes(title_text=y_title, showgrid=True, gridcolor="#F1F5F9", zeroline=False, linecolor="#E2E8F0", tickfont=dict(size=10, color="#64748B"))
    return fig


def make_panama_station_map(df_master, vif_rows, metric_mode="Korelasi Beban (r)"):
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
        cscale = [[0.0, "#93C5FD"], [1.0, "#2563EB"]]
    elif metric_mode == "Rata-rata Suhu (°C)":
        color_vals, ctitle = sdf["mean_t2m"], "°C"
        cscale = [[0.0, "#93C5FD"], [1.0, "#2563EB"]]
    else:
        color_vals, ctitle = sdf["corr"], "r"
        cscale = [[0.0, "#6EE7B7"], [1.0, "#059669"]]

    hover_text = [
        f"<b>{r['city']}</b><br>"
        f"Koordinat: {r['lat']:.2f}°N, {abs(r['lon']):.2f}°W<br>"
        f"Rata-rata T2M: {r['mean_t2m']:.2f} °C<br>"
        f"Korelasi vs Beban: r = {r['corr']:.4f}<br>"
        f"VIF Spasial: {r['vif']:.2f}"
        for _, r in sdf.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=sdf["lon"], lat=sdf["lat"], mode="lines",
        line=dict(width=1.5, color="rgba(37, 99, 235, 0.35)", dash="dot"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scattergeo(
        lon=sdf["lon"], lat=sdf["lat"],
        mode="markers+text",
        text=[f"{r['code']} ({r['mean_t2m']:.1f}°C)" for _, r in sdf.iterrows()],
        textposition=["top right", "bottom center", "top left"],
        textfont=dict(size=11, color="#0F172A"),
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(
            size=[15, 14, 14],
            color=color_vals,
            colorscale=cscale,
            showscale=True,
            colorbar=dict(title=dict(text=ctitle, font=dict(size=10)), tickfont=dict(size=9), len=0.72, thickness=9),
            line=dict(width=1.8, color="#FFFFFF"),
        ),
    ))
    fig.update_geos(
        resolution=50,
        showcoastlines=True, coastlinecolor="#CBD5E1",
        showcountries=True, countrycolor="#CBD5E1",
        showland=True, landcolor="#F8FAFC",
        showocean=True, oceancolor="#EFF6FF",
        lataxis=dict(range=[6.8, 10.1]),
        lonaxis=dict(range=[-83.4, -77.3]),
        bgcolor="#FFFFFF",
    )
    fig.update_layout(
        title=dict(text="Peta 3 Stasiun Cuaca Panama", font=dict(family="Inter, sans-serif", size=14, color="#0F172A"), x=0),
        height=350,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=4, r=4, t=42, b=4),
        showlegend=False,
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#CBD5E1", font=dict(size=11, color="#0F172A")),
    )
    return fig


def make_regime_timeline(df_master, regime_filter="Semua (2015–2020)"):
    daily = df_master.set_index("datetime")["nat_demand"].resample("D").mean().reset_index()
    bo = df_master[df_master["is_blackout"] == 1]

    if regime_filter == "Training (2015–2018)":
        daily = daily[daily["datetime"] <= "2018-12-31 23:00:00"]
        bo = bo[bo["datetime"] <= "2018-12-31 23:00:00"]
    elif regime_filter == "Test Normal (2019)":
        daily = daily[(daily["datetime"] >= "2019-01-01") & (daily["datetime"] <= "2019-12-31 23:00:00")]
        bo = bo[(bo["datetime"] >= "2019-01-01") & (bo["datetime"] <= "2019-12-31 23:00:00")]
    elif regime_filter == "Test Disrupsi (2020)":
        daily = daily[daily["datetime"] >= "2020-01-01"]
        bo = bo[bo["datetime"] >= "2020-01-01"]

    fig = go.Figure()
    if regime_filter == "Semua (2015–2020)":
        fig.add_vrect(x0="2015-01-03", x1="2018-12-31", fillcolor="rgba(148, 163, 184, 0.08)", line_width=0, annotation_text="Train (2015–2018)", annotation_position="top left", annotation_font=dict(size=10, color="#64748B"))
        fig.add_vrect(x0="2019-01-01", x1="2019-12-31", fillcolor="rgba(37, 99, 235, 0.06)", line_width=0, annotation_text="Test 2019", annotation_position="top left", annotation_font=dict(size=10, color="#2563EB"))
        fig.add_vrect(x0="2020-03-01", x1="2020-06-27", fillcolor="rgba(220, 38, 38, 0.08)", line_width=0, annotation_text="COVID-19", annotation_position="top left", annotation_font=dict(size=10, color="#DC2626"))

    fig.add_trace(go.Scatter(
        x=daily["datetime"], y=daily["nat_demand"], mode="lines",
        line=dict(color=COLORS["blue"], width=1.5),
        name="Rata-rata Harian (MWh)",
    ))
    if len(bo) > 0:
        fig.add_trace(go.Scatter(
            x=bo["datetime"], y=bo["nat_demand"], mode="markers",
            marker=dict(color=COLORS["red"], size=7, symbol="diamond", line=dict(width=1, color="#FFFFFF")),
            name="Blackout (<600 MWh)",
        ))
    return apply_clean_plotly_layout(fig, title="Beban Listrik Harian Nasional Panama (2015–2020)", height=350, y_title="MWh")


def make_dual_axis_explorer(df_sub, weather_col, weather_label, show_blackout=True, show_lockdown=True):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df_sub["datetime"], y=df_sub["nat_demand"], name="Beban Nasional (MWh)", line=dict(color=COLORS["blue"], width=1.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_sub["datetime"], y=df_sub[weather_col], name=weather_label, line=dict(color=COLORS["muted"], width=1.2), opacity=0.8), secondary_y=True)
    if show_blackout:
        bo = df_sub[df_sub["is_blackout"] == 1]
        if len(bo) > 0:
            fig.add_trace(go.Scatter(x=bo["datetime"], y=bo["nat_demand"], mode="markers", marker=dict(color=COLORS["red"], size=7, symbol="x"), name="Blackout (<600 MWh)"), secondary_y=False)
    if show_lockdown and df_sub["datetime"].max() >= pd.Timestamp("2020-03-01"):
        x0 = max(df_sub["datetime"].min(), pd.Timestamp("2020-03-01"))
        fig.add_vrect(x0=x0, x1=df_sub["datetime"].max(), fillcolor=COLORS["red_fill"], line_width=0)

    apply_clean_plotly_layout(fig, title=f"Beban Nasional vs {weather_label}", height=390, margin=dict(l=52, r=52, t=68, b=38))
    fig.update_yaxes(title_text="nat_demand (MWh)", secondary_y=False, gridcolor="#F1F5F9")
    fig.update_yaxes(title_text=weather_label, secondary_y=True, showgrid=False)
    return fig


def make_weekly_zoom_chart(df_week, selected_models, show_pi=True):
    col_style = {
        "pred_naive": ("Seasonal Naive (t-168)", COLORS["light_gray"], "dot"),
        "pred_sarima": ("SARIMA", "#94A3B8", "dash"),
        "pred_sarimax": ("SARIMAX", COLORS["red"], "dash"),
        "pred_gam": ("GAM (Set A)", COLORS["blue"], "solid"),
        "pred_gbm": ("HistGradientBoosting (Set B)", COLORS["emerald"], "solid"),
    }
    fig = go.Figure()
    if show_pi and "pred_gbm" in selected_models:
        fig.add_trace(go.Scatter(
            x=pd.concat([df_week["datetime"], df_week["datetime"][::-1]]),
            y=pd.concat([df_week["pred_gbm_q95"], df_week["pred_gbm_q05"][::-1]]),
            fill="toself", fillcolor=COLORS["emerald_fill"], line=dict(color="rgba(0,0,0,0)"),
            name="PI 90% Boosting (Q05–Q95)", hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(x=df_week["datetime"], y=df_week["nat_demand"], mode="lines", line=dict(color=COLORS["actual"], width=2.2), name="Aktual"))
    for col in selected_models:
        lbl, clr, dash = col_style[col]
        fig.add_trace(go.Scatter(x=df_week["datetime"], y=df_week[col], mode="lines", line=dict(color=clr, width=1.7, dash=dash), name=lbl))
    return apply_clean_plotly_layout(fig, title="Zoom Mingguan: Aktual vs Prediksi Model", height=390, y_title="MWh")


def make_st_forecast_chart(hist_7d, future_st, threshold_mwh=1250.0, show_pi=True, show_q50=True):
    fig = go.Figure()
    if show_pi:
        fig.add_trace(go.Scatter(
            x=pd.concat([future_st["datetime"], future_st["datetime"][::-1]]),
            y=pd.concat([future_st["pred_st_q95"], future_st["pred_st_q05"][::-1]]),
            fill="toself", fillcolor=COLORS["emerald_fill"], line=dict(color="rgba(0,0,0,0)"),
            name="Interval Prediksi 90% (Q05–Q95)", hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(x=hist_7d["datetime"], y=hist_7d["nat_demand"], name="Aktual 7 Hari Terakhir", line=dict(color=COLORS["actual"], width=2.0)))
    fig.add_trace(go.Scatter(x=future_st["datetime"], y=future_st["pred_st_mean"], name="Forecast STLF 168j", line=dict(color=COLORS["emerald"], width=2.2)))
    if show_q50:
        fig.add_trace(go.Scatter(x=future_st["datetime"], y=future_st["pred_st_q50"], name="Median (Q50)", line=dict(color=COLORS["blue"], width=1.3, dash="dot")))

    exceed = future_st[future_st["pred_st_mean"] > threshold_mwh]
    if len(exceed) > 0:
        fig.add_trace(go.Scatter(
            x=exceed["datetime"], y=exceed["pred_st_mean"], mode="markers",
            marker=dict(color=COLORS["red"], size=6, line=dict(width=1, color="#FFFFFF")),
            name=f"Melebihi Ambang ({threshold_mwh:,.0f} MWh)",
        ))

    cutoff_ms = hist_7d["datetime"].max().timestamp() * 1000
    fig.add_vline(x=cutoff_ms, line_dash="dash", line_color="#64748B", annotation_text="Akhir Dataset", annotation_font=dict(color="#64748B", size=10))
    fig.add_hline(y=threshold_mwh, line_dash="dot", line_color=COLORS["red"], annotation_text=f"Ambang {threshold_mwh:,.0f} MWh", annotation_font=dict(color=COLORS["red"], size=10))
    return apply_clean_plotly_layout(fig, title="Proyeksi Jangka Pendek 7 Hari / 168 Jam (27 Jun – 4 Jul 2020)", height=410, y_title="Beban (MWh)")


def make_lt_forecast_chart(hist_monthly, lt_monthly, show_normal=True, show_covid=True, custom_monthly=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_monthly["datetime"], y=hist_monthly["nat_demand"], name="Aktual Bulanan (2015–Jun 2020)", line=dict(color=COLORS["actual"], width=2.0)))

    if show_normal:
        fig.add_trace(go.Scatter(
            x=pd.concat([lt_monthly["datetime"], lt_monthly["datetime"][::-1]]),
            y=pd.concat([lt_monthly["lt_base_hi"], lt_monthly["lt_base_lo"][::-1]]),
            fill="toself", fillcolor=COLORS["blue_fill"], line=dict(color="rgba(0,0,0,0)"),
            name="95% CI Skenario Normal", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(x=lt_monthly["datetime"], y=lt_monthly["lt_base_mean"], mode="lines+markers", name="Skenario A: Normal", line=dict(color=COLORS["blue"], width=2.0, dash="dash"), marker=dict(size=5)))

    if show_covid:
        fig.add_trace(go.Scatter(
            x=pd.concat([lt_monthly["datetime"], lt_monthly["datetime"][::-1]]),
            y=pd.concat([lt_monthly["lt_cov_hi"], lt_monthly["lt_cov_lo"][::-1]]),
            fill="toself", fillcolor=COLORS["red_fill"], line=dict(color="rgba(0,0,0,0)"),
            name="95% CI Skenario COVID-19", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(x=lt_monthly["datetime"], y=lt_monthly["lt_cov_mean"], mode="lines+markers", name="Skenario B: COVID-19", line=dict(color=COLORS["red"], width=2.0, dash="dashdot"), marker=dict(size=5)))

    if custom_monthly is not None:
        fig.add_trace(go.Scatter(
            x=lt_monthly["datetime"], y=custom_monthly["Custom_Mean_MWh"], mode="lines+markers",
            name="Skenario Kustom", line=dict(color=COLORS["emerald"], width=2.5), marker=dict(size=6, symbol="diamond"),
        ))

    fig.add_vline(x=pd.Timestamp("2020-06-27").timestamp() * 1000, line_dash="dot", line_color="#64748B")
    apply_clean_plotly_layout(fig, title="Proyeksi Jangka Panjang 12 Bulan (Jul 2020 – Jun 2021)", height=400, y_title="Rata-rata Bulanan (MWh)")
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
    fig.add_trace(go.Scatter(x=sim_df["hour"], y=base_y, name="Baseline Normal 2019", line=dict(color="#94A3B8", width=1.8, dash="dot")))
    fig.add_trace(go.Scatter(x=sim_df["hour"], y=sim_df["pred_mean"], mode="lines+markers", name="Simulasi What-If", line=dict(color=COLORS["blue"], width=2.4), marker=dict(size=5)))

    focus_row = sim_df[sim_df["hour"] == float(hour_focus)].iloc[0]
    fig.add_trace(go.Scatter(
        x=[focus_row["hour"]], y=[focus_row["pred_mean"]], mode="markers",
        marker=dict(size=11, color=COLORS["emerald"], line=dict(width=2, color="#FFFFFF")),
        name=f"Jam Fokus ({int(hour_focus):02d}:00)",
    ))
    apply_clean_plotly_layout(fig, title="Simulasi Profil Beban 24 Jam vs Baseline Normal 2019", height=380, x_title="Jam (0–23)", y_title="Prediksi Beban (MWh)")
    fig.update_xaxes(dtick=2)
    fig.update_layout(uirevision="whatif_24h")
    return fig


# ==============================================================================
# PUBLICATION-GRADE MATPLOTLIB FIGURES
# ==============================================================================
def plot_mpl_intraday_profiles_and_heatmap(df_master, df_sub):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.3))

    slices = [
        ("2015", (df_master["year"] == 2015), "#94A3B8", "--", 1.4),
        ("2017", (df_master["year"] == 2017), "#64748B", "-.", 1.4),
        ("2019 (Normal)", (df_master["year"] == 2019), "#2563EB", "-", 2.1),
        ("Mar–Jun 2020 (COVID-19)", (df_master["datetime"] >= "2020-03-01"), "#DC2626", "-", 2.1),
    ]
    for lbl, msk, clr, ls, lw in slices:
        prof = df_master.loc[msk].groupby("hour")["nat_demand"].mean()
        axes[0].plot(prof.index, prof.values, label=lbl, color=clr, linestyle=ls, linewidth=lw)
    axes[0].set_title("A. Evolusi Profil Beban 24 Jam (2015–2020)")
    axes[0].set_xlabel("Jam (0–23)")
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
    axes[1].set_title("B. Heatmap Beban Intraharian (Periode Terpilih)")
    axes[1].grid(False)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("MWh", fontsize=8.5)

    plt.tight_layout()
    return fig


def plot_mpl_fft_and_acf_stem(fft_spectrum, acf_vals):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.1))

    axes[0].plot(fft_spectrum["periods"], fft_spectrum["power"], color="#2563EB", lw=1.4)
    axes[0].axvline(24, color="#DC2626", ls="--", lw=1.1, label="S1 = 24 jam (Harian)")
    axes[0].axvline(168, color="#64748B", ls=":", lw=1.3, label="S2 = 168 jam (Mingguan)")
    axes[0].set_title("A. Periodogram FFT: Musiman Ganda")
    axes[0].set_xlabel("Periode (Jam)")
    axes[0].set_ylabel("Spektral Power")
    axes[0].legend(fontsize=8.5)

    lags = np.arange(len(acf_vals))
    axes[1].stem(lags, acf_vals, linefmt="#2563EB", markerfmt=" ", basefmt="#0F172A")
    axes[1].axvline(24, color="#DC2626", ls="--", lw=1.1, label="Lag 24")
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
    clrs = ["#DC2626" if v > 10 else "#2563EB" for v in vdf["VIF"]]
    bars = ax.barh(vdf["Variabel_Cuaca"], vdf["VIF"], color=clrs, height=0.66)
    ax.axvline(10, color="#64748B", ls="--", lw=1.2, label="Ambang VIF = 10")
    for bar, v in zip(bars, vdf["VIF"]):
        ax.text(v + 0.35, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", fontsize=8, color="#0F172A")
    ax.set_xlim(0, max(vdf["VIF"]) * 1.15)
    ax.set_title("Audit Multikolinearitas Spasial 12 Fitur Cuaca (VIF)")
    ax.set_xlabel("Variance Inflation Factor (VIF)")
    ax.grid(True, axis="x", color="#F1F5F9")
    ax.grid(False, axis="y")
    ax.legend(fontsize=8.5, loc="lower right")
    plt.tight_layout()
    return fig


def plot_mpl_backtest_and_acf_comparison(summary_df, acf_err, use_no_bo=True):
    rmse_col = "RMSE_2019_NoBO" if use_no_bo else "RMSE_2019_All"
    mae_col = "MAE_2019_NoBO" if use_no_bo else "MAE_2019_All"
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.3))

    x = np.arange(len(summary_df))
    b1 = axes[0].bar(x - 0.18, summary_df[rmse_col], width=0.35, color="#2563EB", label="RMSE 2019 (MWh)")
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
    axes[1].plot(lags, acf_err["sarimax"], label="SARIMAX", color="#DC2626", lw=1.3)
    axes[1].plot(lags, acf_err["gam"], label="GAM (Set A)", color="#2563EB", lw=1.3)
    axes[1].plot(lags, acf_err["gbm"], label="HistGradientBoosting (Set B)", color="#059669", lw=1.4)
    axes[1].axvline(168, color="#64748B", ls="--", lw=1.1, label="Lag 168")
    axes[1].axhline(0, color="#0F172A", lw=0.7)
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
        axes[0].barh(f, tr_years[i], left=2015, height=0.44, color="#94A3B8", label="Blok Training" if i == 0 else "")
        axes[0].barh(f, 1, left=2015 + tr_years[i], height=0.44, color="#2563EB", label="Blok Validasi" if i == 0 else "")
        axes[0].text(2015 + tr_years[i] + 0.5, f, f"Val {2015 + tr_years[i]}", va="center", ha="center", color="#FFFFFF", fontsize=8, fontweight="bold")
    axes[0].set_yticks(folds)
    axes[0].set_yticklabels([f"Fold {f}" for f in folds])
    axes[0].invert_yaxis()
    axes[0].set_xlim(2014.9, 2019.2)
    axes[0].set_xticks([2015, 2016, 2017, 2018, 2019])
    axes[0].set_title("A. Skema Expanding-Window TimeSeriesSplit (2015–2018)")
    axes[0].grid(True, axis="x", color="#F1F5F9")
    axes[0].grid(False, axis="y")
    axes[0].legend(fontsize=8.5, loc="lower left")

    gam_r = [r["RMSE"] for r in cv_gam_rows]
    gbm_r = [r["RMSE"] for r in cv_gbm_rows]
    x = np.arange(len(folds))
    b1 = axes[1].bar(x - 0.17, gam_r, width=0.33, color="#2563EB", label="GAM (Set A)")
    b2 = axes[1].bar(x + 0.17, gbm_r, width=0.33, color="#059669", label="HistGradientBoosting (Set B)")
    for b in list(b1) + list(b2):
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width() / 2, h + 1.2, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Fold {f} ({2015+f})" for f in folds])
    axes[1].set_ylim(0, max(gam_r) * 1.2)
    axes[1].set_ylabel("RMSE Validasi (MWh)")
    axes[1].set_title("B. Stabilitas RMSE Antar-Fold CV (3-Fold)")
    axes[1].legend(fontsize=8.5)

    plt.tight_layout()
    return fig


def plot_mpl_sarima_inadequacy_3panel(df_19, df_apr, hourly_rmse):
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0))
    from statsmodels.tsa.stattools import acf as sm_acf

    err_19 = (df_19["nat_demand"] - df_19["pred_sarimax"]).values
    acf_err = sm_acf(err_19, nlags=180, fft=True)
    conf = 1.96 / np.sqrt(len(err_19))
    axes[0].stem(np.arange(len(acf_err)), acf_err, linefmt="#2563EB", markerfmt=" ", basefmt="#0F172A")
    axes[0].axhline(conf, color="#DC2626", ls="--", lw=0.9)
    axes[0].axhline(-conf, color="#DC2626", ls="--", lw=0.9)
    axes[0].axvline(168, color="#64748B", ls=":", lw=1.2, label="Lag 168")
    axes[0].set_title("1. ACF Error SARIMAX (2019)")
    axes[0].set_xlabel("Lag (Jam)")
    axes[0].legend(fontsize=8)

    axes[1].plot(df_apr["datetime"], df_apr["nat_demand"], label="Aktual Apr 2020", color="#0F172A", lw=1.3)
    axes[1].plot(df_apr["datetime"], df_apr["pred_sarimax"], label="Prediksi SARIMAX (168j)", color="#DC2626", ls="--", lw=1.2)
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
    axes[0].plot(d20.index, d20["nat_demand"], color="#0F172A", lw=1.6, label="Aktual 2020")
    axes[0].plot(d20.index, d20["cf_mean"], color="#2563EB", ls="--", lw=1.6, label="Counterfactual GAM (Tanpa COVID)")
    axes[0].fill_between(d20.index, d20["cf_ci_lo"], d20["cf_ci_hi"], color="#2563EB", alpha=0.12, label="95% CI")
    axes[0].axvline(pd.Timestamp("2020-03-01"), color="#DC2626", ls=":", lw=1.3, label="Awal Maret 2020")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[0].tick_params(axis="x", rotation=18, labelsize=8.5)
    axes[0].set_title("A. Aktual vs Counterfactual Harian (Jan–Jun 2020)")
    axes[0].set_ylabel("Rata-rata Harian (MWh)")
    axes[0].legend(fontsize=8, loc="lower left")

    is_wknd = 1 if day_type == "Akhir Pekan" else 0
    month_names = {1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni"}
    prof = df_20_full[(df_20_full["month"] == int(month_focus)) & (df_20_full["is_weekend"] == is_wknd)].groupby("hour")[["nat_demand", "cf_mean"]].mean()
    axes[1].plot(prof.index, prof["cf_mean"], color="#2563EB", ls="--", lw=2.0, label="Counterfactual (Tanpa COVID)")
    axes[1].plot(prof.index, prof["nat_demand"], color="#0F172A", lw=2.0, label="Aktual 2020")
    axes[1].fill_between(prof.index, prof["nat_demand"], prof["cf_mean"], color="#DC2626", alpha=0.12, label="Defisit Beban Lockdown")
    axes[1].set_title(f"B. Pergeseran Profil 24 Jam ({month_names[int(month_focus)]} 2020 · {day_type})")
    axes[1].set_xlabel("Jam (0–23)")
    axes[1].set_ylabel("Rata-rata Beban (MWh)")
    axes[1].set_xticks(range(0, 24, 3))
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_mpl_weather_and_holiday(gam_curves, pdp_t2m, holiday_labels, top_n=14):
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.0))

    t = gam_curves["temp"]
    axes[0, 0].plot(t["x"], t["mean"], color="#2563EB", lw=2.0)
    axes[0, 0].fill_between(t["x"], t["ci_lo"], t["ci_hi"], color="#2563EB", alpha=0.13)
    axes[0, 0].set_title("1. Respons Non-Linear Suhu GAM (T2M_pca1 + 95% CI)")
    axes[0, 0].set_xlabel("Indeks Suhu PCA 3 Kota (T2M_pca1)")
    axes[0, 0].set_ylabel("Prediksi Beban (MWh)")

    axes[0, 1].plot(pdp_t2m["x"], pdp_t2m["y"], color="#059669", lw=2.0)
    axes[0, 1].set_title("2. Partial Dependence Boosting: Suhu Tocumen (T2M_toc)")
    axes[0, 1].set_xlabel("T2M_toc (°C)")
    axes[0, 1].set_ylabel("Rata-rata Efek Prediksi (MWh)")

    h = gam_curves["hour"]
    axes[1, 0].plot(h["x"], h["mean"], color="#0F172A", lw=2.0)
    axes[1, 0].fill_between(h["x"], h["ci_lo"], h["ci_hi"], color="#94A3B8", alpha=0.18)
    axes[1, 0].set_title("3. Kurva Siklus Harian GAM (hour + 95% CI)")
    axes[1, 0].set_xlabel("Jam (0–23)")
    axes[1, 0].set_ylabel("Prediksi Beban (MWh)")
    axes[1, 0].set_xticks(range(0, 24, 3))

    hdf = pd.DataFrame(gam_curves["holiday_coefs"])
    hdf["Label"] = [holiday_labels.get(int(i), str(i))[:34] for i in hdf["Holiday_ID"]]
    hdf = hdf.reindex(hdf["Efek_MWh"].abs().sort_values(ascending=True).index).tail(top_n)
    clrs = ["#2563EB" if v < 0 else "#059669" for v in hdf["Efek_MWh"]]
    axes[1, 1].barh(hdf["Label"], hdf["Efek_MWh"], color=clrs, height=0.68)
    axes[1, 1].axvline(0, color="#0F172A", lw=0.8)
    axes[1, 1].set_title(f"4. Efek Spesifik Holiday_ID (Top {top_n}, vs Hari Biasa)")
    axes[1, 1].set_xlabel("Perubahan Beban (MWh)")
    axes[1, 1].tick_params(axis="y", labelsize=8)
    axes[1, 1].grid(True, axis="x", color="#F1F5F9")
    axes[1, 1].grid(False, axis="y")

    plt.tight_layout()
    return fig


def plot_mpl_xai_importance(grouped_rows, top10_rows, method_label):
    gdf = pd.DataFrame(grouped_rows)
    tdf = pd.DataFrame(top10_rows)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.0))

    axes[0].barh(gdf["Kelompok"], gdf["Importance"], color="#2563EB", height=0.56)
    axes[0].set_title(f"A. Kontribusi Kelompok Fitur\n({method_label})")
    axes[0].grid(True, axis="x", color="#F1F5F9")
    axes[0].grid(False, axis="y")

    axes[1].barh(tdf["Fitur"], tdf["Importance"], color="#059669", height=0.60)
    axes[1].set_title("B. Top 10 Fitur Individual (Test 2019)")
    axes[1].grid(True, axis="x", color="#F1F5F9")
    axes[1].grid(False, axis="y")

    plt.tight_layout()
    return fig


def plot_mpl_st_daily_bars(st_daily_df):
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(st_daily_df))
    b1 = ax.bar(x - 0.18, st_daily_df["Mean_MWh"], width=0.35, color="#2563EB", label="Rata-rata Harian (MWh)")
    b2 = ax.bar(x + 0.18, st_daily_df["Peak_MWh"], width=0.35, color="#0F172A", label="Puncak Harian (MWh)")
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 12, f"{h:.0f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['Tanggal'][5:]}\n({r['Hari'][:3]})" for _, r in st_daily_df.iterrows()], fontsize=8)
    ax.set_ylim(0, max(st_daily_df["Peak_MWh"]) * 1.15)
    ax.set_ylabel("Beban (MWh)")
    ax.set_title("Profil Rata-rata & Puncak Harian Forecast 7 Hari")
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
        ax.bar(x - 0.5 * w, ldf["Normal_Energi_GWh"], width=w, color="#2563EB", label="Skenario A: Normal")
        ax.bar(x + 0.5 * w, ldf["COVID_Energi_GWh"], width=w, color="#DC2626", label="Skenario B: COVID-19")
        ax.bar(x + 1.5 * w, custom_monthly["Custom_Energi_GWh"], width=w, color="#059669", label="Skenario Kustom")
    else:
        w = 0.26
        ax.bar(x - w, ldf["Ref_2019_GWh"], width=w, color="#94A3B8", label="Aktual Bulan Sama 2019")
        ax.bar(x, ldf["Normal_Energi_GWh"], width=w, color="#2563EB", label="Skenario A: Normal")
        ax.bar(x + w, ldf["COVID_Energi_GWh"], width=w, color="#DC2626", label="Skenario B: COVID-19")
    ax.set_xticks(x)
    ax.set_xticklabels(ldf["Bulan"], rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("Total Energi (GWh)")
    ax.set_title("Total Energi Bulanan (GWh): Proyeksi 12 Bulan vs Referensi 2019")
    ax.legend(fontsize=8.5, ncol=4, loc="upper right")
    ax.set_ylim(0, max(ldf["Normal_Energi_GWh"].max(), ldf["Ref_2019_GWh"].max()) * 1.18)
    plt.tight_layout()
    return fig


def plot_mpl_whatif_delta_and_temp(sim_df, baseline_hourly_2019, hour_focus):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.5))

    base_y = np.array([baseline_hourly_2019.get(int(h), np.nan) for h in sim_df["hour"]])
    delta = sim_df["pred_mean"].values - base_y
    bar_clrs = ["#DC2626" if d < 0 else "#059669" for d in delta]
    axes[0].bar(sim_df["hour"], delta, color=bar_clrs, width=0.72, alpha=0.88)
    axes[0].axhline(0, color="#0F172A", lw=0.8)
    axes[0].axvline(hour_focus, color="#64748B", ls="--", lw=1.1, label=f"Jam Fokus {int(hour_focus):02d}:00")
    axes[0].set_title("A. Deviasi Beban Simulasi vs Normal 2019 (MWh)")
    axes[0].set_xlabel("Jam (0–23)")
    axes[0].set_ylabel("Delta (MWh)")
    axes[0].set_xticks(range(0, 24, 3))
    axes[0].legend(fontsize=8)

    axes[1].plot(sim_df["hour"], sim_df["T2M_toc"], label="Tocumen", color="#0F172A", lw=1.8)
    axes[1].plot(sim_df["hour"], sim_df["T2M_san"], label="Santiago", color="#2563EB", lw=1.5, ls="--")
    axes[1].plot(sim_df["hour"], sim_df["T2M_dav"], label="David", color="#64748B", lw=1.5, ls="-.")
    axes[1].axvline(hour_focus, color="#94A3B8", ls=":", lw=1.0)
    axes[1].set_title("B. Profil Suhu Diurnal 3 Kota pada Skenario")
    axes[1].set_xlabel("Jam (0–23)")
    axes[1].set_ylabel("Suhu (°C)")
    axes[1].set_xticks(range(0, 24, 3))
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    return fig
