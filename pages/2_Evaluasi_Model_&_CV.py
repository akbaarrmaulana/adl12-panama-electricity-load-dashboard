import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit_shadcn_ui as ui
from utils.charts import (
    make_weekly_zoom_chart,
    plot_mpl_backtest_and_acf_comparison,
    plot_mpl_sarima_inadequacy_3panel,
    plot_mpl_tscv_3fold,
)
from utils.data_loader import load_audit_stats, load_master_hourly
from utils.ui_components import render_methodology_drawer, render_page_header, render_shadcn_kpis

df = load_master_hourly()
audit = load_audit_stats()

render_page_header(
    "Evaluasi Model & Cross-Validation",
    "Perbandingan performa backtesting 5 model (metode dasar vs solusi), validasi TimeSeriesSplit 3-fold, dan audit diagnostik SARIMA/SARIMAX.",
)

summary_df = pd.DataFrame(audit["backtest_summary"])
cv_gam_df = pd.DataFrame(audit["cv_gam"])
cv_gbm_df = pd.DataFrame(audit["cv_gbm"])

tab_options = [
    "Perbandingan 5 Model & Diebold-Mariano",
    "Cross-Validation (3-Fold)",
    "Diagnostik SARIMA / SARIMAX",
]
active_tab = ui.tabs(options=tab_options, value=tab_options[0], key="eval_tabs")
st.write("")

if active_tab == tab_options[0]:
    c1, c2, c3 = st.columns([3.5, 2.0, 2.5])
    model_opts = {
        "Seasonal Naive (Y_t-168)": "pred_naive",
        "SARIMA(1,0,1)(1,1,0)_24": "pred_sarima",
        "SARIMAX (Linear)": "pred_sarimax",
        "GAM (Set A Eksogen)": "pred_gam",
        f"{audit['dataset_meta']['gbm_engine']} (Set B h=168)": "pred_gbm",
    }
    chosen_labels = c1.multiselect("Model pada Grafik Mingguan", list(model_opts.keys()), default=list(model_opts.keys()))
    eval_mode = c2.radio("Evaluasi 2019", ["Tanpa 5 Titik Blackout", "Semua Jam (All)"], horizontal=True)
    show_pi = c3.checkbox("Tampilkan Interval 90% Boosting", value=True)

    use_no_bo = eval_mode == "Tanpa 5 Titik Blackout"
    selected_cols = [model_opts[lbl] for lbl in chosen_labels]

    fig_cmp = plot_mpl_backtest_and_acf_comparison(summary_df, audit["acf_errors_2019"], use_no_bo=use_no_bo)
    st.pyplot(fig_cmp)
    plt.close(fig_cmp)

    df_test_period = df[df["datetime"] >= "2019-01-01 00:00:00"].copy().reset_index(drop=True)
    n_weeks = len(df_test_period) // 168
    week_idx = st.slider("Zoom Minggu Uji (2019–Jun 2020)", min_value=0, max_value=n_weeks - 1, value=19)
    w_slice = df_test_period.iloc[week_idx * 168 : (week_idx + 1) * 168]
    st.caption(f"Jendela minggu terpilih: {w_slice['datetime'].min()} s.d. {w_slice['datetime'].max()}")

    st.plotly_chart(make_weekly_zoom_chart(w_slice, selected_cols, show_pi=show_pi), width="stretch", theme=None, config={"displaylogo": False})

    st.subheader("Tabel Performa Komparatif & Uji Diebold-Mariano")
    disp_cols = [
        "Model", "Paradigma", "Kategori_Forecast",
        "MAE_2019_All", "RMSE_2019_All", "MAE_2019_NoBO", "RMSE_2019_NoBO",
        "MAPE_2019_NoBO_%", "MAPE_MarJun2020_%", "LjungBox_lag168_p", "ARCH_LM_p",
        "DM_vs_SARIMAX", "DM_vs_Naive",
    ]
    st.dataframe(summary_df[disp_cols], width="stretch", hide_index=True)

    st.markdown("**Evaluasi Probabilistik Interval 90% (`HistGradientBoosting` Set B)**")
    st.dataframe(pd.DataFrame(audit["gbm_probabilistic"]), width="stretch", hide_index=True)

elif active_tab == tab_options[1]:
    render_shadcn_kpis([
        {
            "label": "CV GAM (3-Fold Mean RMSE)",
            "value": f"{cv_gam_df['RMSE'].mean():.2f} MWh",
            "description": f"Mean MAPE {cv_gam_df['MAPE_%'].mean():.2f}% (Set A Eksogen)",
        },
        {
            "label": "CV Boosting (3-Fold Mean RMSE)",
            "value": f"{cv_gbm_df['RMSE'].mean():.2f} MWh",
            "description": f"Mean MAPE {cv_gbm_df['MAPE_%'].mean():.2f}% (Set B h=168)",
        },
    ], key_prefix="cv_kpi")

    fig_cv = plot_mpl_tscv_3fold(audit["cv_gam"], audit["cv_gbm"])
    st.pyplot(fig_cv)
    plt.close(fig_cv)

    c_gam, c_gbm = st.columns(2, gap="large")
    with c_gam:
        st.markdown("**Tabel Fold CV — GAM (Set A Eksogen pada `2015–2018`)**")
        st.dataframe(cv_gam_df, width="stretch", hide_index=True)
    with c_gbm:
        st.markdown(f"**Tabel Fold CV — {audit['dataset_meta']['gbm_engine']} (Set B pada `2015–2018`)**")
        st.dataframe(cv_gbm_df, width="stretch", hide_index=True)

else:
    st.caption(f"AIC SARIMA: {audit['sarima_aic']:,.1f} | AIC SARIMAX: {audit['sarimax_aic']:,.1f} — model S=24 membocorkan autokorelasi mingguan ke lag 168.")
    df_19 = df[df["year"] == 2019]
    apr20 = df[(df["datetime"] >= "2020-04-01") & (df["datetime"] <= "2020-04-21")]
    fig_inad = plot_mpl_sarima_inadequacy_3panel(df_19, apr20, audit["sarimax_hourly_rmse"])
    st.pyplot(fig_inad)
    plt.close(fig_inad)

    c_t1, c_t2 = st.columns(2, gap="large")
    with c_t1:
        st.markdown("**Uji Diagnostik Residual (Ljung-Box, Jarque-Bera, ARCH-LM)**")
        st.dataframe(pd.DataFrame(audit["sarima_diagnostics"]), width="stretch", hide_index=True)
    with c_t2:
        st.markdown("**Error Breakdown SARIMAX 2019: Hari Biasa vs Hari Libur**")
        st.dataframe(pd.DataFrame(audit["sarimax_holiday_breakdown"]), width="stretch", hide_index=True)

render_methodology_drawer(
    "Formulasi Matematis Uji Diebold-Mariano & TimeSeriesSplit",
    """
- **TimeSeriesSplit (Expanding Window):** Latih pada rentang awal yang terus membesar ($\\mathcal{T}_1 \\subset \\mathcal{T}_2 \\subset \\mathcal{T}_3$) lalu diuji secara kronologis pada tahun berikutnya (`2016`, `2017`, `2018`) di dalam blok latih `2015–2018`.
- **Diebold-Mariano Test (Newey-West HAC):** Menguji selisih kerugian kuadratik $d_t = e_{\\text{Ref},t}^2 - e_{m,t}^2$ menggunakan estimator varians jangka panjang Bartlett kernel hingga lag $K = h - 1 = 167$.
""",
)
