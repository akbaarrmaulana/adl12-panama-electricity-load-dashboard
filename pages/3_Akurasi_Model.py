import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from utils.charts import (
    make_weekly_zoom_chart,
    plot_mpl_backtest_and_acf_comparison,
    plot_mpl_model_accuracy_layman,
    plot_mpl_sarima_inadequacy_3panel,
    plot_mpl_tscv_3fold,
)
from utils.data_loader import load_audit_stats, load_master_hourly
from utils.ui_components import inject_styles, render_insight_box, render_neu_kpis, render_page_header

inject_styles()

df = load_master_hourly()
audit = load_audit_stats()

render_page_header(
    "Perbandingan Akurasi Model Prediksi",
    "Dari 5 metode yang diuji pada data normal 2019, mana yang paling tepat dan kapan masing-masing sebaiknya digunakan?",
    kicker="Evaluasi Model",
)

summary_df = pd.DataFrame(audit["backtest_summary"])
cv_gam_df = pd.DataFrame(audit["cv_gam"])
cv_gbm_df = pd.DataFrame(audit["cv_gbm"])

gbm_row = summary_df[summary_df["col_key"] == "pred_gbm"].iloc[0]
gam_row = summary_df[summary_df["col_key"] == "pred_gam"].iloc[0]
sx_row = summary_df[summary_df["col_key"] == "pred_sarimax"].iloc[0]
sr_row = summary_df[summary_df["col_key"] == "pred_sarima"].iloc[0]

render_neu_kpis([
    {
        "label": "Terbaik Jangka Pendek (7 Hari)",
        "value": f"Boosting ({100.0 - gbm_row['MAPE_2019_NoBO_%']:.2f}% Akurat)",
        "hint": f"Meleset rata-rata {gbm_row['MAPE_2019_NoBO_%']:.2f}% ({gbm_row['RMSE_2019_NoBO']:.2f} MWh)",
        "badge": "Juara 1 · Operasional",
        "tone": "teal",
    },
    {
        "label": "Terbaik Jangka Panjang (12 Bulan)",
        "value": f"GAM ({100.0 - gam_row['MAPE_2019_NoBO_%']:.2f}% Akurat)",
        "hint": f"Meleset {gam_row['MAPE_2019_NoBO_%']:.2f}% ({gam_row['RMSE_2019_NoBO']:.2f} MWh) tanpa butuh data minggu lalu",
        "badge": "Juara 2 · Perencanaan",
        "tone": "blue",
    },
    {
        "label": "Metode Statistik Klasik",
        "value": "SARIMA / SARIMAX",
        "hint": f"Meleset {sx_row['MAPE_2019_NoBO_%']:.2f}%–{sr_row['MAPE_2019_NoBO_%']:.2f}% ({sx_row['RMSE_2019_NoBO']:.1f}–{sr_row['RMSE_2019_NoBO']:.1f} MWh)",
        "badge": "Pembanding Dasar",
        "tone": "neutral",
    },
])

render_insight_box(
    "Mengapa Tidak Cukup Satu Model?",
    "Model Boosting sangat presisi untuk 7 hari ke depan karena memanfaatkan catatan listrik minggu lalu. Namun untuk prediksi 1 tahun ke depan, catatan minggu lalu belum ada—sehingga model GAM (berbasis cuaca & kalender) menjadi solusi paling akurat.",
    tone="teal",
)

with st.container(border=True):
    include_bo = st.toggle("Sertakan 5 Jam Mati Lampu (Blackout 20 Jan 2019) dalam Perhitungan Error", value=False)
    use_no_bo = not include_bo
    fig_rank = plot_mpl_model_accuracy_layman(summary_df, use_no_bo=use_no_bo)
    st.pyplot(fig_rank)
    plt.close(fig_rank)

with st.container(border=True):
    st.markdown("**Uji Coba Visual per Minggu: Garis Aktual vs Prediksi Model**")
    model_opts = {
        "HistGradientBoosting (Solusi Terbaik 7 Hari)": "pred_gbm",
        "GAM (Pola Cuaca & Kalender)": "pred_gam",
        "Patokan Minggu Lalu (Seasonal Naive)": "pred_naive",
        "Statistik Linear + Cuaca (SARIMAX)": "pred_sarimax",
        "Statistik Klasik Tanpa Cuaca (SARIMA)": "pred_sarima",
    }
    default_models = [
        "GAM (Pola Cuaca & Kalender)",
        "HistGradientBoosting (Solusi Terbaik 7 Hari)",
    ]
    chosen_labels = st.multiselect("Pilih Model yang Ditampilkan", list(model_opts.keys()), default=default_models)
    selected_cols = [model_opts[lbl] for lbl in chosen_labels]

    df_test_period = df[df["datetime"] >= "2019-01-01 00:00:00"].copy().reset_index(drop=True)
    n_weeks = len(df_test_period) // 168
    week_idx = st.slider("Geser untuk Memilih Minggu Uji (2019 – Juni 2020)", min_value=0, max_value=n_weeks - 1, value=19)
    w_slice = df_test_period.iloc[week_idx * 168 : (week_idx + 1) * 168]
    st.caption(f"Minggu terpilih: {w_slice['datetime'].min().strftime('%d %b %Y')} s.d. {w_slice['datetime'].max().strftime('%d %b %Y')}")

    st.plotly_chart(
        make_weekly_zoom_chart(w_slice, selected_cols, show_pi=True),
        width="stretch",
        theme=None,
        config={"displaylogo": False},
    )

with st.expander("Lihat Hasil Uji Konsistensi Antar-Tahun (Cross-Validation 3-Fold)"):
    st.caption(
        f"Rata-rata RMSE pada data latih 2015–2018: GAM = {cv_gam_df['RMSE'].mean():.2f} MWh ({cv_gam_df['MAPE_%'].mean():.2f}%) "
        f"| Boosting = {cv_gbm_df['RMSE'].mean():.2f} MWh ({cv_gbm_df['MAPE_%'].mean():.2f}%)."
    )
    fig_cv = plot_mpl_tscv_3fold(audit["cv_gam"], audit["cv_gbm"])
    st.pyplot(fig_cv)
    plt.close(fig_cv)

    c_gam, c_gbm = st.columns(2, gap="large")
    with c_gam:
        st.markdown("**Rincian Fold — GAM (Set A Eksogen)**")
        st.dataframe(cv_gam_df, width="stretch", hide_index=True)
    with c_gbm:
        st.markdown(f"**Rincian Fold — {audit['dataset_meta']['gbm_engine']} (Set B)**")
        st.dataframe(cv_gbm_df, width="stretch", hide_index=True)

with st.expander("Lihat Detail Statistik Akademik (Uji Diebold-Mariano, ACF Error, Interval 90%, & Diagnostik SARIMA)"):
    fig_cmp = plot_mpl_backtest_and_acf_comparison(summary_df, audit["acf_errors_2019"], use_no_bo=use_no_bo)
    st.pyplot(fig_cmp)
    plt.close(fig_cmp)

    disp_cols = [
        "Model", "Paradigma", "Kategori_Forecast",
        "MAE_2019_All", "RMSE_2019_All", "MAE_2019_NoBO", "RMSE_2019_NoBO",
        "MAPE_2019_NoBO_%", "MAPE_MarJun2020_%", "LjungBox_lag168_p", "ARCH_LM_p",
        "DM_vs_SARIMAX", "DM_vs_Naive",
    ]
    st.dataframe(summary_df[disp_cols], width="stretch", hide_index=True)

    st.markdown("**Evaluasi Probabilistik Interval 90% (`HistGradientBoosting` Set B)**")
    st.dataframe(pd.DataFrame(audit["gbm_probabilistic"]), width="stretch", hide_index=True)

    st.caption(f"AIC SARIMA: {audit['sarima_aic']:,.1f} | AIC SARIMAX: {audit['sarimax_aic']:,.1f} — periodisitas S=24 gagal menangkap siklus mingguan lag 168.")
    df_19 = df[df["year"] == 2019]
    apr20 = df[(df["datetime"] >= "2020-04-01") & (df["datetime"] <= "2020-04-21")]
    fig_inad = plot_mpl_sarima_inadequacy_3panel(df_19, apr20, audit["sarimax_hourly_rmse"])
    st.pyplot(fig_inad)
    plt.close(fig_inad)

    c_t1, c_t2 = st.columns(2, gap="large")
    with c_t1:
        st.markdown("**Uji Diagnostik Residual**")
        st.dataframe(pd.DataFrame(audit["sarima_diagnostics"]), width="stretch", hide_index=True)
    with c_t2:
        st.markdown("**Error Breakdown SARIMAX 2019: Hari Biasa vs Hari Libur**")
        st.dataframe(pd.DataFrame(audit["sarimax_holiday_breakdown"]), width="stretch", hide_index=True)
