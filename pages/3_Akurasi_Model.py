import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from utils.charts import (
    make_transfer_learning_timeline,
    make_weekly_zoom_chart,
    plot_mpl_backtest_and_acf_comparison,
    plot_mpl_model_accuracy_layman,
    plot_mpl_sarima_inadequacy_3panel,
    plot_mpl_transfer_learning_ranking,
    plot_mpl_tscv_3fold,
)
from utils.data_loader import (
    load_audit_stats,
    load_master_hourly,
    load_transfer_learning,
    load_transfer_learning_series,
)
from utils.ui_components import inject_styles, render_insight_box, render_neu_kpis, render_page_header

inject_styles()

df = load_master_hourly()
audit = load_audit_stats()
tl_data = load_transfer_learning()
tl_series = load_transfer_learning_series()

render_page_header(
    "Perbandingan Akurasi & Transfer Learning Model",
    "Bandingkan ketepatan 5 model pada tahun normal (2019), lalu lihat bagaimana Transfer Learning mengadaptasi model dengan cepat saat pandemi 2020.",
    kicker="Evaluasi Model",
)

tab_eval, tab_tl = st.tabs([
    "Perbandingan 5 Model (Tahun Normal 2019)",
    "Transfer Learning (Adaptasi Pandemi 2020)",
])

with tab_eval:
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

with tab_tl:
    tl_df = pd.DataFrame(tl_data["summary"])
    tl_meta = tl_data["split_meta"]
    best_tl = tl_data["best_tl"]
    direct_base = tl_data["direct_gbm_baseline"]

    render_insight_box(
        "Apa Itu Transfer Learning di Kasus Ini?",
        f"Saat pandemi COVID-19 mendadak mengubah pola listrik, model yang hanya dilatih pada kondisi normal 2015–2019 langsung meleset jauh (RMSE {direct_base['RMSE']:.1f} MWh). Dengan Transfer Learning, pengetahuan dari 5 tahun data normal dipindahkan lalu disesuaikan hanya dengan 1 bulan data awal lockdown ({tl_meta['adapt_rows']} jam) agar kembali akurat.",
        tone="blue",
    )

    render_neu_kpis([
        {
            "label": "Transfer Learning Terbaik",
            "value": f"{100.0 - best_tl['MAPE_%']:.2f}% Akurat",
            "hint": f"{best_tl['Model']} (RMSE {best_tl['RMSE']:.2f} MWh · MAPE {best_tl['MAPE_%']:.2f}%)",
            "badge": "Pemenang Adaptasi",
            "tone": "teal",
        },
        {
            "label": "Penghematan Error vs Tanpa Adaptasi",
            "value": f"-{best_tl['Reduksi_RMSE_vs_Direct_%']:.1f}%",
            "hint": f"Menekan selisih dari {direct_base['RMSE']:.1f} MWh menjadi {best_tl['RMSE']:.1f} MWh",
            "badge": "Reduksi RMSE",
            "tone": "blue",
        },
        {
            "label": "Data Adaptasi yang Dibutuhkan",
            "value": f"{tl_meta['adapt_rows']} Jam (1 Bulan)",
            "hint": f"Hanya {tl_meta['adapt_pct_of_source']:.2f}% dari {tl_meta['source_rows']:,} jam data sumber 2015–2020",
            "badge": "Efisien Data",
            "tone": "amber",
        },
    ])

    with st.container(border=True):
        fig_tl_rank = plot_mpl_transfer_learning_ranking(tl_df)
        st.pyplot(fig_tl_rank)
        plt.close(fig_tl_rank)

    with st.container(border=True):
        st.markdown("**Perbandingan Harian di Masa Lockdown (16 April – 27 Juni 2020)**")
        tl_opts = {
            "Tanpa Adaptasi (Boosting Source)": "pred_direct_gbm",
            "TL-2: Tree Warm-Start Transfer (Boosting)": "pred_tl_gbm",
            "TL-1: Neural Net Fine-Tuning (MLP)": "pred_tl_mlp",
            "TL-3: Frozen GAM + Residual Adapter": "pred_tl_gam_adapter",
            "Latih dari Nol di Target (744j)": "pred_scratch_gbm",
            "Tanpa Adaptasi (MLP Source)": "pred_direct_mlp",
        }
        tl_defaults = [
            "Tanpa Adaptasi (Boosting Source)",
            "TL-2: Tree Warm-Start Transfer (Boosting)",
            "TL-1: Neural Net Fine-Tuning (MLP)",
            "TL-3: Frozen GAM + Residual Adapter",
        ]
        chosen_tl = st.multiselect("Pilih Kurva Model Transfer Learning", list(tl_opts.keys()), default=tl_defaults)
        selected_tl_cols = [tl_opts[lbl] for lbl in chosen_tl]
        st.plotly_chart(
            make_transfer_learning_timeline(tl_series, selected_tl_cols),
            width="stretch",
            theme=None,
            config={"displaylogo": False},
        )

    with st.expander("Lihat Tabel Lengkap, Penjelasan 3 Metode Transfer Learning, & Unduh CSV"):
        st.markdown(
            f"- **Domain Sumber (`{tl_meta['source_rows']:,}` jam):** `{tl_meta['source_range']}` | "
            f"**Adaptasi Target (`{tl_meta['adapt_rows']}` jam):** `{tl_meta['adapt_range']}` | "
            f"**Uji Masa Depan Lockdown (`{tl_meta['test_rows']:,}` jam):** `{tl_meta['test_range']}`.\n"
            "- **TL-1 — Neural Net Parameter Fine-Tuning (`MLPRegressor warm_start`):** Membekukan skala & bobot representasi dari 5 tahun data normal lalu memperbarui bobot secara halus (`lr=8e-4`, regularisasi L2) pada `744` jam awal lockdown.\n"
            f"- **TL-2 — Tree Warm-Start Transfer (`HistGradientBoosting warm_start`):** Menambahkan `+60` pohon koreksi (`{tl_meta['gbm_src_trees']} → {tl_meta['gbm_tl_trees']}` pohon) di atas ansambel pra-pandemi untuk mengoreksi penurunan beban siang & sekolah.\n"
            "- **TL-3 — Frozen GAM Backbone + Ridge Residual Adapter:** Menjaga kurva dasar GAM tetap beku (`cf_mean`) lalu melatih *adapter* Ridge ringan untuk menggeser level & pola intraharian pada rezim pandemi."
        )
        disp_tl = tl_df[["Model", "Kategori", "MAE", "RMSE", "MAPE_%", "Bias_MWh", "Reduksi_RMSE_vs_Direct_%"]]
        st.dataframe(disp_tl, width="stretch", hide_index=True)
        st.download_button(
            "Unduh Prediksi Transfer Learning Per Jam (CSV)",
            data=tl_series.to_csv(index=False).encode("utf-8"),
            file_name="panama_transfer_learning_covid2020.csv",
            mime="text/csv",
        )
