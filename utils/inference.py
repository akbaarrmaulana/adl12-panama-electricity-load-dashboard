"""
Real-time inference helpers for custom long-term 12-month scenarios (Page 5)
and interactive What-If hourly/24-hour simulations (Page 6).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

HOLIDAY_LABELS = {
    0: "0 - Hari Biasa (Bukan Libur)",
    1: "1 - Tahun Baru (1 Jan)",
    2: "2 - Hari Martir / Martyrs' Day (9 Jan)",
    3: "3 - Karnaval Sabtu (Sabtu Pra-Puasa)",
    4: "4 - Karnaval Minggu",
    5: "5 - Karnaval Senin",
    6: "6 - Karnaval Selasa (Martes de Carnaval)",
    7: "7 - Rabu Abu (Miércoles de Ceniza)",
    8: "8 - Kamis Putih (Jueves Santo)",
    9: "9 - Jumat Agung (Viernes Santo)",
    10: "10 - Sabtu Suci (Sábado Santo)",
    11: "11 - Minggu Paskah (Domingo de Resurrección)",
    12: "12 - Hari Buruh (1 Mei)",
    13: "13 - Pendirian Kota Panama (15 Agu)",
    14: "14 - Hari Pemisahan (3 Nov)",
    15: "15 - Hari Bendera (4 Nov)",
    16: "16 - Gerakan Kemerdekaan Colón (5 Nov)",
    17: "17 - Pemberontakan Los Santos (10/11 Nov)",
    18: "18 - Kemerdekaan dari Spanyol (28 Nov)",
    19: "19 - Hari Ibu (8 Des)",
    20: "20 - Malam Natal (24 Des)",
    21: "21 - Hari Natal (25 Des)",
    22: "22 - Malam Tahun Baru (31 Des)",
}


def _build_gam_inputs(sub_df, preproc, clip_bounds=None, extra_lin_cols=None):
    lin_cols = preproc["gam_lin_cols"] + (extra_lin_cols or [])
    ex_lin = sm.add_constant(sub_df[lin_cols].astype(float), has_constant="add")
    ex_sm = sub_df[preproc["smooth_cols"]].astype(float).copy()
    if clip_bounds is not None:
        for col, (lo, hi) in clip_bounds.items():
            ex_sm[col] = ex_sm[col].clip(lo, hi)
    return ex_lin, ex_sm


def compute_custom_lt_forecast(lt_df, gam_pre_pkg, gam_cov_pkg, preproc, recovery_pct=50.0, delta_t2m=0.0):
    """
    Re-evaluate the 12-month (8,760h) GAM projection under a custom recovery blend
    and regional temperature anomaly delta_t2m (°C).
    """
    w_rec = float(np.clip(recovery_pct / 100.0, 0.0, 1.0))

    if abs(delta_t2m) < 1e-6:
        pred_norm = lt_df["lt_base_mean"].values
        pred_cov = lt_df["lt_cov_mean"].values
    else:
        work = lt_df.copy()
        for col in preproc["t2m_cols"]:
            work[col] = work[col] + float(delta_t2m)
        work["T2M_pca1"] = (
            preproc["pca_t2m"].transform(preproc["scaler_t2m"].transform(work[preproc["t2m_cols"]])).ravel()
            * preproc["pca_t2m_sign"]
        )
        work["T2M_pca1_x_cos_hour"] = work["T2M_pca1"] * work["cos_hour"]
        work["T2M_pca1_x_sin_hour"] = work["T2M_pca1"] * work["sin_hour"]

        ex_l_a, ex_s_a = _build_gam_inputs(work, preproc, clip_bounds=gam_pre_pkg["bounds"])
        pred_norm = np.asarray(
            gam_pre_pkg["model"].predict(exog=ex_l_a[gam_pre_pkg["exog_cols"]], exog_smooth=ex_s_a),
            dtype=float,
        )

        work["is_covid_regime"] = 1.0
        ex_l_b, ex_s_b = _build_gam_inputs(
            work, preproc, clip_bounds=gam_cov_pkg["bounds"], extra_lin_cols=["is_covid_regime"]
        )
        pred_cov = np.asarray(
            gam_cov_pkg["model"].predict(exog=ex_l_b[gam_cov_pkg["exog_cols"]], exog_smooth=ex_s_b),
            dtype=float,
        )

    pred_custom = w_rec * pred_norm + (1.0 - w_rec) * pred_cov
    df_out = pd.DataFrame({
        "datetime": lt_df["datetime"],
        "Bulan": lt_df["Bulan"],
        "custom_mean": pred_custom,
    })
    monthly = (
        df_out.groupby("Bulan", sort=False)
        .agg(
            Custom_Mean_MWh=("custom_mean", "mean"),
            Custom_Energi_GWh=("custom_mean", lambda s: s.sum() / 1000.0),
        )
        .round(2)
        .reset_index()
    )
    summary = {
        "mean_mwh": round(float(pred_custom.mean()), 2),
        "peak_mwh": round(float(pred_custom.max()), 2),
        "total_gwh": round(float(pred_custom.sum() / 1000.0), 2),
        "total_twh": round(float(pred_custom.sum() / 1_000_000.0), 3),
    }
    return df_out, monthly, summary


def simulate_whatif_24h(
    model_choice,
    month,
    day_of_week,
    holiday_id,
    school,
    t2m_toc,
    t2m_san,
    t2m_dav,
    qv2m_toc,
    hour_focus,
    covid_active,
    lag_regime,
    gam_pre_pkg,
    gam_cov_pkg,
    gbm_pkg,
    preproc,
):
    """
    Generate a 24-hour (00:00..23:00) simulated load curve with uncertainty intervals
    under user-configured calendar, 3-city weather, and pandemic regime settings.
    """
    hours = np.arange(24, dtype=float)
    clim_m = preproc["clim_mdh"][preproc["clim_mdh"]["month"] == int(month)].groupby("hour_int", as_index=True).mean()

    # Build diurnal temperature shapes anchored exactly at user's chosen values at hour_focus
    hf = int(hour_focus)
    t2m_toc_curve = clim_m["T2M_toc"].values + (float(t2m_toc) - float(clim_m.loc[hf, "T2M_toc"]))
    t2m_san_curve = clim_m["T2M_san"].values + (float(t2m_san) - float(clim_m.loc[hf, "T2M_san"]))
    t2m_dav_curve = clim_m["T2M_dav"].values + (float(t2m_dav) - float(clim_m.loc[hf, "T2M_dav"]))
    qv_toc_curve = np.clip(
        clim_m["QV2M_toc"].values + (float(qv2m_toc) - float(clim_m.loc[hf, "QV2M_toc"])),
        0.0095,
        0.0230,
    )

    sim_df = pd.DataFrame({"hour": hours, "day_of_week": float(day_of_week), "month": int(month)})
    # Representative middle-of-month day_of_year
    ref_dt = pd.Timestamp(year=2020, month=int(month), day=15)
    sim_df["day_of_year"] = float(ref_dt.dayofyear)
    sim_df["is_weekend"] = float(1.0 if int(day_of_week) >= 5 else 0.0)
    # Use mid-2020 macro trend position
    sim_df["time_index_years"] = 5.45
    sim_df["sin_hour"] = np.sin(2 * np.pi * sim_df["hour"] / 24.0)
    sim_df["cos_hour"] = np.cos(2 * np.pi * sim_df["hour"] / 24.0)
    sim_df["sin_dow"] = np.sin(2 * np.pi * sim_df["day_of_week"] / 7.0)
    sim_df["cos_dow"] = np.cos(2 * np.pi * sim_df["day_of_week"] / 7.0)
    sim_df["sin_doy"] = np.sin(2 * np.pi * sim_df["day_of_year"] / 365.25)
    sim_df["cos_doy"] = np.cos(2 * np.pi * sim_df["day_of_year"] / 365.25)

    sim_df["holiday"] = 1 if int(holiday_id) > 0 else 0
    sim_df["Holiday_ID"] = int(holiday_id)
    sim_df["school"] = int(school)
    sim_df["pre_holiday"] = 0.0
    sim_df["post_holiday"] = 0.0

    for col in preproc["weather_cols"]:
        sim_df[col] = clim_m[col].values
    sim_df["T2M_toc"] = t2m_toc_curve
    sim_df["T2M_san"] = t2m_san_curve
    sim_df["T2M_dav"] = t2m_dav_curve
    sim_df["QV2M_toc"] = qv_toc_curve

    sim_df["T2M_pca1"] = (
        preproc["pca_t2m"].transform(preproc["scaler_t2m"].transform(sim_df[preproc["t2m_cols"]])).ravel()
        * preproc["pca_t2m_sign"]
    )
    sim_df["T2M_pca1_x_cos_hour"] = sim_df["T2M_pca1"] * sim_df["cos_hour"]
    sim_df["T2M_pca1_x_sin_hour"] = sim_df["T2M_pca1"] * sim_df["sin_hour"]

    for col in preproc["hol_dummy_cols"]:
        hid = int(col.split("_")[1])
        sim_df[col] = 1.0 if int(holiday_id) == hid else 0.0

    if model_choice.startswith("GAM"):
        if covid_active:
            sim_df["is_covid_regime"] = 1.0
            ex_l, ex_s = _build_gam_inputs(
                sim_df, preproc, clip_bounds=gam_cov_pkg["bounds"], extra_lin_cols=["is_covid_regime"]
            )
            sf = gam_cov_pkg["model"].get_prediction(
                exog=ex_l[gam_cov_pkg["exog_cols"]], exog_smooth=ex_s
            ).summary_frame()
        else:
            ex_l, ex_s = _build_gam_inputs(sim_df, preproc, clip_bounds=gam_pre_pkg["bounds"])
            sf = gam_pre_pkg["model"].get_prediction(
                exog=ex_l[gam_pre_pkg["exog_cols"]], exog_smooth=ex_s
            ).summary_frame()
        sim_df["pred_mean"] = sf["mean"].values
        sim_df["pred_lo"] = sf["mean_ci_lower"].values
        sim_df["pred_hi"] = sf["mean_ci_upper"].values
        interval_label = "95% CI (GAM)"
    else:
        # Boosting Set B: build T2M lags from cyclic 24h curve and load lags from selected reference week
        t2m_s = pd.Series(np.concatenate([t2m_toc_curve[-3:], t2m_toc_curve]))
        for lag in [1, 2, 3]:
            sim_df[f"T2M_toc_lag{lag}"] = t2m_s.shift(lag).iloc[3:].values
        sim_df["T2M_toc_roll24"] = float(np.mean(t2m_toc_curve))

        ref_tbl = preproc["cov_ref_week"] if lag_regime == "Lockdown COVID-19 (Apr 2020)" else preproc["norm_ref_week"]
        ref_day = ref_tbl[ref_tbl["day_of_week"] == float(day_of_week)].sort_values("hour")
        if len(ref_day) != 24:
            ref_day = ref_tbl.groupby("hour", as_index=False).mean().sort_values("hour")
        for lag_col in ["load_lag_168", "load_lag_336", "load_lag_504", "load_roll168_mean", "load_roll168_std", "load_roll24_max_lag168"]:
            sim_df[lag_col] = ref_day[lag_col].values

        X_sim = sim_df[gbm_pkg["set_b_features"]].copy()
        for col, dtype_str in gbm_pkg["feature_dtypes"].items():
            X_sim[col] = X_sim[col].astype(dtype_str)

        sim_df["pred_mean"] = gbm_pkg["model_pt"].predict(X_sim)
        q_sorted = np.sort(np.column_stack([
            gbm_pkg["model_q05"].predict(X_sim),
            gbm_pkg["model_q50"].predict(X_sim),
            gbm_pkg["model_q95"].predict(X_sim),
        ]), axis=1)
        sim_df["pred_lo"] = q_sorted[:, 0]
        sim_df["pred_hi"] = q_sorted[:, 2]
        interval_label = "90% PI (Q05–Q95)"

    return sim_df[["hour", "T2M_toc", "T2M_san", "T2M_dav", "pred_mean", "pred_lo", "pred_hi"]], interval_label
