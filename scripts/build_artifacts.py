"""
Offline artifact builder for the Panama Hourly Electricity Load Streamlit Dashboard.
Replicates Analisis_Beban_Listrik_Panama_2015-2020.ipynb identically and exports
pre-computed data tables, diagnostic statistics, and trained models into artifacts/.
"""

import importlib.util
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.fft import rfft, rfftfreq
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.gam.api import BSplines, GLMGam
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, het_breuschpagan
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, kpss

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Panama_beban_listrik_per_jam_2015-2020_.xlsx")
ART_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

HAS_LGBM = importlib.util.find_spec("lightgbm") is not None
HAS_SHAP = importlib.util.find_spec("shap") is not None


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err / y_true)) * 100.0)
    smape = float(np.mean(2.0 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred))) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape}


def pinball_loss(y_true, y_pred, q):
    e = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def winkler_score(y_true, lower, upper, alpha=0.10):
    y = np.asarray(y_true)
    l = np.asarray(lower)
    u = np.asarray(upper)
    width = u - l
    penalty = np.where(y < l, (2.0 / alpha) * (l - y), np.where(y > u, (2.0 / alpha) * (y - u), 0.0))
    return float(np.mean(width + penalty))


def diebold_mariano_test(y_true, pred_a, pred_b, h=168, loss_type="squared"):
    e_a = np.asarray(y_true, dtype=float) - np.asarray(pred_a, dtype=float)
    e_b = np.asarray(y_true, dtype=float) - np.asarray(pred_b, dtype=float)
    d = (e_a ** 2 - e_b ** 2) if loss_type == "squared" else (np.abs(e_a) - np.abs(e_b))
    T = len(d)
    d_bar = float(np.mean(d))
    dc = d - d_bar
    K = min(h - 1, T - 1)
    gamma0 = float(np.dot(dc, dc) / T)
    hac = gamma0
    for k in range(1, K + 1):
        wk = 1.0 - k / (K + 1.0)
        gamma_k = float(np.dot(dc[k:], dc[:-k]) / T)
        hac += 2.0 * wk * gamma_k
    if hac <= 0:
        hac = gamma0
    dm_stat = float(d_bar / np.sqrt(hac / T))
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))
    return dm_stat, p_val, d_bar


def make_gbm(loss_mode="squared", alpha_q=0.5, lr=0.05, max_iter=350, max_leaf=63):
    if HAS_LGBM:
        import lightgbm as lgb

        obj_map = {"squared": "regression", "absolute": "regression_l1", "quantile": "quantile"}
        params = dict(
            objective=obj_map[loss_mode],
            learning_rate=lr,
            n_estimators=max_iter,
            num_leaves=max_leaf,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        if loss_mode == "quantile":
            params["alpha"] = alpha_q
        return lgb.LGBMRegressor(**params)
    loss_sk = {"squared": "squared_error", "absolute": "absolute_error", "quantile": "quantile"}[loss_mode]
    params = dict(
        loss=loss_sk,
        learning_rate=lr,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf,
        min_samples_leaf=40,
        l2_regularization=5.0,
        categorical_features=["Holiday_ID"],
        random_state=42,
    )
    if loss_mode == "quantile":
        params["quantile"] = alpha_q
    return HistGradientBoostingRegressor(**params)


def save_table(df_obj, name_stem):
    pq_path = os.path.join(ART_DIR, f"{name_stem}.parquet")
    try:
        df_obj.to_parquet(pq_path, index=False)
        return pq_path
    except Exception:
        gz_path = os.path.join(ART_DIR, f"{name_stem}.csv.gz")
        df_obj.to_csv(gz_path, index=False, compression="gzip")
        return gz_path


def main():
    t_start = time.time()
    print(f"Loading dataset: {DATA_PATH}", flush=True)
    df = pd.read_excel(DATA_PATH, sheet_name="data")
    desc = pd.read_excel(DATA_PATH, sheet_name="deskripsi")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month

    # ------------------------------------------------------------------
    # Stage 1: Statistical Characterization & Chronological Split
    # ------------------------------------------------------------------
    weather_cols = [
        "T2M_toc", "QV2M_toc", "TQL_toc", "W2M_toc",
        "T2M_san", "QV2M_san", "TQL_san", "W2M_san",
        "T2M_dav", "QV2M_dav", "TQL_dav", "W2M_dav",
    ]
    yearly_stats = (
        df.groupby("year")["nat_demand"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(2)
        .reset_index()
    )

    mj = df[df["month"].isin([3, 4, 5, 6])].groupby(["year", "month"])["nat_demand"].mean().unstack().round(2)
    mj_cmp = pd.DataFrame({
        "month": mj.columns.astype(int),
        "2019_MWh": mj.loc[2019].values,
        "2020_MWh": mj.loc[2020].values,
        "Perubahan_MWh": (mj.loc[2020] - mj.loc[2019]).round(2).values,
        "Perubahan_%": ((mj.loc[2020] - mj.loc[2019]) / mj.loc[2019] * 100).round(2).values,
    })

    blackouts_df = df[df["nat_demand"] < 600][["datetime", "nat_demand", "holiday", "Holiday_ID", "school"]].copy()
    blackouts_df["datetime"] = blackouts_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Stationarity tests
    y_pre = df.loc[df["year"] <= 2019, "nat_demand"]

    def run_stationarity_tests(series, name):
        adf_stat, adf_p, _, _, _, _ = adfuller(series.dropna(), maxlag=48, autolag="AIC")
        kpss_stat, kpss_p, _, _ = kpss(series.dropna(), regression="ct", nlags="auto")
        return {
            "Seri": name,
            "ADF_stat": round(float(adf_stat), 3),
            "ADF_p": round(float(adf_p), 4),
            "KPSS_stat": round(float(kpss_stat), 4),
            "KPSS_p": round(float(kpss_p), 4),
        }

    stat_rows = [
        run_stationarity_tests(y_pre, "Level Y_t (2015-2019)"),
        run_stationarity_tests(y_pre.diff(), "First Difference (1-B)Y_t"),
        run_stationarity_tests(y_pre.diff(24), "Seasonal Diff (1-B^24)Y_t"),
    ]

    # FFT Periodogram
    y_centered = df["nat_demand"].values - df["nat_demand"].mean()
    n_fft = len(y_centered)
    yf = np.abs(rfft(y_centered)) ** 2 / n_fft
    xf = rfftfreq(n_fft, d=1.0)
    periods = np.where(xf > 0, 1.0 / xf, np.inf)
    top_idx = np.argsort(yf)[::-1][:8]
    fft_top_peaks = [
        {"Periode_Jam": round(float(periods[idx]), 2), "Power": round(float(yf[idx]), 1)}
        for idx in top_idx if np.isfinite(periods[idx])
    ]
    mask_p = (periods >= 10) & (periods <= 250)
    fft_spectrum = {
        "periods": np.round(periods[mask_p], 3).tolist(),
        "power": np.round(yf[mask_p], 1).tolist(),
    }

    # ACF of raw demand (lags 0..180)
    acf_raw = acf(df["nat_demand"], nlags=180, fft=True)

    # VIF & correlation
    corr_target = df[weather_cols + ["holiday", "Holiday_ID", "school", "nat_demand"]].corr()["nat_demand"].drop("nat_demand")
    corr_mat_w = df[weather_cols].corr().values
    vifs = np.diag(np.linalg.inv(corr_mat_w))
    vif_df = pd.DataFrame({
        "Variabel_Cuaca": weather_cols,
        "Kota": [c.split("_")[1].upper() for c in weather_cols],
        "Metrik": [c.split("_")[0] for c in weather_cols],
        "VIF": np.round(vifs, 2),
        "Corr_nat_demand": [round(float(corr_target[c]), 4) for c in weather_cols],
    }).sort_values("VIF", ascending=False)

    # Heteroskedasticity tests
    X_ols = sm.add_constant(df[weather_cols + ["holiday", "school"]].astype(float))
    ols_res = sm.OLS(df["nat_demand"], X_ols).fit()
    bp_lm, bp_p, _, _ = het_breuschpagan(ols_res.resid, ols_res.model.exog)
    arch_lm, arch_p, _, _ = het_arch(ols_res.resid, nlags=24)
    heteroskedasticity = {
        "bp_lm": round(float(bp_lm), 2),
        "bp_p": float(bp_p),
        "arch_lm": round(float(arch_lm), 2),
        "arch_p": float(arch_p),
        "holiday0_mean": round(float(df.loc[df["holiday"] == 0, "nat_demand"].mean()), 2),
        "holiday0_std": round(float(df.loc[df["holiday"] == 0, "nat_demand"].std()), 2),
        "holiday1_mean": round(float(df.loc[df["holiday"] == 1, "nat_demand"].mean()), 2),
        "holiday1_std": round(float(df.loc[df["holiday"] == 1, "nat_demand"].std()), 2),
    }

    # Blackout handling & chronological 3-regime split
    df["is_blackout"] = (df["nat_demand"] < 600).astype(int)
    df["nat_demand_clean"] = df["nat_demand"].copy()
    df["nat_demand_clean_full"] = df["nat_demand"].copy()

    train_end_dt = pd.Timestamp("2018-12-31 23:00:00")
    test_norm_end_dt = pd.Timestamp("2019-12-31 23:00:00")
    covid_start_dt = pd.Timestamp("2020-03-01 00:00:00")

    train_mask = df["datetime"] <= train_end_dt
    test_2019_mask = (df["datetime"] > train_end_dt) & (df["datetime"] <= test_norm_end_dt)
    test_2020_mask = df["datetime"] > test_norm_end_dt
    lockdown_2020_mask = df["datetime"] >= covid_start_dt
    mask_2019_clean = test_2019_mask & (df["is_blackout"] == 0)

    bo_indices = df.index[df["is_blackout"] == 1].tolist()
    for idx in bo_indices:
        prev_val = df.loc[idx - 168, "nat_demand"] if idx - 168 >= 0 else np.nan
        next_val = df.loc[idx + 168, "nat_demand"] if idx + 168 < len(df) else np.nan
        imputed = float(np.nanmean([prev_val, next_val]))
        df.loc[idx, "nat_demand_clean_full"] = imputed
        if df.loc[idx, "datetime"] <= train_end_dt:
            df.loc[idx, "nat_demand_clean"] = imputed

    df["regime"] = np.where(
        train_mask, "Training (2015-2018)",
        np.where(
            test_2019_mask, "Test Normal (2019)",
            np.where(lockdown_2020_mask, "Test 2020 (Lockdown Mar-Jun)", "Test 2020 (Pra-Lockdown Jan-Feb)")
        )
    )

    split_summary = [
        {
            "Rezim": "1. Training (2015-2018)",
            "Rentang_Waktu": f"{df.loc[train_mask, 'datetime'].min()} s.d. {df.loc[train_mask, 'datetime'].max()}",
            "Jumlah_Jam": int(train_mask.sum()),
            "Proporsi_%": round(float(train_mask.mean() * 100.0), 2),
            "Titik_Blackout": int((train_mask & (df["is_blackout"] == 1)).sum()),
            "Fungsi": "Estimasi parameter model & CV TimeSeriesSplit (3-fold)",
        },
        {
            "Rezim": "2. Test Normal (2019)",
            "Rentang_Waktu": f"{df.loc[test_2019_mask, 'datetime'].min()} s.d. {df.loc[test_2019_mask, 'datetime'].max()}",
            "Jumlah_Jam": int(test_2019_mask.sum()),
            "Proporsi_%": round(float(test_2019_mask.mean() * 100.0), 2),
            "Titik_Blackout": int((test_2019_mask & (df["is_blackout"] == 1)).sum()),
            "Fungsi": "Evaluasi normal 1 tahun (All & No Blackout) + Placebo Mar-Jun 2019",
        },
        {
            "Rezim": "3a. Test 2020 (All: Jan-Jun)",
            "Rentang_Waktu": f"{df.loc[test_2020_mask, 'datetime'].min()} s.d. {df.loc[test_2020_mask, 'datetime'].max()}",
            "Jumlah_Jam": int(test_2020_mask.sum()),
            "Proporsi_%": round(float(test_2020_mask.mean() * 100.0), 2),
            "Titik_Blackout": int((test_2020_mask & (df["is_blackout"] == 1)).sum()),
            "Fungsi": "Kontrol Jan-Feb 2020 + uji disrupsi COVID-19",
        },
        {
            "Rezim": "3b. Test 2020 (Lockdown: Mar-Jun)",
            "Rentang_Waktu": f"{df.loc[lockdown_2020_mask, 'datetime'].min()} s.d. {df.loc[lockdown_2020_mask, 'datetime'].max()}",
            "Jumlah_Jam": int(lockdown_2020_mask.sum()),
            "Proporsi_%": round(float(lockdown_2020_mask.mean() * 100.0), 2),
            "Titik_Blackout": int((lockdown_2020_mask & (df["is_blackout"] == 1)).sum()),
            "Fungsi": "Evaluasi ketangguhan STLF & dampak kausal lockdown",
        },
    ]

    # ------------------------------------------------------------------
    # Stage 2: Seasonal Naive, SARIMA, & SARIMAX + Diagnostics
    # ------------------------------------------------------------------
    print("Stage 2: Fitting SARIMA & SARIMAX and rolling 168h forecasts...", flush=True)
    df["pred_naive"] = df["nat_demand"].shift(168)

    sarima_win = 1008
    sarima_train_df = df.loc[train_mask].iloc[-sarima_win:].copy()
    sarimax_cols = ["T2M_toc", "QV2M_toc", "W2M_toc", "holiday", "school"]

    scaler_sx = StandardScaler()
    exog_train_sx = pd.DataFrame(
        scaler_sx.fit_transform(sarima_train_df[sarimax_cols]),
        index=sarima_train_df.index,
        columns=sarimax_cols,
    )
    y_sx_train = sarima_train_df["nat_demand_clean"].rename("nat_demand")
    res_sarima = SARIMAX(
        y_sx_train, order=(1, 0, 1), seasonal_order=(1, 1, 0, 24),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=80)
    res_sarimax = SARIMAX(
        y_sx_train, exog=exog_train_sx, order=(1, 0, 1), seasonal_order=(1, 1, 0, 24),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=80)

    def rolling_weekly_sarima_forecast(base_res, full_df, start_pos, exog_df=None, horizon=168, window=1008):
        preds = pd.Series(index=full_df.index[start_pos:], dtype=float)
        pos = start_pos
        while pos < len(full_df):
            end_pos = min(pos + horizon, len(full_df))
            steps = end_pos - pos
            y_win = full_df["nat_demand"].iloc[pos - window : pos]
            if exog_df is None:
                cur_res = base_res.apply(y_win)
                fc = cur_res.get_forecast(steps=steps).predicted_mean
            else:
                ex_win = exog_df.iloc[pos - window : pos]
                ex_fc = exog_df.iloc[pos:end_pos]
                cur_res = base_res.apply(y_win, exog=ex_win)
                fc = cur_res.get_forecast(steps=steps, exog=ex_fc).predicted_mean
            preds.iloc[pos - start_pos : end_pos - start_pos] = fc.values
            pos = end_pos
        return preds

    exog_all_sx = pd.DataFrame(scaler_sx.transform(df[sarimax_cols]), index=df.index, columns=sarimax_cols)
    start_2019_pos = int(df.index[test_2019_mask][0])
    df.loc[start_2019_pos:, "pred_sarima"] = rolling_weekly_sarima_forecast(res_sarima, df, start_2019_pos)
    df.loc[start_2019_pos:, "pred_sarimax"] = rolling_weekly_sarima_forecast(res_sarimax, df, start_2019_pos, exog_df=exog_all_sx)

    sarima_diag_rows = []
    for name, res_obj, pred_col in [
        ("SARIMA (Train Resid)", res_sarima, None),
        ("SARIMAX (Train Resid)", res_sarimax, None),
        ("SARIMA (Test 2019 Error)", None, "pred_sarima"),
        ("SARIMAX (Test 2019 Error)", None, "pred_sarimax"),
    ]:
        r = res_obj.resid.iloc[24:].values if res_obj is not None else (df.loc[mask_2019_clean, "nat_demand"] - df.loc[mask_2019_clean, pred_col]).values
        lb = acorr_ljungbox(r, lags=[24, 48, 168], return_df=True)
        _, jb_p = stats.jarque_bera(r)
        _, arch_p_r, _, _ = het_arch(r, nlags=24)
        sarima_diag_rows.append({
            "Sumber_Residual": name,
            "LjungBox_lag24_p": f"{lb.loc[24, 'lb_pvalue']:.2e}",
            "LjungBox_lag168_p": f"{lb.loc[168, 'lb_pvalue']:.2e}",
            "JarqueBera_p": f"{jb_p:.2e}",
            "ARCH_LM_p": f"{arch_p_r:.2e}",
        })

    sub19_sx = df.loc[mask_2019_clean].copy()
    sub19_sx["hour_tmp"] = sub19_sx["datetime"].dt.hour
    sub19_sx["err_sarimax"] = sub19_sx["nat_demand"] - sub19_sx["pred_sarimax"]
    hol_err = sub19_sx.groupby("holiday").apply(
        lambda g: pd.Series(compute_metrics(g["nat_demand"], g["pred_sarimax"])), include_groups=False
    ).round(2)
    hol_err.index = ["Hari Biasa (holiday=0)", "Hari Libur (holiday=1)"]
    hol_err_rows = hol_err.reset_index().rename(columns={"index": "Tipe_Hari"}).to_dict(orient="records")
    hourly_rmse_sx = sub19_sx.groupby("hour_tmp").apply(
        lambda g: float(np.sqrt(np.mean(g["err_sarimax"] ** 2))), include_groups=False
    )

    # ------------------------------------------------------------------
    # Stage 3: Feature Engineering (Set A & Set B)
    # ------------------------------------------------------------------
    df["time_index_years"] = np.arange(len(df), dtype=float) / 8760.0
    df["hour"] = df["datetime"].dt.hour.astype(float)
    df["day_of_week"] = df["datetime"].dt.dayofweek.astype(float)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)
    df["day_of_year"] = df["datetime"].dt.dayofyear.astype(float)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    daily_hol = df.groupby(df["datetime"].dt.date)["holiday"].max()
    pre_dates = set(pd.Series(daily_hol.index)[daily_hol.shift(-1, fill_value=0).values == 1])
    post_dates = set(pd.Series(daily_hol.index)[daily_hol.shift(1, fill_value=0).values == 1])
    df["pre_holiday"] = df["datetime"].dt.date.isin(pre_dates).astype(float)
    df["post_holiday"] = df["datetime"].dt.date.isin(post_dates).astype(float)

    t2m_cols = ["T2M_toc", "T2M_san", "T2M_dav"]
    scaler_t2m = StandardScaler()
    pca_t2m = PCA(n_components=1)
    t2m_train_scaled = scaler_t2m.fit_transform(df.loc[train_mask, t2m_cols])
    pca_t2m.fit(t2m_train_scaled)
    df["T2M_pca1"] = pca_t2m.transform(scaler_t2m.transform(df[t2m_cols])).ravel()
    pca_t2m_sign = -1.0 if np.corrcoef(df.loc[train_mask, "T2M_pca1"], df.loc[train_mask, "T2M_toc"])[0, 1] < 0 else 1.0
    df["T2M_pca1"] *= pca_t2m_sign

    for lag in [1, 2, 3]:
        df[f"T2M_toc_lag{lag}"] = df["T2M_toc"].shift(lag).bfill()
    df["T2M_toc_roll24"] = df["T2M_toc"].rolling(24, min_periods=1).mean()
    df["T2M_pca1_x_cos_hour"] = df["T2M_pca1"] * df["cos_hour"]
    df["T2M_pca1_x_sin_hour"] = df["T2M_pca1"] * df["sin_hour"]

    hol_dummies = pd.get_dummies(df["Holiday_ID"], prefix="HolID", drop_first=True, dtype=float)
    hol_dummy_cols = list(hol_dummies.columns)
    df = pd.concat([df, hol_dummies], axis=1)

    df["load_lag_168"] = df["nat_demand_clean"].shift(168)
    df["load_lag_336"] = df["nat_demand_clean"].shift(336)
    df["load_lag_504"] = df["nat_demand_clean"].shift(504)
    df["load_roll168_mean"] = df["nat_demand_clean"].shift(168).rolling(168, min_periods=24).mean()
    df["load_roll168_std"] = df["nat_demand_clean"].shift(168).rolling(168, min_periods=24).std()
    df["load_roll24_max_lag168"] = df["nat_demand_clean"].shift(168).rolling(24, min_periods=1).max()

    set_b_features = (
        weather_cols
        + ["T2M_pca1", "T2M_toc_lag1", "T2M_toc_lag2", "T2M_toc_lag3", "T2M_toc_roll24"]
        + ["time_index_years", "hour", "day_of_week", "is_weekend", "day_of_year",
           "sin_hour", "cos_hour", "sin_dow", "cos_dow", "sin_doy", "cos_doy",
           "holiday", "Holiday_ID", "school", "pre_holiday", "post_holiday"]
        + ["load_lag_168", "load_lag_336", "load_lag_504",
           "load_roll168_mean", "load_roll168_std", "load_roll24_max_lag168"]
    )

    # ------------------------------------------------------------------
    # Stage 4: GAM + 3-Fold CV + Placebo + COVID-19 Counterfactual
    # ------------------------------------------------------------------
    print("Stage 4: Running GAM 3-fold CV, full fit, & COVID-19 counterfactual...", flush=True)
    smooth_cols = ["hour", "day_of_year", "T2M_pca1", "QV2M_toc"]
    gam_lin_cols = [
        "time_index_years", "is_weekend", "school", "pre_holiday", "post_holiday",
        "sin_dow", "cos_dow", "T2M_pca1_x_cos_hour", "T2M_pca1_x_sin_hour",
    ] + hol_dummy_cols

    def build_gam_inputs(sub_df, clip_bounds=None, extra_lin_cols=None):
        lin_cols = gam_lin_cols + (extra_lin_cols or [])
        ex_lin = sm.add_constant(sub_df[lin_cols].astype(float), has_constant="add")
        ex_sm = sub_df[smooth_cols].astype(float).copy()
        if clip_bounds is not None:
            for col, (lo, hi) in clip_bounds.items():
                ex_sm[col] = ex_sm[col].clip(lo, hi)
        return ex_lin, ex_sm

    def fit_panama_gam(fit_mask, alpha=None, extra_lin_cols=None, target_col="nat_demand_clean", df_source=None):
        src = df if df_source is None else df_source
        sub = src.loc[fit_mask].copy()
        bounds = {c: (float(sub[c].min()) + 1e-6, float(sub[c].max()) - 1e-6) for c in smooth_cols}
        ex_lin_full, ex_sm = build_gam_inputs(sub, clip_bounds=None, extra_lin_cols=extra_lin_cols)
        active_cols = ["const"] + [c for c in ex_lin_full.columns if c != "const" and ex_lin_full[c].nunique() > 1]
        ex_lin = ex_lin_full[active_cols]
        bs = BSplines(ex_sm, df=[14, 12, 10, 8], degree=[3, 3, 3, 3])
        if alpha is None:
            alpha = [5.0, 5.0, 10.0, 10.0]
        res = GLMGam(sub[target_col].astype(float), exog=ex_lin, smoother=bs, alpha=alpha).fit()
        return res, bounds, list(ex_lin.columns)

    train_idx_all = df.index[train_mask].to_numpy()
    tscv_gam = TimeSeriesSplit(n_splits=3)
    cv_gam_rows = []
    for fold, (tr_pos, val_pos) in enumerate(tscv_gam.split(train_idx_all), 1):
        tr_idx = train_idx_all[tr_pos]
        val_idx = train_idx_all[val_pos]
        res_cv, bounds_cv, cols_cv = fit_panama_gam(tr_idx)
        val_df_cv = df.loc[val_idx]
        ex_l_val, ex_s_val = build_gam_inputs(val_df_cv, clip_bounds=bounds_cv)
        pred_val = res_cv.predict(exog=ex_l_val[cols_cv], exog_smooth=ex_s_val)
        met_cv = compute_metrics(val_df_cv["nat_demand"], pred_val)
        cv_gam_rows.append({
            "Fold": fold,
            "Train_Period": f"{df.loc[tr_idx[0], 'datetime'].strftime('%Y-%m-%d')} s.d. {df.loc[tr_idx[-1], 'datetime'].strftime('%Y-%m-%d')}",
            "Val_Period": f"{df.loc[val_idx[0], 'datetime'].strftime('%Y-%m-%d')} s.d. {df.loc[val_idx[-1], 'datetime'].strftime('%Y-%m-%d')}",
            "N_Train": int(len(tr_idx)),
            "N_Val": int(len(val_idx)),
            "MAE": round(met_cv["MAE"], 2),
            "RMSE": round(met_cv["RMSE"], 2),
            "MAPE_%": round(met_cv["MAPE"], 2),
        })

    res_gam_18, gam_bounds_18, gam_exog_cols = fit_panama_gam(train_mask)
    ex_lin_all, ex_sm_all = build_gam_inputs(df, clip_bounds=gam_bounds_18)
    df["pred_gam"] = res_gam_18.get_prediction(exog=ex_lin_all[gam_exog_cols], exog_smooth=ex_sm_all).summary_frame()["mean"].values

    # Pre-compute GAM partial curves for fast visualization on Page 3
    base_row = df.loc[train_mask].median(numeric_only=True)

    def gam_partial_grid(col_name, grid_vals):
        g_df = pd.DataFrame([base_row] * len(grid_vals)).reset_index(drop=True)
        for c in hol_dummy_cols:
            g_df[c] = 0.0
        g_df["is_weekend"] = 0.0
        g_df["school"] = 1.0
        g_df["pre_holiday"] = 0.0
        g_df["post_holiday"] = 0.0
        g_df[col_name] = grid_vals
        if col_name == "hour":
            g_df["sin_hour"] = np.sin(2 * np.pi * g_df["hour"] / 24.0)
            g_df["cos_hour"] = np.cos(2 * np.pi * g_df["hour"] / 24.0)
            g_df["T2M_pca1_x_sin_hour"] = g_df["T2M_pca1"] * g_df["sin_hour"]
            g_df["T2M_pca1_x_cos_hour"] = g_df["T2M_pca1"] * g_df["cos_hour"]
        ex_l, ex_s = build_gam_inputs(g_df, clip_bounds=gam_bounds_18)
        return res_gam_18.get_prediction(exog=ex_l[gam_exog_cols], exog_smooth=ex_s).summary_frame()

    t_grid = np.linspace(gam_bounds_18["T2M_pca1"][0], gam_bounds_18["T2M_pca1"][1], 60)
    sf_t = gam_partial_grid("T2M_pca1", t_grid)
    h_grid = np.linspace(0, 23, 48)
    sf_h = gam_partial_grid("hour", h_grid)
    yr_grid = np.linspace(0, 4.0, 50)
    sf_yr = gam_partial_grid("time_index_years", yr_grid)
    hol_coefs = res_gam_18.params[hol_dummy_cols].sort_values()

    gam_partial_curves = {
        "temp": {"x": np.round(t_grid, 4).tolist(), "mean": np.round(sf_t["mean"].values, 2).tolist(), "ci_lo": np.round(sf_t["mean_ci_lower"].values, 2).tolist(), "ci_hi": np.round(sf_t["mean_ci_upper"].values, 2).tolist()},
        "hour": {"x": np.round(h_grid, 2).tolist(), "mean": np.round(sf_h["mean"].values, 2).tolist(), "ci_lo": np.round(sf_h["mean_ci_lower"].values, 2).tolist(), "ci_hi": np.round(sf_h["mean_ci_upper"].values, 2).tolist()},
        "trend": {"x": np.round(2015.0 + yr_grid, 3).tolist(), "mean": np.round(sf_yr["mean"].values, 2).tolist(), "ci_lo": np.round(sf_yr["mean_ci_lower"].values, 2).tolist(), "ci_hi": np.round(sf_yr["mean_ci_upper"].values, 2).tolist()},
        "holiday_coefs": [{"Holiday_Dummy": str(k), "Holiday_ID": int(str(k).split("_")[1]), "Efek_MWh": round(float(v), 2)} for k, v in hol_coefs.items()],
    }

    # Placebo 2019 & Counterfactual 2020
    placebo_mask = (df["datetime"] >= "2019-03-01") & (df["datetime"] <= "2019-06-30 23:00:00")
    diff_pl = df.loc[placebo_mask, "nat_demand"] - df.loc[placebo_mask, "pred_gam"]
    placebo_2019 = {
        "bias_mwh": round(float(diff_pl.mean()), 2),
        "bias_pct": round(float((diff_pl / df.loc[placebo_mask, "pred_gam"]).mean() * 100.0), 2),
        "mae_mwh": round(float(np.abs(diff_pl).mean()), 2),
    }

    pre_covid_mask = df["datetime"] < covid_start_dt
    res_gam_pre, gam_bounds_pre, gam_cols_pre = fit_panama_gam(pre_covid_mask)
    ex_lin_20, ex_sm_20 = build_gam_inputs(df.loc[test_2020_mask], clip_bounds=gam_bounds_pre)
    cf_20 = res_gam_pre.get_prediction(exog=ex_lin_20[gam_cols_pre], exog_smooth=ex_sm_20).summary_frame()

    df["cf_mean"] = np.nan
    df["cf_ci_lo"] = np.nan
    df["cf_ci_hi"] = np.nan
    df.loc[test_2020_mask, "cf_mean"] = cf_20["mean"].values
    df.loc[test_2020_mask, "cf_ci_lo"] = cf_20["mean_ci_lower"].values
    df.loc[test_2020_mask, "cf_ci_hi"] = cf_20["mean_ci_upper"].values

    df_20 = df.loc[test_2020_mask, ["datetime", "month", "hour", "is_weekend", "nat_demand", "cf_mean"]].copy()
    df_20["delta_mwh"] = df_20["nat_demand"] - df_20["cf_mean"]
    df_20["delta_pct"] = df_20["delta_mwh"] / df_20["cf_mean"] * 100.0
    monthly_cf = (
        df_20.groupby("month")
        .agg(
            Aktual_Mean_MWh=("nat_demand", "mean"),
            Counterfactual_Mean_MWh=("cf_mean", "mean"),
            Delta_Mean_MWh=("delta_mwh", "mean"),
            Delta_Persen=("delta_pct", "mean"),
            Total_Energi_Hilang_GWh=("delta_mwh", lambda s: s.sum() / 1000.0),
        )
        .round(2)
        .reset_index()
    )
    mar_jun = df_20[df_20["month"] >= 3]
    covid_summary = {
        "total_lost_gwh": round(float(mar_jun["delta_mwh"].sum() / 1000.0), 2),
        "mean_delta_mwh": round(float(mar_jun["delta_mwh"].mean()), 2),
        "mean_delta_pct": round(float(mar_jun["delta_pct"].mean()), 2),
    }

    # ------------------------------------------------------------------
    # Stage 5: Gradient Boosting + 3-Fold CV + Quantile Loss + XAI
    # ------------------------------------------------------------------
    print("Stage 5: Running Gradient Boosting 3-fold CV, Point+Quantile training, & XAI...", flush=True)
    gbm_name = "LightGBM" if HAS_LGBM else "HistGradientBoosting"
    train_b_mask = train_mask & df[set_b_features].notna().all(axis=1)
    X_train_b = df.loc[train_b_mask, set_b_features]
    y_train_b = df.loc[train_b_mask, "nat_demand_clean"]
    dt_train_b = df.loc[train_b_mask, "datetime"]
    X_test_19 = df.loc[test_2019_mask, set_b_features]
    X_test_20 = df.loc[test_2020_mask, set_b_features]

    tscv = TimeSeriesSplit(n_splits=3)
    cv_gbm_rows = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_b), 1):
        m_cv = make_gbm("squared", lr=0.05, max_iter=350)
        m_cv.fit(X_train_b.iloc[tr_idx], y_train_b.iloc[tr_idx])
        p_cv = m_cv.predict(X_train_b.iloc[val_idx])
        met_cv = compute_metrics(y_train_b.iloc[val_idx], p_cv)
        cv_gbm_rows.append({
            "Fold": fold,
            "Train_Period": f"{dt_train_b.iloc[tr_idx[0]].strftime('%Y-%m-%d')} s.d. {dt_train_b.iloc[tr_idx[-1]].strftime('%Y-%m-%d')}",
            "Val_Period": f"{dt_train_b.iloc[val_idx[0]].strftime('%Y-%m-%d')} s.d. {dt_train_b.iloc[val_idx[-1]].strftime('%Y-%m-%d')}",
            "N_Train": int(len(tr_idx)),
            "N_Val": int(len(val_idx)),
            "MAE": round(met_cv["MAE"], 2),
            "RMSE": round(met_cv["RMSE"], 2),
            "MAPE_%": round(met_cv["MAPE"], 2),
        })

    model_pt = make_gbm("squared")
    model_q05 = make_gbm("quantile", alpha_q=0.05)
    model_q50 = make_gbm("quantile", alpha_q=0.50)
    model_q95 = make_gbm("quantile", alpha_q=0.95)
    model_pt.fit(X_train_b, y_train_b)
    model_q05.fit(X_train_b, y_train_b)
    model_q50.fit(X_train_b, y_train_b)
    model_q95.fit(X_train_b, y_train_b)

    for m_mask, X_m in [(test_2019_mask, X_test_19), (test_2020_mask, X_test_20)]:
        df.loc[m_mask, "pred_gbm"] = model_pt.predict(X_m)
        df.loc[m_mask, "pred_gbm_q05"] = model_q05.predict(X_m)
        df.loc[m_mask, "pred_gbm_q50"] = model_q50.predict(X_m)
        df.loc[m_mask, "pred_gbm_q95"] = model_q95.predict(X_m)

    gbm_prob_rows = []
    for lbl, msk in [("2019 (All)", test_2019_mask), ("2019 (No Blackout)", mask_2019_clean), ("Mar-Jun 2020 (COVID)", lockdown_2020_mask)]:
        yt = df.loc[msk, "nat_demand"]
        yp = df.loc[msk, "pred_gbm"]
        ql = df.loc[msk, "pred_gbm_q05"]
        qm = df.loc[msk, "pred_gbm_q50"]
        qh = df.loc[msk, "pred_gbm_q95"]
        met = compute_metrics(yt, yp)
        avg_pb = float(np.mean([pinball_loss(yt, ql, 0.05), pinball_loss(yt, qm, 0.50), pinball_loss(yt, qh, 0.95)]))
        picp = float(np.mean((yt >= ql) & (yt <= qh)) * 100.0)
        ws = winkler_score(yt, ql, qh, alpha=0.10)
        gbm_prob_rows.append({
            "Periode": lbl,
            "MAE": round(met["MAE"], 2),
            "RMSE": round(met["RMSE"], 2),
            "MAPE_%": round(met["MAPE"], 2),
            "Pinball": round(avg_pb, 2),
            "PICP_90_%": round(picp, 1),
            "Winkler": round(ws, 2),
        })

    # XAI (Permutation Importance + PDP for T2M_toc)
    X_19_sample = X_test_19.iloc[::4].copy()
    y_19_sample = df.loc[test_2019_mask, "nat_demand"].iloc[::4].copy()

    def map_feature_group(col):
        if col.startswith("load_"):
            return "1. Lag Beban Historis"
        if any(col.startswith(p) for p in ["T2M_", "QV2M_", "TQL_", "W2M_"]):
            return "2. Cuaca 3 Kota"
        if col in ["holiday", "Holiday_ID", "school", "pre_holiday", "post_holiday"]:
            return "3. Kalender (Holiday/School)"
        return "4. Waktu Siklikal & Tren"

    if HAS_SHAP and HAS_LGBM:
        import shap
        shap_vals = shap.TreeExplainer(model_pt).shap_values(X_19_sample)
        imp_series = pd.Series(np.abs(shap_vals).mean(axis=0), index=set_b_features)
        method_label = "Mean |SHAP| (MWh)"
    else:
        perm = permutation_importance(model_pt, X_19_sample, y_19_sample, n_repeats=5, random_state=42, scoring="neg_root_mean_squared_error")
        imp_series = pd.Series(perm.importances_mean, index=set_b_features)
        method_label = "Permutation Importance (Delta RMSE MWh)"

    grouped_imp = imp_series.groupby([map_feature_group(c) for c in imp_series.index]).sum().sort_values(ascending=True)
    top10_feat = imp_series.sort_values(ascending=True).tail(10)
    pdp_res = partial_dependence(model_pt, X_19_sample, features=["T2M_toc"], grid_resolution=40)
    xai_gbm = {
        "method_label": method_label,
        "grouped": [{"Kelompok": str(k), "Importance": round(float(v), 3)} for k, v in grouped_imp.items()],
        "top10": [{"Fitur": str(k), "Importance": round(float(v), 3)} for k, v in top10_feat.items()],
        "pdp_t2m_toc": {
            "x": np.round(pdp_res["grid_values"][0], 3).tolist(),
            "y": np.round(pdp_res["average"][0], 2).tolist(),
        },
    }

    # ------------------------------------------------------------------
    # Stage 7: Full 5-Model Backtesting Comparison & Diebold-Mariano
    # ------------------------------------------------------------------
    print("Stage 7: Computing 5-model backtest summary & ACF error curves...", flush=True)
    models_eval = [
        ("Seasonal Naive (Y_t-168)", "pred_naive", "Baseline", "Jangka Pendek (h=168j)"),
        ("SARIMA(1,0,1)(1,1,0)_24", "pred_sarima", "Statistik Univariat", "Jangka Pendek (h=168j)"),
        ("SARIMAX (Linear)", "pred_sarimax", "Statistik Multivariat", "Jangka Pendek (h=168j)"),
        ("GAM (Set A Eksogen)", "pred_gam", "Spline Non-Linear", "Jangka Panjang / Eksogen (1 thn)"),
        (f"{gbm_name} (Set B h=168)", "pred_gbm", "Gradient Boosted Trees", "Jangka Pendek (h=168j)"),
    ]
    summary_rows = []
    y_19_cln = df.loc[mask_2019_clean, "nat_demand"].values
    pred_sx_19 = df.loc[mask_2019_clean, "pred_sarimax"].values
    pred_nv_19 = df.loc[mask_2019_clean, "pred_naive"].values
    for m_name, col, paradigm, horizon_cat in models_eval:
        m19_all = compute_metrics(df.loc[test_2019_mask, "nat_demand"], df.loc[test_2019_mask, col])
        m19_cln = compute_metrics(y_19_cln, df.loc[mask_2019_clean, col])
        m20_cov = compute_metrics(df.loc[lockdown_2020_mask, "nat_demand"], df.loc[lockdown_2020_mask, col])
        err_19 = y_19_cln - df.loc[mask_2019_clean, col].values
        lb_p = acorr_ljungbox(err_19, lags=[168], return_df=True).loc[168, "lb_pvalue"]
        _, arch_p_e, _, _ = het_arch(err_19, nlags=24)
        if col == "pred_sarimax":
            dm_sx_str, dm_sx_val = "Ref", 0.0
        else:
            dm_s, dm_p, _ = diebold_mariano_test(y_19_cln, pred_sx_19, df.loc[mask_2019_clean, col].values, h=168)
            dm_sx_str, dm_sx_val = f"{dm_s:+.2f} (p={dm_p:.2e})", round(dm_s, 2)
        if col == "pred_naive":
            dm_nv_str, dm_nv_val = "Ref", 0.0
        else:
            dm_s2, dm_p2, _ = diebold_mariano_test(y_19_cln, pred_nv_19, df.loc[mask_2019_clean, col].values, h=168)
            dm_nv_str, dm_nv_val = f"{dm_s2:+.2f} (p={dm_p2:.2e})", round(dm_s2, 2)
        summary_rows.append({
            "Model": m_name,
            "col_key": col,
            "Paradigma": paradigm,
            "Kategori_Forecast": horizon_cat,
            "MAE_2019_All": round(m19_all["MAE"], 2),
            "RMSE_2019_All": round(m19_all["RMSE"], 2),
            "MAPE_2019_All_%": round(m19_all["MAPE"], 2),
            "MAE_2019_NoBO": round(m19_cln["MAE"], 2),
            "RMSE_2019_NoBO": round(m19_cln["RMSE"], 2),
            "MAPE_2019_NoBO_%": round(m19_cln["MAPE"], 2),
            "MAPE_MarJun2020_%": round(m20_cov["MAPE"], 2),
            "LjungBox_lag168_p": f"{lb_p:.2e}",
            "ARCH_LM_p": f"{arch_p_e:.2e}",
            "DM_vs_SARIMAX": dm_sx_str,
            "DM_vs_SARIMAX_val": dm_sx_val,
            "DM_vs_Naive": dm_nv_str,
            "DM_vs_Naive_val": dm_nv_val,
        })

    acf_errors_2019 = {"lags": list(range(1, 175))}
    for key, col in [("sarimax", "pred_sarimax"), ("gam", "pred_gam"), ("gbm", "pred_gbm")]:
        e_vec = (df.loc[mask_2019_clean, "nat_demand"] - df.loc[mask_2019_clean, col]).values
        acf_errors_2019[key] = np.round(acf(e_vec, nlags=175, fft=True)[1:175], 4).tolist()

    # ------------------------------------------------------------------
    # Stage 8A: Out-of-Dataset Short-Term 7-Day Forecast (168h)
    # ------------------------------------------------------------------
    print("Stage 8A: Refitting full Boosting models & generating 168h forecast...", flush=True)
    df_full_b = df.copy()
    s_full = df_full_b["nat_demand_clean_full"]
    df_full_b["load_lag_168"] = s_full.shift(168)
    df_full_b["load_lag_336"] = s_full.shift(336)
    df_full_b["load_lag_504"] = s_full.shift(504)
    df_full_b["load_roll168_mean"] = s_full.shift(168).rolling(168, min_periods=24).mean()
    df_full_b["load_roll168_std"] = s_full.shift(168).rolling(168, min_periods=24).std()
    df_full_b["load_roll24_max_lag168"] = s_full.shift(168).rolling(24, min_periods=1).max()

    valid_full_b = df_full_b[set_b_features].notna().all(axis=1)
    X_full_b = df_full_b.loc[valid_full_b, set_b_features]
    y_full_b = df_full_b.loc[valid_full_b, "nat_demand_clean_full"]

    model_st_pt = make_gbm("squared")
    model_st_q05 = make_gbm("quantile", alpha_q=0.05)
    model_st_q50 = make_gbm("quantile", alpha_q=0.50)
    model_st_q95 = make_gbm("quantile", alpha_q=0.95)
    model_st_pt.fit(X_full_b, y_full_b)
    model_st_q05.fit(X_full_b, y_full_b)
    model_st_q50.fit(X_full_b, y_full_b)
    model_st_q95.fit(X_full_b, y_full_b)

    future_st_dt = pd.date_range(start=df["datetime"].max() + pd.Timedelta(hours=1), periods=168, freq="h")
    future_st = pd.DataFrame({"datetime": future_st_dt})
    future_st["month"] = future_st["datetime"].dt.month
    future_st["day"] = future_st["datetime"].dt.day
    future_st["hour_int"] = future_st["datetime"].dt.hour

    clim_base = df[df["year"] <= 2019].copy()
    clim_base["day"] = clim_base["datetime"].dt.day
    clim_base["hour_int"] = clim_base["datetime"].dt.hour
    clim_mdh = clim_base.groupby(["month", "day", "hour_int"], as_index=False)[weather_cols].mean()

    future_st = future_st.merge(clim_mdh, on=["month", "day", "hour_int"], how="left")
    t2m_cont = pd.concat([df["T2M_toc"].iloc[-24:], future_st["T2M_toc"]], ignore_index=True)
    for lag in [1, 2, 3]:
        future_st[f"T2M_toc_lag{lag}"] = t2m_cont.shift(lag).iloc[24:].values
    future_st["T2M_toc_roll24"] = t2m_cont.rolling(24, min_periods=1).mean().iloc[24:].values
    future_st["T2M_pca1"] = pca_t2m.transform(scaler_t2m.transform(future_st[t2m_cols])).ravel() * pca_t2m_sign

    future_st["time_index_years"] = (len(df) + np.arange(len(future_st), dtype=float)) / 8760.0
    future_st["hour"] = future_st["datetime"].dt.hour.astype(float)
    future_st["day_of_week"] = future_st["datetime"].dt.dayofweek.astype(float)
    future_st["is_weekend"] = (future_st["day_of_week"] >= 5).astype(float)
    future_st["day_of_year"] = future_st["datetime"].dt.dayofyear.astype(float)
    future_st["sin_hour"] = np.sin(2 * np.pi * future_st["hour"] / 24.0)
    future_st["cos_hour"] = np.cos(2 * np.pi * future_st["hour"] / 24.0)
    future_st["sin_dow"] = np.sin(2 * np.pi * future_st["day_of_week"] / 7.0)
    future_st["cos_dow"] = np.cos(2 * np.pi * future_st["day_of_week"] / 7.0)
    future_st["sin_doy"] = np.sin(2 * np.pi * future_st["day_of_year"] / 365.25)
    future_st["cos_doy"] = np.cos(2 * np.pi * future_st["day_of_year"] / 365.25)
    future_st["holiday"] = 0
    future_st["Holiday_ID"] = 0
    future_st["school"] = 1
    future_st["pre_holiday"] = 0.0
    future_st["post_holiday"] = 0.0

    tail_load = pd.concat([s_full.iloc[-672:], pd.Series(np.nan, index=range(168))], ignore_index=True)
    future_st["load_lag_168"] = tail_load.shift(168).iloc[-168:].values
    future_st["load_lag_336"] = tail_load.shift(336).iloc[-168:].values
    future_st["load_lag_504"] = tail_load.shift(504).iloc[-168:].values
    future_st["load_roll168_mean"] = tail_load.shift(168).rolling(168, min_periods=24).mean().iloc[-168:].values
    future_st["load_roll168_std"] = tail_load.shift(168).rolling(168, min_periods=24).std().iloc[-168:].values
    future_st["load_roll24_max_lag168"] = tail_load.shift(168).rolling(24, min_periods=1).max().iloc[-168:].values

    X_future_st = future_st[set_b_features].copy()
    for col in X_full_b.columns:
        X_future_st[col] = X_future_st[col].astype(X_full_b[col].dtype)
    assert int(X_future_st.isna().sum().sum()) == 0

    future_st["pred_st_mean"] = model_st_pt.predict(X_future_st)
    q_sorted = np.sort(np.column_stack([
        model_st_q05.predict(X_future_st),
        model_st_q50.predict(X_future_st),
        model_st_q95.predict(X_future_st),
    ]), axis=1)
    future_st["pred_st_q05"] = q_sorted[:, 0]
    future_st["pred_st_q50"] = q_sorted[:, 1]
    future_st["pred_st_q95"] = q_sorted[:, 2]
    future_st["pi_width_90"] = future_st["pred_st_q95"] - future_st["pred_st_q05"]

    day_names_id = {0: "Senin", 1: "Selasa", 2: "Rabu", 3: "Kamis", 4: "Jumat", 5: "Sabtu", 6: "Minggu"}
    future_st["Tanggal"] = future_st["datetime"].dt.strftime("%Y-%m-%d")
    future_st["Hari"] = future_st["datetime"].dt.dayofweek.map(day_names_id)
    st_daily_df = (
        future_st.groupby(["Tanggal", "Hari"], sort=False)
        .agg(
            Jumlah_Jam=("pred_st_mean", "count"),
            Mean_MWh=("pred_st_mean", "mean"),
            Min_MWh=("pred_st_mean", "min"),
            Peak_MWh=("pred_st_mean", "max"),
            Q05_Mean=("pred_st_q05", "mean"),
            Q95_Mean=("pred_st_q95", "mean"),
            Total_GWh=("pred_st_mean", lambda s: s.sum() / 1000.0),
        )
        .reset_index()
    )
    for c in ["Mean_MWh", "Min_MWh", "Peak_MWh", "Q05_Mean", "Q95_Mean", "Total_GWh"]:
        st_daily_df[c] = st_daily_df[c].round(2)

    st_summary = {
        "mean_mwh": round(float(future_st["pred_st_mean"].mean()), 2),
        "min_mwh": round(float(future_st["pred_st_mean"].min()), 2),
        "max_mwh": round(float(future_st["pred_st_mean"].max()), 2),
        "total_gwh": round(float(future_st["pred_st_mean"].sum() / 1000.0), 2),
        "mean_pi_width": round(float(future_st["pi_width_90"].mean()), 2),
        "n_train_rows": int(len(X_full_b)),
    }

    # ------------------------------------------------------------------
    # Stage 8B: Out-of-Dataset Long-Term 12-Month Forecast (Jul 2020 - Jun 2021)
    # ------------------------------------------------------------------
    print("Stage 8B: Fitting full GAM with COVID regime & generating 12m forecast...", flush=True)
    future_lt_dt = pd.date_range(start=df["datetime"].max() + pd.Timedelta(hours=1), end="2021-06-30 23:00:00", freq="h")
    future_lt = pd.DataFrame({"datetime": future_lt_dt})
    future_lt["month"] = future_lt["datetime"].dt.month
    future_lt["day"] = future_lt["datetime"].dt.day
    future_lt["hour_int"] = future_lt["datetime"].dt.hour
    future_lt = future_lt.merge(clim_mdh, on=["month", "day", "hour_int"], how="left")

    cal_19 = df[df["year"] == 2019].copy()
    cal_19["day"] = cal_19["datetime"].dt.day
    cal_19["hour_int"] = cal_19["datetime"].dt.hour
    cal_19 = cal_19[["month", "day", "hour_int", "holiday", "Holiday_ID", "school"]].drop_duplicates(subset=["month", "day", "hour_int"])
    future_lt = future_lt.merge(cal_19, on=["month", "day", "hour_int"], how="left")

    future_lt["time_index_years"] = (future_lt["datetime"] - df["datetime"].iloc[0]).dt.total_seconds() / 3600.0 / 8760.0
    future_lt["hour"] = future_lt["datetime"].dt.hour.astype(float)
    future_lt["day_of_week"] = future_lt["datetime"].dt.dayofweek.astype(float)
    future_lt["is_weekend"] = (future_lt["day_of_week"] >= 5).astype(float)
    future_lt["day_of_year"] = future_lt["datetime"].dt.dayofyear.astype(float)
    future_lt["sin_hour"] = np.sin(2 * np.pi * future_lt["hour"] / 24.0)
    future_lt["cos_hour"] = np.cos(2 * np.pi * future_lt["hour"] / 24.0)
    future_lt["sin_dow"] = np.sin(2 * np.pi * future_lt["day_of_week"] / 7.0)
    future_lt["cos_dow"] = np.cos(2 * np.pi * future_lt["day_of_week"] / 7.0)

    daily_hol_lt = future_lt.groupby(future_lt["datetime"].dt.date)["holiday"].max()
    pre_lt = set(pd.Series(daily_hol_lt.index)[daily_hol_lt.shift(-1, fill_value=0).values == 1])
    post_lt = set(pd.Series(daily_hol_lt.index)[daily_hol_lt.shift(1, fill_value=0).values == 1])
    future_lt["pre_holiday"] = future_lt["datetime"].dt.date.isin(pre_lt).astype(float)
    future_lt["post_holiday"] = future_lt["datetime"].dt.date.isin(post_lt).astype(float)

    future_lt["T2M_pca1"] = pca_t2m.transform(scaler_t2m.transform(future_lt[t2m_cols])).ravel() * pca_t2m_sign
    future_lt["T2M_pca1_x_cos_hour"] = future_lt["T2M_pca1"] * future_lt["cos_hour"]
    future_lt["T2M_pca1_x_sin_hour"] = future_lt["T2M_pca1"] * future_lt["sin_hour"]
    for col in hol_dummy_cols:
        hol_id_val = int(col.split("_")[1])
        future_lt[col] = (future_lt["Holiday_ID"] == hol_id_val).astype(float)

    ex_lin_lt_a, ex_sm_lt_a = build_gam_inputs(future_lt, clip_bounds=gam_bounds_pre)
    sf_lt_a = res_gam_pre.get_prediction(exog=ex_lin_lt_a[gam_cols_pre], exog_smooth=ex_sm_lt_a).summary_frame()
    future_lt["lt_base_mean"] = sf_lt_a["mean"].values
    future_lt["lt_base_lo"] = sf_lt_a["mean_ci_lower"].values
    future_lt["lt_base_hi"] = sf_lt_a["mean_ci_upper"].values

    df_lt_fit = df.copy()
    df_lt_fit["is_covid_regime"] = (df_lt_fit["datetime"] >= covid_start_dt).astype(float)
    res_gam_covid, gam_bounds_cov, gam_cols_cov = fit_panama_gam(
        pd.Series(True, index=df_lt_fit.index),
        extra_lin_cols=["is_covid_regime"],
        target_col="nat_demand_clean_full",
        df_source=df_lt_fit,
    )
    cov_coef = float(res_gam_covid.params["is_covid_regime"])
    cov_pval = float(res_gam_covid.pvalues["is_covid_regime"])

    future_lt["is_covid_regime"] = 1.0
    ex_lin_lt_b, ex_sm_lt_b = build_gam_inputs(future_lt, clip_bounds=gam_bounds_cov, extra_lin_cols=["is_covid_regime"])
    sf_lt_b = res_gam_covid.get_prediction(exog=ex_lin_lt_b[gam_cols_cov], exog_smooth=ex_sm_lt_b).summary_frame()
    future_lt["lt_cov_mean"] = sf_lt_b["mean"].values
    future_lt["lt_cov_lo"] = sf_lt_b["mean_ci_lower"].values
    future_lt["lt_cov_hi"] = sf_lt_b["mean_ci_upper"].values

    lt_12m = future_lt[future_lt["datetime"] >= "2020-07-01 00:00:00"].copy().reset_index(drop=True)
    lt_12m["Bulan"] = lt_12m["datetime"].dt.to_period("M").astype(str)
    ref_19_m = (df.loc[test_2019_mask].groupby("month")["nat_demand"].sum() / 1000.0).to_dict()

    lt_monthly_df = (
        lt_12m.groupby("Bulan", sort=False)
        .agg(
            month_num=("month", "first"),
            Jam=("lt_base_mean", "count"),
            Normal_Mean_MWh=("lt_base_mean", "mean"),
            Normal_CI_Lo=("lt_base_lo", "mean"),
            Normal_CI_Hi=("lt_base_hi", "mean"),
            Normal_Energi_GWh=("lt_base_mean", lambda s: s.sum() / 1000.0),
            COVID_Mean_MWh=("lt_cov_mean", "mean"),
            COVID_Energi_GWh=("lt_cov_mean", lambda s: s.sum() / 1000.0),
        )
        .reset_index()
    )
    lt_monthly_df["Ref_2019_GWh"] = [round(float(ref_19_m[m]), 2) for m in lt_monthly_df["month_num"]]
    lt_monthly_df["Selisih_GWh"] = lt_monthly_df["COVID_Energi_GWh"] - lt_monthly_df["Normal_Energi_GWh"]
    lt_monthly_df["Normal_CI95_MWh"] = [f"[{lo:,.1f}, {hi:,.1f}]" for lo, hi in zip(lt_monthly_df["Normal_CI_Lo"], lt_monthly_df["Normal_CI_Hi"])]
    for c in ["Normal_Mean_MWh", "Normal_CI_Lo", "Normal_CI_Hi", "Normal_Energi_GWh", "COVID_Mean_MWh", "COVID_Energi_GWh", "Selisih_GWh"]:
        lt_monthly_df[c] = lt_monthly_df[c].round(2)

    norm_tot_gwh = float(lt_12m["lt_base_mean"].sum() / 1000.0)
    cov_tot_gwh = float(lt_12m["lt_cov_mean"].sum() / 1000.0)
    actual_19_gwh = float(df.loc[test_2019_mask, "nat_demand"].sum() / 1000.0)
    actual_19_mean = float(df.loc[test_2019_mask, "nat_demand"].mean())

    lt_summary = {
        "cov_coef_mwh": round(cov_coef, 2),
        "cov_pval": cov_pval,
        "ref_2019_mean_mwh": round(actual_19_mean, 2),
        "ref_2019_total_gwh": round(actual_19_gwh, 2),
        "normal_mean_mwh": round(float(lt_12m["lt_base_mean"].mean()), 2),
        "normal_peak_mwh": round(float(lt_12m["lt_base_mean"].max()), 2),
        "normal_total_gwh": round(norm_tot_gwh, 2),
        "normal_total_twh": round(norm_tot_gwh / 1000.0, 3),
        "covid_mean_mwh": round(float(lt_12m["lt_cov_mean"].mean()), 2),
        "covid_peak_mwh": round(float(lt_12m["lt_cov_mean"].max()), 2),
        "covid_total_gwh": round(cov_tot_gwh, 2),
        "covid_total_twh": round(cov_tot_gwh / 1000.0, 3),
        "diff_gwh": round(cov_tot_gwh - norm_tot_gwh, 2),
        "diff_pct": round((cov_tot_gwh - norm_tot_gwh) / norm_tot_gwh * 100.0, 2),
    }

    # ------------------------------------------------------------------
    # Save Parquet/CSV.gz, JSON, and Pickled Models + Preprocessors
    # ------------------------------------------------------------------
    master_cols = (
        ["datetime", "year", "month", "hour", "day_of_week", "is_weekend", "day_of_year",
         "nat_demand", "nat_demand_clean", "nat_demand_clean_full", "is_blackout", "regime"]
        + weather_cols
        + ["T2M_pca1", "holiday", "Holiday_ID", "school",
           "pred_naive", "pred_sarima", "pred_sarimax", "pred_gam",
           "pred_gbm", "pred_gbm_q05", "pred_gbm_q50", "pred_gbm_q95",
           "cf_mean", "cf_ci_lo", "cf_ci_hi"]
    )
    save_table(df[master_cols], "master_hourly")

    st_cols = [
        "datetime", "Tanggal", "Hari", "hour", "is_weekend",
        "pred_st_mean", "pred_st_q05", "pred_st_q50", "pred_st_q95", "pi_width_90",
    ] + weather_cols + ["T2M_pca1"]
    save_table(future_st[st_cols], "forecast_st_168h")

    lt_cols = [
        "datetime", "Bulan", "month", "day", "hour", "day_of_week", "is_weekend", "day_of_year",
        "time_index_years", "sin_hour", "cos_hour", "sin_dow", "cos_dow",
        "holiday", "Holiday_ID", "school", "pre_holiday", "post_holiday",
    ] + t2m_cols + ["QV2M_toc", "T2M_pca1", "T2M_pca1_x_cos_hour", "T2M_pca1_x_sin_hour"] + hol_dummy_cols + [
        "lt_base_mean", "lt_base_lo", "lt_base_hi", "lt_cov_mean", "lt_cov_lo", "lt_cov_hi",
    ]
    save_table(lt_12m[lt_cols], "forecast_lt_12m")

    # Weekly reference templates for What-If simulator in Boosting mode
    norm_ref_week = df_full_b[
        (df_full_b["datetime"] >= "2019-05-13 00:00:00") & (df_full_b["datetime"] <= "2019-05-19 23:00:00")
    ].copy()
    cov_ref_week = df_full_b[
        (df_full_b["datetime"] >= "2020-04-13 00:00:00") & (df_full_b["datetime"] <= "2020-04-19 23:00:00")
    ].copy()
    baseline_hourly_2019 = (
        df[test_2019_mask & (df["is_blackout"] == 0) & (df["holiday"] == 0) & (df["is_weekend"] == 0)]
        .groupby("hour")["nat_demand"]
        .mean()
        .round(2)
        .to_dict()
    )

    audit_stats = {
        "dataset_meta": {
            "rows": int(len(df)),
            "cols": int(17),
            "start": str(df["datetime"].min()),
            "end": str(df["datetime"].max()),
            "nat_demand_mean": round(float(df["nat_demand"].mean()), 2),
            "nat_demand_std": round(float(df["nat_demand"].std()), 2),
            "nat_demand_min": round(float(df["nat_demand"].min()), 2),
            "nat_demand_max": round(float(df["nat_demand"].max()), 2),
            "gbm_engine": gbm_name,
        },
        "deskripsi": desc.to_dict(orient="records"),
        "yearly_stats": yearly_stats.to_dict(orient="records"),
        "mj_yoy_cmp": mj_cmp.to_dict(orient="records"),
        "blackouts": blackouts_df.to_dict(orient="records"),
        "stationarity": stat_rows,
        "fft_top_peaks": fft_top_peaks,
        "fft_spectrum": fft_spectrum,
        "acf_demand": np.round(acf_raw, 4).tolist(),
        "vif_table": vif_df.to_dict(orient="records"),
        "heteroskedasticity": heteroskedasticity,
        "split_summary": split_summary,
        "sarima_aic": round(float(res_sarima.aic), 1),
        "sarimax_aic": round(float(res_sarimax.aic), 1),
        "sarima_diagnostics": sarima_diag_rows,
        "sarimax_holiday_breakdown": hol_err_rows,
        "sarimax_hourly_rmse": [round(float(hourly_rmse_sx.loc[h]), 2) for h in range(24)],
        "cv_gam": cv_gam_rows,
        "cv_gbm": cv_gbm_rows,
        "gam_meta": {
            "edf": round(float(res_gam_18.edf.sum()), 2),
            "aic": round(float(res_gam_18.aic), 1),
            "pseudo_r2": round(float(1.0 - res_gam_18.deviance / res_gam_18.null_deviance), 4),
            "pca_t2m_var_pct": round(float(pca_t2m.explained_variance_ratio_[0] * 100.0), 2),
        },
        "gam_partial_curves": gam_partial_curves,
        "placebo_2019": placebo_2019,
        "covid_monthly_cf": monthly_cf.to_dict(orient="records"),
        "covid_summary": covid_summary,
        "gbm_probabilistic": gbm_prob_rows,
        "xai_gbm": xai_gbm,
        "backtest_summary": summary_rows,
        "acf_errors_2019": acf_errors_2019,
        "st_summary": st_summary,
        "st_daily": st_daily_df.to_dict(orient="records"),
        "lt_summary": lt_summary,
        "lt_monthly": lt_monthly_df.to_dict(orient="records"),
        "baseline_hourly_2019": {int(k): float(v) for k, v in baseline_hourly_2019.items()},
    }

    with open(os.path.join(ART_DIR, "audit_stats.json"), "w", encoding="utf-8") as f:
        json.dump(audit_stats, f, ensure_ascii=False, indent=2)

    joblib.dump(
        {"model": res_gam_pre, "bounds": gam_bounds_pre, "exog_cols": gam_cols_pre},
        os.path.join(ART_DIR, "model_gam_pre.pkl"),
        compress=3,
    )
    joblib.dump(
        {"model": res_gam_covid, "bounds": gam_bounds_cov, "exog_cols": gam_cols_cov, "cov_coef": cov_coef},
        os.path.join(ART_DIR, "model_gam_covid.pkl"),
        compress=3,
    )
    joblib.dump(
        {
            "engine": gbm_name,
            "model_pt": model_st_pt,
            "model_q05": model_st_q05,
            "model_q50": model_st_q50,
            "model_q95": model_st_q95,
            "set_b_features": set_b_features,
            "feature_dtypes": {c: str(X_full_b[c].dtype) for c in set_b_features},
        },
        os.path.join(ART_DIR, "model_gbm_full.pkl"),
        compress=3,
    )
    joblib.dump(
        {
            "scaler_t2m": scaler_t2m,
            "pca_t2m": pca_t2m,
            "pca_t2m_sign": pca_t2m_sign,
            "t2m_cols": t2m_cols,
            "weather_cols": weather_cols,
            "smooth_cols": smooth_cols,
            "gam_lin_cols": gam_lin_cols,
            "hol_dummy_cols": hol_dummy_cols,
            "clim_mdh": clim_mdh,
            "train_start_dt": str(df["datetime"].iloc[0]),
            "norm_ref_week": norm_ref_week[["day_of_week", "hour"] + weather_cols + ["load_lag_168", "load_lag_336", "load_lag_504", "load_roll168_mean", "load_roll168_std", "load_roll24_max_lag168"]].reset_index(drop=True),
            "cov_ref_week": cov_ref_week[["day_of_week", "hour"] + weather_cols + ["load_lag_168", "load_lag_336", "load_lag_504", "load_roll168_mean", "load_roll168_std", "load_roll24_max_lag168"]].reset_index(drop=True),
            "cal_19": cal_19.reset_index(drop=True),
        },
        os.path.join(ART_DIR, "preprocessors.pkl"),
        compress=3,
    )

    # Smoke-test unpickling and prediction on 2 rows
    g_pre_chk = joblib.load(os.path.join(ART_DIR, "model_gam_pre.pkl"))
    g_cov_chk = joblib.load(os.path.join(ART_DIR, "model_gam_covid.pkl"))
    gbm_chk = joblib.load(os.path.join(ART_DIR, "model_gbm_full.pkl"))
    ex_l_chk, ex_s_chk = build_gam_inputs(lt_12m.iloc[:2], clip_bounds=g_pre_chk["bounds"])
    _ = g_pre_chk["model"].get_prediction(exog=ex_l_chk[g_pre_chk["exog_cols"]], exog_smooth=ex_s_chk).summary_frame()
    lt_cov_sample = lt_12m.iloc[:2].copy()
    lt_cov_sample["is_covid_regime"] = 1.0
    ex_lc_chk, ex_sc_chk = build_gam_inputs(lt_cov_sample, clip_bounds=g_cov_chk["bounds"], extra_lin_cols=["is_covid_regime"])
    _ = g_cov_chk["model"].get_prediction(exog=ex_lc_chk[g_cov_chk["exog_cols"]], exog_smooth=ex_sc_chk).summary_frame()
    _ = gbm_chk["model_pt"].predict(X_future_st.iloc[:2])

    print(f"All artifacts built and smoke-tested in {time.time() - t_start:.2f}s!", flush=True)
    print("Key Parity Checks:", flush=True)
    for row in summary_rows:
        print(f"  {row['Model']:35s} | RMSE_19_NoBO={row['RMSE_2019_NoBO']:6.2f} | MAPE_19_NoBO={row['MAPE_2019_NoBO_%']:5.2f}% | DM_vs_SX={row['DM_vs_SARIMAX']}", flush=True)
    print(f"  CV GAM Mean RMSE     : {np.mean([r['RMSE'] for r in cv_gam_rows]):.2f} MWh", flush=True)
    print(f"  CV Boosting Mean RMSE: {np.mean([r['RMSE'] for r in cv_gbm_rows]):.2f} MWh", flush=True)
    print(f"  COVID Mar-Jun 2020   : {covid_summary['total_lost_gwh']} GWh ({covid_summary['mean_delta_pct']}%)", flush=True)
    print(f"  Tahap 8A (7d)        : Mean={st_summary['mean_mwh']} MWh | Peak={st_summary['max_mwh']} MWh | Total={st_summary['total_gwh']} GWh", flush=True)
    print(f"  Tahap 8B (12m)       : Normal={lt_summary['normal_total_gwh']} GWh | COVID={lt_summary['covid_total_gwh']} GWh ({lt_summary['diff_pct']}%)", flush=True)


if __name__ == "__main__":
    main()
