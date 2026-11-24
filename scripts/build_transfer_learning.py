"""
Fast Domain-Transfer Learning pipeline (Pre-Pandemic Source -> COVID-19 Adaptation -> Lockdown Test).
Reads artifacts/master_hourly.parquet and exports:
  - artifacts/transfer_learning.json
  - artifacts/transfer_learning_series.parquet
"""

import copy
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART_DIR = os.path.join(BASE_DIR, "artifacts")

SOURCE_END = "2020-03-15 23:00:00"
ADAPT_START = "2020-03-16 00:00:00"
ADAPT_END = "2020-04-15 23:00:00"
TEST_START = "2020-04-16 00:00:00"

EXOG_FEATURES = [
    "hour", "sin_hour", "cos_hour",
    "day_of_week", "sin_dow", "cos_dow", "is_weekend",
    "sin_doy", "cos_doy", "holiday", "school",
    "T2M_toc", "QV2M_toc", "TQL_toc", "W2M_toc",
    "T2M_san", "QV2M_san", "TQL_san", "W2M_san",
    "T2M_dav", "QV2M_dav", "T2M_pca1",
]

ADAPTER_FEATURES = [
    "sin_hour", "cos_hour", "sin_2hour", "cos_2hour",
    "is_weekend", "holiday", "school", "T2M_pca1", "QV2M_toc",
]


def _add_tl_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["sin_2hour"] = np.sin(4 * np.pi * out["hour"] / 24.0)
    out["cos_2hour"] = np.cos(4 * np.pi * out["hour"] / 24.0)
    out["sin_dow"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["cos_dow"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
    out["sin_doy"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    return out


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, ref_rmse: float) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err / y_true)) * 100.0)
    bias = float(np.mean(y_pred - y_true))
    reduksi = float((ref_rmse - rmse) / ref_rmse * 100.0) if ref_rmse > 0 else 0.0
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE_%": round(mape, 2),
        "Bias_MWh": round(bias, 2),
        "Reduksi_RMSE_vs_Direct_%": round(reduksi, 2),
    }


def run_transfer_learning_pipeline() -> dict:
    t0 = time.perf_counter()
    df_raw = pd.read_parquet(os.path.join(ART_DIR, "master_hourly.parquet"))
    df = _add_tl_features(df_raw)

    src_df = df[df["datetime"] <= SOURCE_END].copy()
    adapt_df = df[(df["datetime"] >= ADAPT_START) & (df["datetime"] <= ADAPT_END)].copy()
    test_df = df[df["datetime"] >= TEST_START].copy()

    X_src = src_df[EXOG_FEATURES].to_numpy(dtype=float)
    y_src = src_df["nat_demand_clean_full"].to_numpy(dtype=float)

    X_adapt = adapt_df[EXOG_FEATURES].to_numpy(dtype=float)
    y_adapt = adapt_df["nat_demand"].to_numpy(dtype=float)

    X_test = test_df[EXOG_FEATURES].to_numpy(dtype=float)
    y_test = test_df["nat_demand"].to_numpy(dtype=float)

    # 1. Baseline 1: Direct Transfer (Boosting Source-Only)
    gbm_src = HistGradientBoostingRegressor(
        max_iter=180,
        max_depth=6,
        learning_rate=0.06,
        early_stopping=False,
        random_state=42,
    )
    gbm_src.fit(X_src, y_src)
    pred_direct_gbm = gbm_src.predict(X_test)
    ref_rmse = float(np.sqrt(np.mean((y_test - pred_direct_gbm) ** 2)))

    # 2. Baseline 2: Direct Transfer (MLP Source-Only)
    scaler_src = StandardScaler()
    X_src_sc = scaler_src.fit_transform(X_src)
    X_adapt_sc = scaler_src.transform(X_adapt)
    X_test_sc = scaler_src.transform(X_test)

    mlp_src = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=60,
        learning_rate_init=3e-3,
        early_stopping=False,
        random_state=42,
    )
    mlp_src.fit(X_src_sc, y_src)
    pred_direct_mlp = mlp_src.predict(X_test_sc)

    # 3. Baseline 3: Target-Only Scratch (Boosting on 744h only)
    gbm_scratch = HistGradientBoostingRegressor(
        max_iter=120,
        max_depth=5,
        learning_rate=0.06,
        early_stopping=False,
        random_state=42,
    )
    gbm_scratch.fit(X_adapt, y_adapt)
    pred_scratch_gbm = gbm_scratch.predict(X_test)

    # 4. TL-1: Neural Network Parameter Fine-Tuning (MLP warm_start)
    mlp_ft = copy.deepcopy(mlp_src)
    if getattr(mlp_ft, "best_loss_", None) is None:
        mlp_ft.best_loss_ = float(mlp_ft.loss_)
    mlp_ft.set_params(
        warm_start=True,
        early_stopping=False,
        learning_rate_init=8e-4,
        alpha=1e-2,
        max_iter=mlp_src.n_iter_ + 25,
    )
    mlp_ft.fit(X_adapt_sc, y_adapt)
    pred_tl_mlp = mlp_ft.predict(X_test_sc)

    # 5. TL-2: Tree-Based Warm-Start Transfer (Boosting warm_start)
    gbm_tl = copy.deepcopy(gbm_src)
    src_trees = int(gbm_src.n_iter_)
    gbm_tl.set_params(
        warm_start=True,
        early_stopping=False,
        learning_rate=0.05,
        max_iter=src_trees + 60,
    )
    gbm_tl.fit(X_adapt, y_adapt)
    assert gbm_tl.n_iter_ > src_trees, f"Expected warm_start trees > {src_trees}, got {gbm_tl.n_iter_}"
    pred_tl_gbm = gbm_tl.predict(X_test)

    # 6. TL-3: Frozen GAM Backbone + Ridge Residual Adapter
    res_adapt = y_adapt - adapt_df["cf_mean"].to_numpy(dtype=float)
    adapter = Pipeline([
        ("sc", StandardScaler()),
        ("ridge", Ridge(alpha=15.0)),
    ])
    adapter.fit(adapt_df[ADAPTER_FEATURES].to_numpy(dtype=float), res_adapt)
    pred_tl_gam_adapter = test_df["cf_mean"].to_numpy(dtype=float) + adapter.predict(
        test_df[ADAPTER_FEATURES].to_numpy(dtype=float)
    )

    specs = [
        ("Tanpa Adaptasi (Boosting Source)", "Tanpa Adaptasi", "pred_direct_gbm", pred_direct_gbm),
        ("Tanpa Adaptasi (MLP Source)", "Tanpa Adaptasi", "pred_direct_mlp", pred_direct_mlp),
        ("Latih dari Nol di Target (744j)", "Tanpa Transfer", "pred_scratch_gbm", pred_scratch_gbm),
        ("TL-1: Neural Net Fine-Tuning (MLP)", "Transfer Learning", "pred_tl_mlp", pred_tl_mlp),
        ("TL-2: Tree Warm-Start Transfer (Boosting)", "Transfer Learning", "pred_tl_gbm", pred_tl_gbm),
        ("TL-3: Frozen GAM + Residual Adapter", "Transfer Learning", "pred_tl_gam_adapter", pred_tl_gam_adapter),
    ]

    summary_rows = []
    for name, cat, col_key, preds in specs:
        row = {"Model": name, "Kategori": cat, "col_key": col_key}
        row.update(_eval_metrics(y_test, preds, ref_rmse))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("RMSE", ascending=True).reset_index(drop=True)

    series_df = pd.DataFrame({
        "datetime": test_df["datetime"].values,
        "hour": test_df["hour"].values,
        "nat_demand": y_test,
        "pred_direct_gbm": np.round(pred_direct_gbm, 2),
        "pred_direct_mlp": np.round(pred_direct_mlp, 2),
        "pred_scratch_gbm": np.round(pred_scratch_gbm, 2),
        "pred_tl_mlp": np.round(pred_tl_mlp, 2),
        "pred_tl_gbm": np.round(pred_tl_gbm, 2),
        "pred_tl_gam_adapter": np.round(pred_tl_gam_adapter, 2),
    })
    series_df.to_parquet(os.path.join(ART_DIR, "transfer_learning_series.parquet"), index=False)

    tl_only = summary_df[summary_df["Kategori"] == "Transfer Learning"].sort_values("RMSE")
    best_tl = tl_only.iloc[0].to_dict()
    direct_gbm_row = summary_df[summary_df["col_key"] == "pred_direct_gbm"].iloc[0].to_dict()

    hourly_prof = (
        series_df.groupby("hour")[["nat_demand", "pred_direct_gbm", "pred_tl_mlp", "pred_tl_gbm", "pred_tl_gam_adapter"]]
        .mean()
        .round(2)
        .reset_index()
        .to_dict(orient="list")
    )

    payload = {
        "split_meta": {
            "source_range": f"{src_df['datetime'].min().strftime('%Y-%m-%d')} s.d. 2020-03-15",
            "source_rows": int(len(src_df)),
            "adapt_range": "2020-03-16 s.d. 2020-04-15",
            "adapt_rows": int(len(adapt_df)),
            "adapt_pct_of_source": round(len(adapt_df) / len(src_df) * 100.0, 2),
            "test_range": f"2020-04-16 s.d. {test_df['datetime'].max().strftime('%Y-%m-%d')}",
            "test_rows": int(len(test_df)),
            "n_features": len(EXOG_FEATURES),
            "mlp_src_iters": int(mlp_src.n_iter_),
            "mlp_ft_iters": int(mlp_ft.n_iter_),
            "gbm_src_trees": src_trees,
            "gbm_tl_trees": int(gbm_tl.n_iter_),
            "runtime_sec": round(time.perf_counter() - t0, 2),
        },
        "summary": summary_df.to_dict(orient="records"),
        "best_tl": best_tl,
        "direct_gbm_baseline": direct_gbm_row,
        "hourly_profile": hourly_prof,
    }

    with open(os.path.join(ART_DIR, "transfer_learning.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("=== Transfer Learning Domain Adaptation Results (Test: Apr 16 - Jun 27, 2020) ===")
    print(f"Source={len(src_df)}h | Adapt={len(adapt_df)}h | Test={len(test_df)}h | Runtime={payload['split_meta']['runtime_sec']}s")
    print(f"MLP iters: {mlp_src.n_iter_} -> {mlp_ft.n_iter_} | Boosting trees: {src_trees} -> {gbm_tl.n_iter_}")
    print(summary_df[["Model", "Kategori", "MAE", "RMSE", "MAPE_%", "Bias_MWh", "Reduksi_RMSE_vs_Direct_%"]].to_string(index=False))
    return payload


if __name__ == "__main__":
    run_transfer_learning_pipeline()
