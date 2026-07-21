"""Tests del pipeline. Corren 100% sobre datos sintéticos (rápidos, sin Kaggle)."""
import numpy as np
import pytest

import pandas as pd

from src import config
from src.data import _cap_income_outliers, _clean_sentinels, make_synthetic
from src.evaluate import (
    calibration_metrics,
    cost_curve,
    expected_calibration_error,
    ks_statistic,
    optimal_threshold,
)
from src.features import add_domain_features, build_xy, feature_lists
from src.train import cv_auc, make_baseline, split


@pytest.fixture(scope="module")
def df():
    return make_synthetic(n=5000)


# ------------------------------------------------------------------ datos
def test_schema(df):
    assert config.TARGET in df.columns and config.ID_COL in df.columns
    assert df[config.ID_COL].is_unique
    assert set(df[config.TARGET].unique()) <= {0, 1}


def test_default_rate_plausible(df):
    rate = df[config.TARGET].mean()
    assert 0.03 < rate < 0.20, f"tasa de default implausible: {rate:.3f}"


def test_days_employed_sentinel_becomes_nan():
    """365243 (~1000 años) es el sentinel de nulo del dataset real, no un valor real."""
    raw = pd.DataFrame(
        {config.ID_COL: [1, 2], "DAYS_EMPLOYED": [config.DAYS_EMPLOYED_SENTINEL, -1000]}
    )
    cleaned = _clean_sentinels(raw)
    assert np.isnan(cleaned.loc[0, "DAYS_EMPLOYED"])
    assert cleaned.loc[1, "DAYS_EMPLOYED"] == -1000


def test_amt_income_outlier_is_capped_not_dropped():
    """117M es un error de captura casi seguro; recortar, no perder la fila."""
    raw = pd.DataFrame(
        {config.ID_COL: [1, 2], "AMT_INCOME_TOTAL": [117_000_000.0, 90_000.0]}
    )
    capped = _cap_income_outliers(raw)
    assert capped.loc[0, "AMT_INCOME_TOTAL"] == config.AMT_INCOME_CAP
    assert capped.loc[1, "AMT_INCOME_TOTAL"] == 90_000.0
    assert len(capped) == len(raw)  # ninguna fila se elimina


# --------------------------------------------------------------- features
def test_domain_features_no_leakage_rowwise(df):
    """Las features deben ser función de la fila: mismo resultado en subconjuntos."""
    full = add_domain_features(df)
    half = add_domain_features(df.iloc[: len(df) // 2])
    common = half.index
    assert np.allclose(
        full.loc[common, "CREDIT_INCOME_RATIO"],
        half["CREDIT_INCOME_RATIO"],
        equal_nan=True,
    )


def test_build_xy_shapes(df):
    X, y = build_xy(df)
    numeric, categorical = feature_lists()
    assert list(X.columns) == numeric + categorical
    assert len(X) == len(y) == len(df)
    assert config.TARGET not in X.columns  # el target nunca entra como feature


# --------------------------------------------------------------- métricas
def test_ks_bounds():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    random_ = np.array([0.5, 0.5, 0.5, 0.5])
    assert ks_statistic(y, perfect) == pytest.approx(1.0)
    assert ks_statistic(y, random_) <= 0.5


def test_calibration_perfect_vs_miscalibrated():
    """ECE debe ser bajo cuando la predicción coincide con la tasa real por
    grupo, y alto cuando la predicción está sistemáticamente invertida."""
    rng = np.random.default_rng(0)
    n = 3000

    p = rng.uniform(0.05, 0.95, n)
    y_calibrated = rng.binomial(1, p)
    ece_good = expected_calibration_error(y_calibrated, p, n_bins=10)

    p_bad = rng.uniform(0.05, 0.95, n)
    y_miscalibrated = rng.binomial(1, 1 - p_bad)  # tasa real es la inversa de lo predicho
    ece_bad = expected_calibration_error(y_miscalibrated, p_bad, n_bins=10)

    assert ece_good < 0.05
    assert ece_bad > 0.3
    assert ece_good < ece_bad


def test_calibration_metrics_bounds():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.1, 1000)
    p = np.clip(y * 0.6 + rng.normal(0.1, 0.1, 1000), 0, 1)
    metrics = calibration_metrics(y, p)
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["ece"] >= 0.0


def test_cost_curve_monotone_extremes():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.1, 1000)
    p = np.clip(y * 0.6 + rng.normal(0.3, 0.1, 1000), 0, 1)
    curve = cost_curve(y, p)
    # umbral 0 = rechazar a todos -> solo FP; umbral 1 = aprobar a todos -> solo FN
    assert curve.iloc[0]["fn"] == 0
    assert curve.iloc[-1]["fp"] == 0
    best = optimal_threshold(y, p)
    assert 0 < best["threshold"] < 1


# ------------------------------------------------------------- end-to-end
def test_baseline_beats_random(df):
    X, y = build_xy(df)
    X_train, _, y_train, _ = split(X, y)
    auc, _ = cv_auc(make_baseline(), X_train, y_train)
    assert auc > 0.65, f"AUC {auc:.3f}: el baseline debería superar ampliamente el azar"
