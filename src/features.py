"""Feature engineering: funciones puras, testeables, sin estado global.

Regla anti-leakage: ninguna transformación aquí usa estadísticos del conjunto
completo; los imputadores/encoders con estado viven dentro del Pipeline de
sklearn (src/train.py) y se ajustan solo con el fold de entrenamiento.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios financieros estándar de scoring. Operan fila a fila (sin leakage)."""
    out = df.copy()
    eps = 1e-9

    out["CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / (out["AMT_INCOME_TOTAL"] + eps)
    out["ANNUITY_INCOME_RATIO"] = out["AMT_ANNUITY"] / (out["AMT_INCOME_TOTAL"] + eps)
    out["CREDIT_TERM"] = out["AMT_ANNUITY"] / (out["AMT_CREDIT"] + eps)
    out["GOODS_CREDIT_RATIO"] = out["AMT_GOODS_PRICE"] / (out["AMT_CREDIT"] + eps)

    out["AGE_YEARS"] = (-out["DAYS_BIRTH"] / 365).astype(float)
    out["EMPLOYED_YEARS"] = (-out["DAYS_EMPLOYED"] / 365).clip(lower=0).astype(float)
    out["EMPLOYED_AGE_RATIO"] = out["EMPLOYED_YEARS"] / (out["AGE_YEARS"] + eps)

    out["EXT_SOURCES_MEAN"] = out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    out["EXT_SOURCES_MIN"] = out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1)
    out["EXT_SOURCES_NULLS"] = (
        out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isna().sum(axis=1)
    )
    return out


ENGINEERED_NUMERIC = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM",
    "GOODS_CREDIT_RATIO",
    "AGE_YEARS",
    "EMPLOYED_YEARS",
    "EMPLOYED_AGE_RATIO",
    "EXT_SOURCES_MEAN",
    "EXT_SOURCES_MIN",
    "EXT_SOURCES_NULLS",
]


def feature_lists(df: pd.DataFrame | None = None) -> tuple[list[str], list[str]]:
    """(numéricas, categóricas) que entran al modelo.

    Si se pasa `df`, incluye automáticamente las columnas agregadas
    multi-tabla (prefijos BUREAU_/PREV_ de src/aggregates.py).
    """
    numeric = config.NUMERIC_COLS + ENGINEERED_NUMERIC
    if df is not None:
        from .aggregates import aggregate_feature_names

        numeric = numeric + aggregate_feature_names(df)
    return numeric, list(config.CATEGORICAL_COLS)


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = add_domain_features(df)
    numeric, categorical = feature_lists(df)
    X = df[numeric + categorical]
    y = df[config.TARGET].astype(int)
    return X, y
