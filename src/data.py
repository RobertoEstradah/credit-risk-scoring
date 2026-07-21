"""Carga y validación de datos (tabla principal + tablas secundarias).

`load_data()` lee application_train.csv (Home Credit) si existe; si no, genera
datos sintéticos con el mismo esquema. Lo mismo aplica a bureau.csv y
previous_application.csv para las agregaciones multi-tabla (src/aggregates.py).
El README y results.json declaran SIEMPRE la fuente usada — nada de presentar
resultados sintéticos como reales.

La señal sintética incluye una interacción no lineal (apalancamiento pesa más
cuando el score de buró es bajo) para reproducir el comportamiento del dataset
real, donde los modelos de boosting superan al baseline lineal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Chequeos mínimos de contrato de datos."""
    missing = [c for c in [config.TARGET, config.ID_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas obligatorias ausentes: {missing}")
    if not set(df[config.TARGET].dropna().unique()) <= {0, 1}:
        raise ValueError("TARGET debe ser binario {0,1}")
    if df[config.ID_COL].duplicated().any():
        raise ValueError("IDs duplicados en SK_ID_CURR")
    return df


def make_synthetic(n: int = 20_000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Dataset sintético con el esquema de application_train (tasa default ~8%)."""
    rng = np.random.default_rng(seed)

    income = rng.lognormal(mean=11.9, sigma=0.5, size=n)
    credit = income * rng.uniform(1.5, 8.0, n)
    df = pd.DataFrame(
        {
            config.ID_COL: np.arange(100_000, 100_000 + n),
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": credit / rng.uniform(10, 30, n),
            "AMT_GOODS_PRICE": credit * rng.uniform(0.8, 1.0, n),
            "DAYS_BIRTH": -rng.integers(21 * 365, 68 * 365, n),
            "DAYS_EMPLOYED": -rng.integers(0, 40 * 365, n),
            "EXT_SOURCE_1": rng.beta(3, 2, n),
            "EXT_SOURCE_2": rng.beta(3, 2, n),
            "EXT_SOURCE_3": rng.beta(3, 2, n),
            "CNT_CHILDREN": rng.poisson(0.5, n),
            "REGION_POPULATION_RELATIVE": rng.uniform(0.001, 0.07, n),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
            "CODE_GENDER": rng.choice(["F", "M"], n, p=[0.65, 0.35]),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], n, p=[0.34, 0.66]),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n, p=[0.69, 0.31]),
            "NAME_INCOME_TYPE": rng.choice(
                ["Working", "Commercial associate", "Pensioner", "State servant"],
                n, p=[0.52, 0.23, 0.18, 0.07]),
            "NAME_EDUCATION_TYPE": rng.choice(
                ["Secondary", "Higher education", "Incomplete higher", "Lower secondary"],
                n, p=[0.71, 0.24, 0.03, 0.02]),
            "NAME_FAMILY_STATUS": rng.choice(
                ["Married", "Single", "Civil marriage", "Separated", "Widow"],
                n, p=[0.64, 0.15, 0.10, 0.06, 0.05]),
            "NAME_HOUSING_TYPE": rng.choice(
                ["House / apartment", "With parents", "Rented apartment"],
                n, p=[0.88, 0.07, 0.05]),
            # proporciones aproximadas a las reales (58 categorías reales,
            # aquí solo las más frecuentes + "Other_misc" como cola larga)
            "ORGANIZATION_TYPE": rng.choice(
                ["Business Entity Type 3", "XNA", "Self-employed", "Other", "Medicine",
                 "Business Entity Type 2", "Government", "School", "Trade: type 7",
                 "Kindergarten", "Construction", "Business Entity Type 1",
                 "Transport: type 4", "Other_misc"],
                n, p=[0.221, 0.180, 0.125, 0.054, 0.036, 0.034, 0.034, 0.029,
                      0.025, 0.022, 0.022, 0.019, 0.018, 0.181]),
        }
    )

    # señal con interacción no lineal: el apalancamiento castiga más a buró bajo
    ext_mean = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    leverage = (df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]).clip(0, 12) / 12
    tenure = (-df["DAYS_EMPLOYED"] / 365).clip(0, 40) / 40
    logit = (
        -3.0
        - 6.0 * (ext_mean - 0.6)
        + 1.5 * leverage
        + 4.0 * leverage * np.clip(0.55 - ext_mean, 0, None)  # interacción
        - 1.2 * tenure
        - 1.0 * np.where(tenure < 0.05, -0.6, 0.0)            # umbral: recién empleados
    )
    p = 1 / (1 + np.exp(-logit))
    df[config.TARGET] = rng.binomial(1, p)

    # nulos realistas
    for col, frac in [("EXT_SOURCE_1", 0.44), ("EXT_SOURCE_3", 0.17), ("AMT_ANNUITY", 0.01)]:
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan

    return _validate(df)


def make_synthetic_bureau(app: pd.DataFrame, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Tabla estilo bureau.csv: créditos previos en otras instituciones (1:N)."""
    rng = np.random.default_rng(seed + 1)
    ids = app[config.ID_COL].to_numpy()
    y = app[config.TARGET].to_numpy()
    n_loans = rng.poisson(3 + 0.6 * y, len(ids))  # morosos: historial algo más cargado
    rows_id = np.repeat(ids, n_loans)
    rows_y = np.repeat(y, n_loans)
    m = len(rows_id)
    return pd.DataFrame(
        {
            config.ID_COL: rows_id,
            "DAYS_CREDIT": -rng.integers(30, 3000, m),
            "AMT_CREDIT_SUM": rng.lognormal(12.5, 1.0, m),
            "AMT_CREDIT_SUM_DEBT": rng.lognormal(11.0, 1.2, m) * rng.binomial(1, 0.55, m),
            "CREDIT_DAY_OVERDUE": rng.binomial(1, 0.03 + 0.09 * rows_y, m)
            * rng.integers(1, 120, m),
            "CREDIT_ACTIVE": rng.choice(["Active", "Closed"], m, p=[0.4, 0.6]),
        }
    )


def make_synthetic_prev(app: pd.DataFrame, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Tabla estilo previous_application.csv: solicitudes previas en la casa (1:N)."""
    rng = np.random.default_rng(seed + 2)
    ids = app[config.ID_COL].to_numpy()
    y = app[config.TARGET].to_numpy()
    n_prev = rng.poisson(2, len(ids))
    rows_id = np.repeat(ids, n_prev)
    rows_y = np.repeat(y, n_prev)
    m = len(rows_id)
    p_refused = 0.10 + 0.08 * rows_y  # morosos: algo más de rechazos previos
    status = np.where(
        rng.random(m) < p_refused,
        "Refused",
        rng.choice(["Approved", "Canceled"], m, p=[0.85, 0.15]),
    )
    return pd.DataFrame(
        {
            config.ID_COL: rows_id,
            "AMT_APPLICATION": rng.lognormal(12.0, 0.8, m),
            "AMT_CREDIT": rng.lognormal(12.0, 0.8, m),
            "NAME_CONTRACT_STATUS": status,
            "DAYS_DECISION": -rng.integers(30, 2500, m),
        }
    )


def _clean_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Sentinels de nulo del dataset real que no son NaN explícito en el CSV."""
    df = df.copy()
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
            config.DAYS_EMPLOYED_SENTINEL, np.nan
        )
    return df


def _cap_income_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Recorta AMT_INCOME_TOTAL a un tope fijo (ver config.AMT_INCOME_CAP).

    Cap, no NaN: a diferencia del sentinel de DAYS_EMPLOYED, esto no es un
    código de nulo — es un ingreso real pero implausible. Recortar preserva
    la fila (el resto de las columnas sigue siendo información válida) en
    vez de perderla por completo.
    """
    df = df.copy()
    if "AMT_INCOME_TOTAL" in df.columns:
        df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].clip(upper=config.AMT_INCOME_CAP)
    return df


def load_data() -> tuple[pd.DataFrame, str]:
    """Devuelve (df_principal, fuente) donde fuente ∈ {'kaggle', 'synthetic'}."""
    if config.RAW_FILE.exists():
        cols = [config.ID_COL, config.TARGET] + config.NUMERIC_COLS + config.CATEGORICAL_COLS
        df = pd.read_csv(config.RAW_FILE, usecols=lambda c: c in cols)
        df = _clean_sentinels(df)
        df = _cap_income_outliers(df)
        return _validate(df), "kaggle"
    return make_synthetic(), "synthetic"


def load_secondary(app: pd.DataFrame, source: str) -> dict[str, pd.DataFrame]:
    """Tablas secundarias reales si existen; si no, sintéticas consistentes."""
    out: dict[str, pd.DataFrame] = {}
    bureau_f = config.DATA_DIR / "bureau.csv"
    prev_f = config.DATA_DIR / "previous_application.csv"
    if source == "kaggle" and bureau_f.exists():
        out["bureau"] = pd.read_csv(
            bureau_f,
            usecols=[config.ID_COL, "DAYS_CREDIT", "AMT_CREDIT_SUM",
                     "AMT_CREDIT_SUM_DEBT", "CREDIT_DAY_OVERDUE", "CREDIT_ACTIVE"],
        )
    elif source == "synthetic":
        out["bureau"] = make_synthetic_bureau(app)
    if source == "kaggle" and prev_f.exists():
        out["prev"] = pd.read_csv(
            prev_f,
            usecols=[config.ID_COL, "AMT_APPLICATION", "AMT_CREDIT",
                     "NAME_CONTRACT_STATUS", "DAYS_DECISION"],
        )
    elif source == "synthetic":
        out["prev"] = make_synthetic_prev(app)
    return out
