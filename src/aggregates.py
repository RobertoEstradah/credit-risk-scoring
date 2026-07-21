"""Agregaciones multi-tabla (F2b): bureau y previous_application → 1 fila por cliente.

Patrón estándar de scoring: colapsar historiales 1:N en estadísticos por
SK_ID_CURR y unirlos a la tabla principal con left join. Todas las columnas
generadas llevan prefijo BUREAU_ / PREV_ para que features.py las detecte
automáticamente como numéricas.

Anti-leakage: las agregaciones son por-cliente sobre su propio historial
(pasado del cliente), nunca estadísticos cruzados del dataset.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import config


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    g = bureau.groupby(config.ID_COL)
    out = pd.DataFrame(index=g.size().index)
    out["BUREAU_LOAN_COUNT"] = g.size()
    out["BUREAU_ACTIVE_COUNT"] = g["CREDIT_ACTIVE"].apply(lambda s: (s == "Active").sum())
    out["BUREAU_DEBT_SUM"] = g["AMT_CREDIT_SUM_DEBT"].sum()
    out["BUREAU_CREDIT_SUM"] = g["AMT_CREDIT_SUM"].sum()
    out["BUREAU_DEBT_CREDIT_RATIO"] = out["BUREAU_DEBT_SUM"] / (out["BUREAU_CREDIT_SUM"] + 1e-9)
    out["BUREAU_OVERDUE_MAX"] = g["CREDIT_DAY_OVERDUE"].max()
    out["BUREAU_OVERDUE_ANY"] = (out["BUREAU_OVERDUE_MAX"] > 0).astype(int)
    out["BUREAU_DAYS_CREDIT_MEAN"] = g["DAYS_CREDIT"].mean()
    return out.reset_index()


def aggregate_prev(prev: pd.DataFrame) -> pd.DataFrame:
    g = prev.groupby(config.ID_COL)
    out = pd.DataFrame(index=g.size().index)
    out["PREV_APP_COUNT"] = g.size()
    out["PREV_REFUSED_COUNT"] = g["NAME_CONTRACT_STATUS"].apply(lambda s: (s == "Refused").sum())
    out["PREV_REFUSED_RATE"] = out["PREV_REFUSED_COUNT"] / out["PREV_APP_COUNT"]
    out["PREV_AMT_APPLICATION_MEAN"] = g["AMT_APPLICATION"].mean()
    out["PREV_CREDIT_APPLICATION_RATIO"] = g["AMT_CREDIT"].sum() / (
        g["AMT_APPLICATION"].sum() + 1e-9
    )
    out["PREV_DAYS_DECISION_MEAN"] = g["DAYS_DECISION"].mean()
    return out.reset_index()


# --------------------------------------------------------------------------
# Versión SQL (DuckDB): mismas agregaciones, pero ejecutadas directamente
# sobre los CSV en disco vía SQL en vez de cargar la tabla completa a pandas
# y agrupar en memoria. Útil para tablas grandes (previous_application.csv
# pesa ~400MB) donde no queremos materializar todas las columnas sin usar.
# Se mantiene la versión pandas como la que corre el pipeline; ambas están
# probadas equivalentes (ver tests/test_aggregates_sql.py).
# --------------------------------------------------------------------------

# COALESCE(SUM(...), 0) replica el default de pandas .sum() (skipna=True):
# la suma de un grupo enteramente NULL da 0, no NULL como en SQL estándar.
BUREAU_SQL = """
    SELECT
        SK_ID_CURR,
        COUNT(*)                                              AS BUREAU_LOAN_COUNT,
        SUM((CREDIT_ACTIVE = 'Active')::INT)                  AS BUREAU_ACTIVE_COUNT,
        COALESCE(SUM(AMT_CREDIT_SUM_DEBT), 0)                 AS BUREAU_DEBT_SUM,
        COALESCE(SUM(AMT_CREDIT_SUM), 0)                      AS BUREAU_CREDIT_SUM,
        COALESCE(SUM(AMT_CREDIT_SUM_DEBT), 0)
            / (COALESCE(SUM(AMT_CREDIT_SUM), 0) + 1e-9)       AS BUREAU_DEBT_CREDIT_RATIO,
        MAX(CREDIT_DAY_OVERDUE)                               AS BUREAU_OVERDUE_MAX,
        COALESCE((MAX(CREDIT_DAY_OVERDUE) > 0)::INT, 0)       AS BUREAU_OVERDUE_ANY,
        AVG(DAYS_CREDIT)                                      AS BUREAU_DAYS_CREDIT_MEAN
    FROM read_csv_auto(?)
    GROUP BY SK_ID_CURR
"""

PREV_SQL = """
    SELECT
        SK_ID_CURR,
        COUNT(*)                                              AS PREV_APP_COUNT,
        SUM((NAME_CONTRACT_STATUS = 'Refused')::INT)          AS PREV_REFUSED_COUNT,
        SUM((NAME_CONTRACT_STATUS = 'Refused')::INT) / COUNT(*)::DOUBLE AS PREV_REFUSED_RATE,
        AVG(AMT_APPLICATION)                                  AS PREV_AMT_APPLICATION_MEAN,
        COALESCE(SUM(AMT_CREDIT), 0)
            / (COALESCE(SUM(AMT_APPLICATION), 0) + 1e-9)      AS PREV_CREDIT_APPLICATION_RATIO,
        AVG(DAYS_DECISION)                                    AS PREV_DAYS_DECISION_MEAN
    FROM read_csv_auto(?)
    GROUP BY SK_ID_CURR
"""


def aggregate_bureau_sql(csv_path: Path | str) -> pd.DataFrame:
    """Equivalente a aggregate_bureau(), pero lee y agrupa bureau.csv con DuckDB."""
    return duckdb.execute(BUREAU_SQL, [str(csv_path)]).df()


def aggregate_prev_sql(csv_path: Path | str) -> pd.DataFrame:
    """Equivalente a aggregate_prev(), pero lee y agrupa previous_application.csv con DuckDB."""
    return duckdb.execute(PREV_SQL, [str(csv_path)]).df()


def merge_aggregates(app: pd.DataFrame, secondary: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Left join de agregados sobre la tabla principal. Nulos = sin historial
    (señal en sí misma; el imputador del Pipeline los maneja)."""
    out = app.copy()
    if "bureau" in secondary:
        out = out.merge(aggregate_bureau(secondary["bureau"]), on=config.ID_COL, how="left")
    if "prev" in secondary:
        out = out.merge(aggregate_prev(secondary["prev"]), on=config.ID_COL, how="left")
    # clientes sin historial: counts en 0 tiene semántica clara (no imputar mediana)
    for col in ["BUREAU_LOAN_COUNT", "BUREAU_ACTIVE_COUNT", "PREV_APP_COUNT",
                "PREV_REFUSED_COUNT", "BUREAU_OVERDUE_ANY"]:
        if col in out.columns:
            out[col] = out[col].fillna(0)
    return out


AGG_PREFIXES = ("BUREAU_", "PREV_")


def aggregate_feature_names(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(AGG_PREFIXES)]
