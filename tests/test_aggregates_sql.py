"""Equivalencia pandas vs DuckDB para las agregaciones multi-tabla (FASE B).

Los datos de prueba incluyen a propósito grupos con columnas enteramente NaN
(p.ej. AMT_CREDIT_SUM_DEBT nulo en todas las filas de un cliente), porque ahí
es donde pandas y SQL estándar difieren por default: pandas `.sum()` de un
grupo vacío/todo-NaN da 0.0, SQL `SUM()` da NULL. La versión SQL usa
COALESCE(SUM(...), 0) para igualar el comportamiento de pandas.
"""
import numpy as np
import pandas as pd

from src.aggregates import (
    aggregate_bureau,
    aggregate_bureau_sql,
    aggregate_prev,
    aggregate_prev_sql,
)


def _bureau_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 2, 2, 3, 4, 4],
            "DAYS_CREDIT": [-100, -500, -30, -200, -10, -1000, -50, -75],
            "AMT_CREDIT_SUM": [10000.0, 5000.0, 2000.0, 8000.0, np.nan, 3000.0, np.nan, np.nan],
            "AMT_CREDIT_SUM_DEBT": [1000.0, np.nan, 500.0, np.nan, np.nan, 0.0, np.nan, np.nan],
            "CREDIT_DAY_OVERDUE": [0, 0, 5, 0, 0, np.nan, np.nan, np.nan],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Closed", "Active", "Closed", "Active", "Closed"],
        }
    )


def _prev_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 2, 2, 3, 4],
            "AMT_APPLICATION": [5000.0, np.nan, 3000.0, 2000.0, np.nan, np.nan, 1000.0],
            "AMT_CREDIT": [4500.0, 1000.0, np.nan, 1800.0, np.nan, np.nan, 900.0],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Refused", "Approved", "Refused", "Canceled", "Approved"],
            "DAYS_DECISION": [-100, -50, -300, -10, -5, -900, -20],
        }
    )


def _assert_equivalent(pandas_out: pd.DataFrame, sql_out: pd.DataFrame) -> None:
    pandas_out = pandas_out.sort_values("SK_ID_CURR").reset_index(drop=True)
    sql_out = sql_out.sort_values("SK_ID_CURR").reset_index(drop=True)
    assert list(pandas_out["SK_ID_CURR"]) == list(sql_out["SK_ID_CURR"])
    for col in pandas_out.columns:
        if col == "SK_ID_CURR":
            continue
        np.testing.assert_allclose(
            pandas_out[col].to_numpy(dtype=float),
            sql_out[col].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"columna {col} difiere entre pandas y SQL",
        )


def test_bureau_pandas_vs_sql_equivalent(tmp_path):
    df = _bureau_df()
    csv_path = tmp_path / "bureau.csv"
    df.to_csv(csv_path, index=False)

    _assert_equivalent(aggregate_bureau(df), aggregate_bureau_sql(csv_path))


def test_prev_pandas_vs_sql_equivalent(tmp_path):
    df = _prev_df()
    csv_path = tmp_path / "previous_application.csv"
    df.to_csv(csv_path, index=False)

    _assert_equivalent(aggregate_prev(df), aggregate_prev_sql(csv_path))
