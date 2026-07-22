"""Entrenamiento: baseline (regresión logística) vs LightGBM.

Todo preprocesamiento con estado (imputación, escalado, one-hot) vive dentro
del Pipeline de sklearn → se ajusta solo con datos de entrenamiento en cada
fold (cero leakage). MLflow es opcional: si está instalado, cada corrida se
registra; si no, el pipeline sigue funcionando.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .features import feature_lists

try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    HAS_MLFLOW = False

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:  # pragma: no cover
    HAS_LGBM = False


def _preprocessor(for_linear: bool, columns=None) -> ColumnTransformer:
    numeric, categorical = columns if columns else feature_lists()
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if for_linear:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.01),
                categorical,
            ),
        ]
    )


def make_baseline(columns=None) -> Pipeline:
    # Sin class_weight: el desbalance ~92/8 ya se maneja con el umbral de
    # decisión por costos (src/evaluate.py::optimal_threshold), no con la
    # pérdida de entrenamiento. Rebalancear aquí distorsiona predict_proba
    # (ver evaluate.py::calibration_metrics).
    return Pipeline(
        [
            ("prep", _preprocessor(for_linear=True, columns=columns)),
            (
                "clf",
                LogisticRegression(max_iter=2000, C=0.1),
            ),
        ]
    )


def make_lgbm(columns=None) -> Pipeline:
    if not HAS_LGBM:
        raise ImportError("lightgbm no instalado; usa make_baseline()")
    return Pipeline(
        [
            ("prep", _preprocessor(for_linear=False, columns=columns)),
            (
                "clf",
                LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=50,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=config.RANDOM_STATE,
                    verbose=-1,
                ),
            ),
        ]
    )


def split(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )


def cv_auc(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    cv = StratifiedKFold(
        n_splits=config.N_SPLITS_CV, shuffle=True, random_state=config.RANDOM_STATE
    )
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def log_run(name: str, params: dict, metrics: dict) -> None:
    """Registra en MLflow si está disponible; si no, imprime."""
    if HAS_MLFLOW:
        mlflow.set_experiment("credit-risk-scoring")
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
    else:
        print(f"[{name}] params={params} metrics={metrics}")
