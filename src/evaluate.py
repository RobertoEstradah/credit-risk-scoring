"""Métricas de scoring y análisis de decisión basado en costos.

AUC-ROC y KS son las métricas estándar de riesgo crediticio. La parte de
negocio: elegir el umbral de aprobación que minimiza el costo esperado dada
una matriz de costos FN/FP (config.COST_FN, config.COST_FP).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from . import config


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov–Smirnov: separación máxima entre TPR y FPR."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def core_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ks": ks_statistic(y_true, y_prob),
        "default_rate": float(np.mean(y_true)),
    }


def cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = config.COST_FN,
    cost_fp: float = config.COST_FP,
    n_thresholds: int = 101,
) -> pd.DataFrame:
    """Costo total esperado por umbral. Predicción 1 = rechazar solicitud."""
    thresholds = np.linspace(0, 1, n_thresholds)
    rows = []
    y_true = np.asarray(y_true)
    for t in thresholds:
        reject = y_prob >= t
        fn = np.sum((~reject) & (y_true == 1))  # aprobado que hace default
        fp = np.sum(reject & (y_true == 0))     # buen cliente rechazado
        rows.append(
            {
                "threshold": t,
                "fn": int(fn),
                "fp": int(fp),
                "approval_rate": float(np.mean(~reject)),
                "cost": float(fn * cost_fn + fp * cost_fp),
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, **kw) -> dict:
    curve = cost_curve(y_true, y_prob, **kw)
    best = curve.loc[curve["cost"].idxmin()]
    return {
        "threshold": float(best["threshold"]),
        "cost": float(best["cost"]),
        "approval_rate": float(best["approval_rate"]),
        "fn": int(best["fn"]),
        "fp": int(best["fp"]),
    }
