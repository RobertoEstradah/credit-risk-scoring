"""Explicabilidad con SHAP sobre el modelo LightGBM entrenado.

Dos niveles:
  - global: importancia media |SHAP| por feature (qué mueve al modelo)
  - local:  desglose de una predicción individual (por qué se rechazó a X),
            el formato que un comité de crédito / regulador puede leer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    import shap

    HAS_SHAP = True
except ImportError:  # pragma: no cover
    HAS_SHAP = False


def _transformed(model: Pipeline, X: pd.DataFrame):
    prep = model.named_steps["prep"]
    Xt = prep.transform(X)
    names = prep.get_feature_names_out()
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    return pd.DataFrame(Xt, columns=names, index=X.index)


def shap_values(model: Pipeline, X: pd.DataFrame):
    """(shap_values_df, X_transformed) para el clasificador del pipeline.

    Usa TreeExplainer para modelos de árboles y LinearExplainer para lineales,
    de modo que la explicabilidad funcione con cualquier modelo seleccionado.
    """
    if not HAS_SHAP:
        raise ImportError("shap no instalado")
    Xt = _transformed(model, X)
    clf = model.named_steps["clf"]
    if hasattr(clf, "booster_") or type(clf).__name__.startswith(("LGBM", "XGB")):
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
        if isinstance(sv, list):  # algunas versiones devuelven [clase0, clase1]
            sv = sv[1]
    else:  # modelos lineales (p. ej. LogisticRegression)
        explainer = shap.LinearExplainer(clf, Xt)
        sv = explainer.shap_values(Xt)
    return pd.DataFrame(sv, columns=Xt.columns, index=Xt.index), Xt


def global_importance(sv: pd.DataFrame, top: int = 15) -> pd.Series:
    return sv.abs().mean().sort_values(ascending=False).head(top)


def explain_case(sv: pd.DataFrame, idx, top: int = 8) -> pd.DataFrame:
    """Top contribuciones (positivas = empujan a rechazo) de un caso."""
    row = sv.loc[idx]
    out = row.reindex(row.abs().sort_values(ascending=False).index).head(top)
    return out.to_frame("shap_value").assign(
        direction=lambda d: np.where(d["shap_value"] > 0, "↑ riesgo", "↓ riesgo")
    )
