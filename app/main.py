"""Servicio de scoring: FastAPI sobre el artefacto entrenado (models/model.joblib).

Endpoints:
    GET  /health  — liveness + metadata del modelo cargado
    POST /score   — recibe features de un solicitante, devuelve PD, decisión
                    (según umbral óptimo por costos) y metadata

Diseño:
    - El artefacto incluye modelo + umbral + columnas esperadas + fuente de
      datos, de modo que la respuesta declara si fue entrenado con datos
      sintéticos o reales (restricción de honestidad).
    - Campos faltantes llegan como null → el imputador del Pipeline los maneja,
      igual que en entrenamiento (paridad train/serve).

Correr local:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"

app = FastAPI(title="Credit Risk Scoring API", version="0.1.0")
_artifact: dict | None = None


def get_artifact() -> dict:
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Modelo no encontrado; corre `python run_pipeline.py` primero",
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


class Applicant(BaseModel):
    """Solicitud de crédito. Campos opcionales → null se imputa como en training."""

    AMT_INCOME_TOTAL: float = Field(..., gt=0)
    AMT_CREDIT: float = Field(..., gt=0)
    AMT_ANNUITY: float | None = None
    AMT_GOODS_PRICE: float | None = None
    DAYS_BIRTH: int = Field(..., lt=0, description="días negativos desde nacimiento")
    DAYS_EMPLOYED: int | None = None
    EXT_SOURCE_1: float | None = Field(None, ge=0, le=1)
    EXT_SOURCE_2: float | None = Field(None, ge=0, le=1)
    EXT_SOURCE_3: float | None = Field(None, ge=0, le=1)
    CNT_CHILDREN: int = 0
    REGION_POPULATION_RELATIVE: float | None = None
    NAME_CONTRACT_TYPE: str = "Cash loans"
    CODE_GENDER: str = "F"
    FLAG_OWN_CAR: str = "N"
    FLAG_OWN_REALTY: str = "Y"
    NAME_INCOME_TYPE: str = "Working"
    NAME_EDUCATION_TYPE: str = "Secondary"
    NAME_FAMILY_STATUS: str = "Married"
    NAME_HOUSING_TYPE: str = "House / apartment"
    ORGANIZATION_TYPE: str = "Business Entity Type 3"
    # agregados de historial (opcionales; null = sin historial conocido)
    BUREAU_LOAN_COUNT: float | None = 0
    BUREAU_ACTIVE_COUNT: float | None = 0
    BUREAU_DEBT_SUM: float | None = None
    BUREAU_CREDIT_SUM: float | None = None
    BUREAU_DEBT_CREDIT_RATIO: float | None = None
    BUREAU_OVERDUE_MAX: float | None = None
    BUREAU_OVERDUE_ANY: float | None = 0
    BUREAU_DAYS_CREDIT_MEAN: float | None = None
    PREV_APP_COUNT: float | None = 0
    PREV_REFUSED_COUNT: float | None = 0
    PREV_REFUSED_RATE: float | None = None
    PREV_AMT_APPLICATION_MEAN: float | None = None
    PREV_CREDIT_APPLICATION_RATIO: float | None = None
    PREV_DAYS_DECISION_MEAN: float | None = None


class ScoreResponse(BaseModel):
    probability_of_default: float
    decision: str
    threshold: float
    model: str
    trained_on: str


@app.get("/health")
def health():
    art = get_artifact()
    return {
        "status": "ok",
        "model": art["model_name"],
        "trained_on": art["data_source"],
        "holdout_auc": round(art["holdout_metrics"]["auc"], 4),
    }


@app.post("/score", response_model=ScoreResponse)
def score(applicant: Applicant):
    art = get_artifact()
    row = pd.DataFrame([applicant.model_dump()])

    # features de dominio, idénticas a entrenamiento (paridad train/serve)
    from src.features import add_domain_features

    row = add_domain_features(row)
    X = row.reindex(columns=art["feature_columns"])

    pd_hat = float(art["model"].predict_proba(X)[0, 1])
    decision = "reject" if pd_hat >= art["threshold"] else "approve"
    return ScoreResponse(
        probability_of_default=round(pd_hat, 4),
        decision=decision,
        threshold=art["threshold"],
        model=art["model_name"],
        trained_on=art["data_source"],
    )
