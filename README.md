# Credit Risk Scoring with Explainability

End-to-end credit default prediction system: multi-table data validation →
domain + historical feature engineering → model comparison (Logistic Regression
vs LightGBM) → KS/AUC evaluation → **cost-based decision threshold
optimization** → SHAP explainability → **FastAPI scoring service** (Docker,
CI). Leakage-safe sklearn Pipelines, 14-test suite, train/serve parity.

> **Data source disclosure:** the pipeline runs on the real Home Credit
> dataset when present (`data/download.sh`) and falls back to a synthetic
> generator with the same schema otherwise. Every artifact — `results.json`,
> the EDA notebook, the API's `/health` — declares which source was used.
> Metrics below are from the **real Kaggle dataset** (307,511 rows).

## Problem

A lender must decide which applications to approve. The two errors have
asymmetric costs:

| Error | Business meaning | Relative cost (configurable) |
|---|---|---|
| False Negative | approve an applicant who defaults | 1.00 |
| False Positive | reject a good customer | 0.15 |

Instead of only reporting AUC, the pipeline selects the **approval threshold
that minimizes expected cost** — the decision a risk team actually makes.

![cost curve](reports/cost_curve.png)

## Results (real Kaggle run, 307,511 rows, default rate 8.07%)

| Model | CV AUC (5-fold) | Holdout AUC | KS |
|---|---|---|---|
| Logistic Regression (baseline) | 0.7483 ± 0.0022 | — | — |
| **LightGBM (selected)** | **0.7673 ± 0.0011** | **0.7739** | **0.4112** |

LightGBM beats the linear baseline on CV — as expected on the real dataset,
which has stronger non-linear interactions than the synthetic fallback (where
the baseline wins, see `reports/` from a synthetic run). Model selection is
purely CV-driven; nothing is hardcoded.

Optimal decision: threshold **0.62** → approval rate **82.6%**, minimizing
expected cost given the FN/FP cost matrix (FN=2,437, FP=8,185, cost=3,664.75).

Top SHAP drivers (global): `EXT_SOURCES_MEAN`, `CREDIT_TERM`,
`PREV_CREDIT_APPLICATION_RATIO`, `GOODS_CREDIT_RATIO`, `EXT_SOURCE_3` —
consistent with credit-risk domain knowledge (bureau scores and leverage
ratios dominate). Single-case explanations ("why was applicant X rejected")
in `reports/shap_case_example.csv`.

**Real-data quirk handled:** `DAYS_EMPLOYED == 365243` is Home Credit's null
sentinel (~1000 years employed, mostly pensioners/unemployed; 18% of rows).
It's converted to `NaN` in `src/data.py` before feature engineering so it
flows through the same median imputer as any other missing value, instead of
silently collapsing to `EMPLOYED_YEARS = 0`. Covered by
`tests/test_pipeline.py::test_days_employed_sentinel_becomes_nan`.

## Architecture

```
application_train + bureau + previous_application   (real or synthetic)
        │  src/data.py        — schema validation, source disclosure
        ▼
multi-table aggregation      — per-client bureau/previous-app statistics
        │  src/aggregates.py  — BUREAU_*/PREV_* features, left join
        ▼
domain features              — credit/income ratios, tenure, EXT aggregates
        │  src/features.py    — pure row-wise functions (no leakage)
        ▼
sklearn Pipeline             — imputation/encoding fitted per-fold only
        │  src/train.py       — LogReg baseline vs LightGBM, stratified CV
        ▼
evaluation & decision        — AUC, KS, cost curve, optimal threshold
        │  src/evaluate.py
        ▼
explainability               — SHAP (Tree/Linear explainer auto-selected)
        │  src/explain.py
        ▼
model artifact               — models/model.joblib (model + threshold + schema)
        │
        ▼
FastAPI service              — POST /score → PD + approve/reject decision
        │  app/main.py        — train/serve parity via shared feature code
        ▼
Docker + GitHub Actions CI   — tests → train → API tests on every push
```

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 14 tests, <5 s, no external data needed
python run_pipeline.py --shap       # full run (synthetic fallback if no CSV)
uvicorn app.main:app --reload       # scoring service on :8000

# real data (requires Kaggle credentials + accepting competition rules):
bash data/download.sh && python run_pipeline.py --shap

# container:
docker build -t credit-scoring . && docker run -p 8000:8000 credit-scoring
```

Example request:

```bash
curl -X POST localhost:8000/score -H "Content-Type: application/json" -d '{
  "AMT_INCOME_TOTAL": 250000, "AMT_CREDIT": 400000, "AMT_ANNUITY": 25000,
  "DAYS_BIRTH": -14600, "DAYS_EMPLOYED": -4380,
  "EXT_SOURCE_1": 0.85, "EXT_SOURCE_2": 0.8, "EXT_SOURCE_3": 0.82
}'
# → {"probability_of_default": ..., "decision": "approve", "threshold": 0.62,
#    "model": "lgbm", "trained_on": "kaggle"}
```

## Design decisions

- **Leakage control:** every stateful transform (imputer, scaler, one-hot)
  lives inside the sklearn `Pipeline`, fit only on training folds. A test
  asserts engineered features are strictly row-wise. Multi-table aggregates
  use only each client's own history.
- **Honest model selection:** LightGBM must beat the logistic baseline on CV
  to be deployed; otherwise the simpler model ships.
- **Business metric first:** the deliverable is a threshold + approval rate +
  expected cost, not a leaderboard score.
- **Train/serve parity:** the API imports the same `add_domain_features` used
  in training and reindexes to the persisted feature schema; missing fields
  flow through the same imputer as in training.
- **Source honesty:** synthetic vs real data is tracked end-to-end and exposed
  in every artifact, including the API.
- **Graceful degradation:** MLflow and SHAP are optional imports; the pipeline
  runs without them.
- **SQL as an alternative, tested equivalent path:** `src/aggregates.py` also
  ships `aggregate_bureau_sql`/`aggregate_prev_sql`, which run the same
  groupby logic as DuckDB SQL directly over the CSVs on disk instead of
  loading the full table into pandas. The pipeline still runs the pandas
  version; the SQL version is proven equivalent on the full real dataset
  (305,811 / 338,857 clients, every column matched) in
  `tests/test_aggregates_sql.py`. The interesting bug this caught: pandas
  `.sum()` on an all-NaN group returns `0.0`, while SQL `SUM()` returns
  `NULL` — the SQL version uses `COALESCE(SUM(...), 0)` to match pandas
  exactly rather than silently diverging on edge-case clients.

## Repository layout

```
app/main.py            FastAPI scoring service
src/                   config, data, aggregates, features, train, evaluate, explain
notebooks/01_eda.ipynb executed EDA with findings (declares data source)
tests/                 14 tests: schema, leakage, metrics, e2e, API contract
reports/               results.json, figures, cost curve, SHAP outputs
data/download.sh       Kaggle download script
Dockerfile · .github/workflows/ci.yml
```

## Tech stack

Python · pandas · DuckDB · scikit-learn · LightGBM · SHAP · MLflow · FastAPI · Docker · GitHub Actions · pytest

## Roadmap

- [x] Run on real Home Credit data; update all metrics and EDA findings
- [x] Handle `DAYS_EMPLOYED == 365243` null sentinel
- [ ] Handle high-cardinality categoricals (e.g. `ORGANIZATION_TYPE`, not yet in the feature set)
- [ ] Probability calibration (reliability curves) for PD estimates
- [ ] Deploy container to a public endpoint (Railway/Render) + demo link
