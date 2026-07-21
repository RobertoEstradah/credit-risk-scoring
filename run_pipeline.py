"""Pipeline completo end-to-end:

    datos (app + bureau + prev) → agregaciones multi-tabla → features →
    baseline vs LightGBM (CV) → holdout → KS + umbral óptimo por costos →
    SHAP → persistencia de modelo (models/) → reports/

Uso:
    python run_pipeline.py            # rápido (sin SHAP)
    python run_pipeline.py --shap     # incluye explicabilidad
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src import config
from src.aggregates import merge_aggregates
from src.data import load_data, load_secondary
from src.evaluate import core_metrics, cost_curve, optimal_threshold
from src.features import add_domain_features, build_xy, feature_lists
from src.train import HAS_LGBM, cv_auc, log_run, make_baseline, make_lgbm, split


def main(with_shap: bool = False) -> dict:
    t0 = time.time()
    config.REPORTS_DIR.mkdir(exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------ 1. datos multi-tabla
    app, source = load_data()
    secondary = load_secondary(app, source)
    df = merge_aggregates(app, secondary)
    print(
        f"[data] fuente={source} filas={len(df):,} "
        f"default_rate={df[config.TARGET].mean():.3f} "
        f"tablas_secundarias={list(secondary)}"
    )

    # -------------------------------------------------------- 2. features
    X, y = build_xy(df)
    columns = feature_lists(add_domain_features(df))
    X_train, X_test, y_train, y_test = split(X, y)
    print(f"[features] {X.shape[1]} columnas | train={len(X_train):,} test={len(X_test):,}")

    results = {"data_source": source, "n_rows": len(df), "n_features": X.shape[1]}

    # --------------------------------------------------- 3. baseline (CV)
    base = make_baseline(columns)
    auc_m, auc_s = cv_auc(base, X_train, y_train)
    print(f"[baseline LogReg] CV AUC = {auc_m:.4f} ± {auc_s:.4f}")
    log_run("logreg_baseline", {"model": "logreg", "C": 0.1, "source": source}, {"cv_auc": auc_m})
    results["baseline_cv_auc"] = auc_m

    # --------------------------------------------------- 4. LightGBM (CV)
    model, model_name = base, "logreg"
    if HAS_LGBM:
        lgbm = make_lgbm(columns)
        auc_m2, auc_s2 = cv_auc(lgbm, X_train, y_train)
        print(f"[LightGBM]        CV AUC = {auc_m2:.4f} ± {auc_s2:.4f}")
        log_run("lightgbm", {"model": "lgbm", "n_estimators": 400, "source": source},
                {"cv_auc": auc_m2})
        results["lgbm_cv_auc"] = auc_m2
        if auc_m2 >= auc_m:
            model, model_name = lgbm, "lgbm"

    # -------------------------------------------------- 5. holdout final
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = core_metrics(y_test.to_numpy(), y_prob)
    print(f"[holdout {model_name}] AUC={metrics['auc']:.4f} KS={metrics['ks']:.4f}")
    results["selected_model"] = model_name
    results["holdout"] = metrics

    # ---------------------------------------- 6. umbral óptimo por costos
    best = optimal_threshold(y_test.to_numpy(), y_prob)
    print(
        f"[decisión] umbral*={best['threshold']:.2f} "
        f"aprobación={best['approval_rate']:.1%} costo={best['cost']:.1f} "
        f"(FN={best['fn']}, FP={best['fp']})"
    )
    results["optimal_threshold"] = best
    curve = cost_curve(y_test.to_numpy(), y_prob)
    curve.to_csv(config.REPORTS_DIR / "cost_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=110)
    ax.plot(curve["threshold"], curve["cost"], color="#2563EB", lw=1.5)
    ax.scatter([best["threshold"]], [best["cost"]], color="#DC2626", zorder=3,
               label=f"óptimo: umbral={best['threshold']:.2f}")
    ax.set_xlabel("umbral de rechazo")
    ax.set_ylabel("costo esperado")
    ax.set_title(f"Costo esperado por umbral (fuente: {source})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "cost_curve.png")
    plt.close(fig)

    # ------------------------------------------------------------ 7. SHAP
    if with_shap:
        from src.explain import explain_case, global_importance, shap_values

        sample = X_test.sample(min(2000, len(X_test)), random_state=config.RANDOM_STATE)
        sv, _ = shap_values(model, sample)
        gi = global_importance(sv)
        gi.to_csv(config.REPORTS_DIR / "shap_global_importance.csv")
        print("[shap] top 5 features globales:")
        print(gi.head().to_string())

        probs_sample = model.predict_proba(sample)[:, 1]
        worst = sample.index[probs_sample.argmax()]
        case = explain_case(sv, worst)
        case.to_csv(config.REPORTS_DIR / "shap_case_example.csv")
        print(f"[shap] caso {worst} (p_default={probs_sample.max():.2f}):")
        print(case.to_string())

    # -------------------------------------- 8. persistencia para servir
    artifact = {
        "model": model,
        "model_name": model_name,
        "threshold": best["threshold"],
        "feature_columns": list(X.columns),
        "data_source": source,
        "holdout_metrics": metrics,
    }
    joblib.dump(artifact, config.MODELS_DIR / "model.joblib")
    print(f"[persist] models/model.joblib ({model_name}, umbral={best['threshold']:.2f})")

    results["elapsed_s"] = round(time.time() - t0, 1)
    (config.REPORTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n✓ Pipeline completo en {results['elapsed_s']}s → reports/results.json")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap", action="store_true", help="incluir explicabilidad SHAP")
    main(with_shap=ap.parse_args().shap)
