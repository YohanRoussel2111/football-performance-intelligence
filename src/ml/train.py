"""
Model training, temporal validation, comparison and calibration.

Pipeline
--------
1. Build the leakage-safe learning table and the temporal train/valid/test split.
2. Train three candidate models that span the bias/variance spectrum:
       - Logistic Regression (scaled, class-weighted)  -> interpretable baseline
       - Random Forest (class-weighted)                -> non-linear, robust
       - Gradient Boosting / XGBoost (scale_pos_weight) -> usually strongest
3. Report a temporal cross-validation score on the training window and full
   metrics on the validation window (ROC-AUC, PR-AUC, precision/recall/F1,
   Brier, confusion matrix).
4. Select the best model by validation PR-AUC (the right metric for rare events).
5. Calibrate the selected model (Platt scaling) on the validation window and
   evaluate the *calibrated* model on the untouched test window.
6. Persist a self-describing model bundle (model, calibrator, feature list,
   operating threshold, metrics, metadata, drift reference, SHAP background).

Class imbalance is handled at the estimator level (class_weight / scale_pos_weight)
rather than by resampling, which keeps the calibration meaningful.
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.feature_engineering.features import MODEL_FEATURES, build_features
from src.data_processing import pipeline as dp_pipeline
from src.ml import dataset as ds

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:                       # pragma: no cover
    HAS_XGB = False

MODEL_VERSION = "1.0.0"
BUNDLE_PATH = config.MODELS_DIR / "model_bundle.joblib"
METRICS_PATH = config.MODELS_DIR / "metrics.json"


# --------------------------------------------------------------------------- #
# Candidate models
# --------------------------------------------------------------------------- #
def _candidates(pos_weight: float) -> dict:
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=2000, C=0.5)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=7, min_samples_leaf=25,
            class_weight="balanced_subsample", random_state=config.RANDOM_SEED,
            n_jobs=-1),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=0.045,
            subsample=0.85, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=1.5, scale_pos_weight=pos_weight,
            eval_metric="aucpr", random_state=config.RANDOM_SEED, n_jobs=-1)
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.85, random_state=config.RANDOM_SEED)
    return models


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)) if y_true.sum() else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)) if y_true.sum() else float("nan"),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)),
        "confusion_matrix": cm.tolist(),
        "base_rate": float(y_true.mean()),
        "threshold": float(threshold),
    }


def temporal_cv_score(model_factory, train_df, feature_cols, n_folds: int = 4) -> float:
    """Expanding-window temporal CV PR-AUC on the training window.

    Longitudinal data must not be shuffled: each fold trains on an earlier
    contiguous block and validates on the block immediately after it.
    """
    df = train_df.sort_values("season_day")
    days = df["season_day"].to_numpy()
    lo, hi = days.min(), days.max()
    edges = np.linspace(lo, hi, n_folds + 2)
    scores = []
    for k in range(1, n_folds + 1):
        tr = df[df["season_day"] <= edges[k]]
        va = df[(df["season_day"] > edges[k]) & (df["season_day"] <= edges[k + 1])]
        if va["target"].sum() < 3 or tr["target"].sum() < 3:
            continue
        Xtr, ytr = ds.get_xy(tr, feature_cols)
        Xva, yva = ds.get_xy(va, feature_cols)
        m = model_factory()
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        scores.append(average_precision_score(yva, p))
    return float(np.mean(scores)) if scores else float("nan")


def best_f1_threshold(y_true, proba) -> float:
    """Operating threshold that maximizes F1 on the validation window."""
    grid = np.linspace(0.05, 0.9, 60)
    f1s = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


# --------------------------------------------------------------------------- #
# Calibration (version-robust)
# --------------------------------------------------------------------------- #
def _calibrate(prefit_model, X_valid, y_valid):
    from sklearn.calibration import CalibratedClassifierCV
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(prefit_model), method="sigmoid")
        cal.fit(X_valid, y_valid)
    except Exception:
        cal = CalibratedClassifierCV(prefit_model, method="sigmoid", cv="prefit")
        cal.fit(X_valid, y_valid)
    return cal


def calibration_curve_points(y_true, proba, n_bins: int = 10):
    """Reliability curve points (mean predicted vs observed frequency)."""
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(
        y_true, proba, n_bins=n_bins, strategy="quantile")
    return {"mean_predicted": mean_pred.tolist(),
            "observed_frequency": frac_pos.tolist()}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train_and_select(seed: int = config.RANDOM_SEED, verbose: bool = True) -> dict:
    out = dp_pipeline.run_pipeline(seed=seed, persist=False)
    feats = build_features(out["daily"])
    lt = ds.build_learning_table(feats)
    sp = ds.temporal_split(lt)
    feature_cols = MODEL_FEATURES

    Xtr, ytr = ds.get_xy(sp["train"], feature_cols)
    Xva, yva = ds.get_xy(sp["valid"], feature_cols)
    Xte, yte = ds.get_xy(sp["test"], feature_cols)

    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    comparison = {}
    fitted = {}
    for name, model in _candidates(pos_weight).items():
        model.fit(Xtr, ytr)
        proba_va = model.predict_proba(Xva)[:, 1]
        # Model comparison uses ranking metrics (PR-AUC / ROC-AUC), which are
        # threshold-independent; F1 here is reported at a provisional operating
        # point for context only.
        thr = best_f1_threshold(yva, proba_va)
        metrics_va = evaluate(yva, proba_va, thr)
        cv = temporal_cv_score(
            lambda n=name: _candidates(pos_weight)[n], sp["train"], feature_cols)
        metrics_va["temporal_cv_pr_auc"] = cv
        comparison[name] = metrics_va
        fitted[name] = model
        if verbose:
            print(f"{name:20s} valid PR-AUC={metrics_va['pr_auc']:.3f} "
                  f"ROC-AUC={metrics_va['roc_auc']:.3f} "
                  f"F1={metrics_va['f1']:.3f} CV={cv:.3f}")

    # Select by the temporal cross-validation PR-AUC (mean over expanding-window
    # folds). This is a more robust estimate of generalization than a single
    # validation window, which can be dominated by one congested period.
    best_name = max(comparison, key=lambda k: comparison[k]["temporal_cv_pr_auc"])
    best_model = fitted[best_name]

    # Calibrate on validation, evaluate calibrated model on the untouched test set.
    calibrated = _calibrate(best_model, Xva, yva)
    # IMPORTANT: choose the operating threshold on the *calibrated* validation
    # probabilities, then apply that same threshold to the calibrated test
    # probabilities. Selecting the threshold on uncalibrated scores and applying
    # it to calibrated scores would mix two different probability scales.
    proba_va_cal = calibrated.predict_proba(Xva)[:, 1]
    threshold = best_f1_threshold(yva, proba_va_cal)

    proba_te = calibrated.predict_proba(Xte)[:, 1]
    test_metrics = evaluate(yte, proba_te, threshold)
    test_metrics["calibration_curve"] = calibration_curve_points(yte, proba_te, n_bins=8)

    # Uncalibrated test metrics for comparison (shows calibration effect on Brier).
    proba_te_raw = best_model.predict_proba(Xte)[:, 1]
    uncal_brier = float(brier_score_loss(yte, proba_te_raw))

    # Monitoring-level bands (LOW/MODERATE/HIGH) from calibrated probability.
    base_rate = float(ytr.mean())
    level_bands = {
        "moderate": round(max(base_rate, 0.10), 3),
        "high": round(max(threshold, 2 * base_rate), 3),
    }

    metadata = {
        "model_version": MODEL_VERSION,
        "selected_model": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": int(len(ytr)), "n_valid": int(len(yva)), "n_test": int(len(yte)),
        "n_features": len(feature_cols),
        "prediction_horizon_days": config.PREDICTION_HORIZON_DAYS,
        "train_base_rate": base_rate,
        "positive_weight": pos_weight,
        "python": platform.python_version(),
        "seed": seed,
        "target_definition": (
            "Player currently available; reduced-availability event "
            f"(modified/unavailable) within next {config.PREDICTION_HORIZON_DAYS} days."
        ),
        "split": {"train_end_day": float(sp["cut_train"]),
                  "valid_end_day": float(sp["cut_valid"])},
    }

    # Drift reference: training feature means/std for the monitoring page.
    drift_reference = {
        "means": Xtr.mean().to_dict(),
        "stds": Xtr.std().replace(0, 1e-9).to_dict(),
        "pred_distribution": np.histogram(
            best_model.predict_proba(Xtr)[:, 1], bins=20, range=(0, 1))[0].tolist(),
    }

    bundle = {
        "base_model": best_model,
        "calibrated_model": calibrated,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "level_bands": level_bands,
        "metadata": metadata,
        "comparison": comparison,
        "test_metrics": test_metrics,
        "uncalibrated_test_brier": uncal_brier,
        "drift_reference": drift_reference,
        # small background sample for SHAP (train rows)
        "shap_background": Xtr.sample(min(200, len(Xtr)), random_state=seed),
    }

    if verbose:
        print(f"\nSelected: {best_name}")
        print(f"Test (calibrated): PR-AUC={test_metrics['pr_auc']:.3f} "
              f"ROC-AUC={test_metrics['roc_auc']:.3f} "
              f"recall={test_metrics['recall']:.3f} "
              f"precision={test_metrics['precision']:.3f}")
        print(f"Brier: calibrated={test_metrics['brier']:.4f} "
              f"uncalibrated={uncal_brier:.4f}")
    return bundle


def save_bundle(bundle: dict) -> None:
    joblib.dump(bundle, BUNDLE_PATH)
    # human-readable metrics summary
    summary = {
        "metadata": bundle["metadata"],
        "comparison": {k: {kk: vv for kk, vv in v.items()
                           if kk not in ("confusion_matrix",)}
                       for k, v in bundle["comparison"].items()},
        "test_metrics": {k: v for k, v in bundle["test_metrics"].items()
                         if k != "calibration_curve"},
        "uncalibrated_test_brier": bundle["uncalibrated_test_brier"],
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2))


def load_bundle() -> dict:
    return joblib.load(BUNDLE_PATH)


if __name__ == "__main__":
    b = train_and_select()
    save_bundle(b)
    print(f"\nSaved bundle -> {BUNDLE_PATH}")
