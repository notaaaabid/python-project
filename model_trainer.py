import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, accuracy_score,
                              confusion_matrix, roc_auc_score)
from sklearn.model_selection import cross_val_score
import joblib
import os

MODEL_PATH = "/tmp/sleep_best_model.pkl"

# ── Candidate models ──────────────────────────────────────────────────────────
def _build_candidates() -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1,
            max_depth=4, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=42, C=1.0
        ),
        "SVM": SVC(
            kernel="rbf", probability=True,
            class_weight="balanced", random_state=42
        ),
    }


def train_and_evaluate(X_train: np.ndarray,
                        X_test:  np.ndarray,
                        y_train: np.ndarray,
                        y_test:  np.ndarray,
                        feature_names: list,
                        label_encoder) -> dict:
    """
    Train all candidate models, pick the best by test accuracy,
    persist it, and return a rich results dict for the GUI dashboard.

    Returns
    -------
    {
      "best_model_name": str,
      "best_model": fitted estimator,
      "accuracy": float,
      "report": str,
      "confusion_matrix": ndarray,
      "feature_importance": pd.Series or None,
      "all_scores": {name: cv_mean_acc},
      "classes": list[str]
    }
    """
    candidates    = _build_candidates()
    best_model    = None
    best_name     = ""
    best_accuracy = 0.0
    all_scores    = {}

    print("\n[model_trainer] ── Training candidates ──────────────────────────")
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
        test_acc  = accuracy_score(y_test, clf.predict(X_test))
        all_scores[name] = round(cv_scores.mean(), 4)
        print(f"  {name:<22}  CV={cv_scores.mean():.4f}  Test={test_acc:.4f}")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model    = clf
            best_name     = name

    # ── Full evaluation of best model ─────────────────────────────────────────
    y_pred    = best_model.predict(X_test)
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_
    )

    print(f"\n[model_trainer] Best model: {best_name}  (Test acc={best_accuracy:.4f})")
    print(report)

    # ── Feature importance (tree models only) ─────────────────────────────────
    feature_importance = None
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=feature_names)
        feature_importance = fi.sort_values(ascending=False)

    # ── Persist ───────────────────────────────────────────────────────────────
    joblib.dump(best_model, MODEL_PATH)
    print(f"[model_trainer] Model saved → {MODEL_PATH}")

    return {
        "best_model_name":  best_name,
        "best_model":       best_model,
        "accuracy":         round(best_accuracy, 4),
        "report":           report,
        "confusion_matrix": cm,
        "feature_importance": feature_importance,
        "all_scores":       all_scores,
        "classes":          list(label_encoder.classes_),
    }


def load_model():
    """Load a previously saved model from disk."""
    if os.path.isfile(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    raise FileNotFoundError(
        f"No trained model found at {MODEL_PATH}. "
        "Run train_and_evaluate() first."
    )


def predict_single(model, X_single: np.ndarray,
                   label_encoder, threshold: float = 0.40) -> dict:
    """
    Run inference on a single scaled feature vector.

    Returns
    -------
    {
      "label": str,          # predicted class name
      "probabilities": dict, # {class_name: probability}
      "risk_level": str,     # "Low" / "Moderate" / "High"
    }
    """
    proba   = model.predict_proba(X_single)[0]
    classes = label_encoder.classes_
    prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

    max_prob      = float(np.max(proba))
    predicted_idx = int(np.argmax(proba))

    # If the model isn't confident enough, flag as uncertain
    if max_prob < threshold:
        predicted_label = "Uncertain"
    else:
        predicted_label = classes[predicted_idx]

    # Sleep Apnea probability drives the risk level
    apnea_prob = prob_dict.get("Sleep Apnea", 0.0)
    insomnia_prob = prob_dict.get("Insomnia", 0.0)
    disorder_prob = apnea_prob + insomnia_prob

    if disorder_prob >= 0.60:
        risk = "High"
    elif disorder_prob >= 0.30:
        risk = "Moderate"
    else:
        risk = "Low"

    return {
        "label":         predicted_label,
        "probabilities": prob_dict,
        "risk_level":    risk,
    }


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader  import load_dataset
    from preprocessor import preprocess

    df = load_dataset()
    X_tr, X_te, y_tr, y_te, sc, les, feats = preprocess(df)
    results = train_and_evaluate(X_tr, X_te, y_tr, y_te, feats, les["__target__"])
    print("\nAll CV scores:")
    for k, v in results["all_scores"].items():
        print(f"  {k}: {v}")
    if results["feature_importance"] is not None:
        print("\nTop 5 features:")
        print(results["feature_importance"].head(5))