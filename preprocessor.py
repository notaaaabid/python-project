import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# ── Paths for persisting fitted encoders/scalers ─────────────────────────────
SCALER_PATH  = "/tmp/sleep_scaler.pkl"
ENCODER_PATH = "/tmp/sleep_label_encoders.pkl"

# ── Column groups ─────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["Gender", "Occupation", "BMI Category"]
TARGET_COL       = "Sleep Disorder"
DROP_COLS        = ["Person ID"]          # Not a predictive feature


def _parse_blood_pressure(bp_series: pd.Series) -> pd.DataFrame:
    """
    Split '120/80' into two numeric columns: Systolic_BP, Diastolic_BP.
    Non-parseable values are filled with the column median.
    """
    split = bp_series.str.split("/", expand=True).apply(pd.to_numeric, errors="coerce")
    split.columns = ["Systolic_BP", "Diastolic_BP"]
    split = split.fillna(split.median())
    return split


def preprocess(df: pd.DataFrame,
               fit: bool = True,
               scaler: StandardScaler | None = None,
               label_encoders: dict | None = None
               ) -> tuple[np.ndarray, np.ndarray,
                          np.ndarray, np.ndarray,
                          StandardScaler, dict, list]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df             : raw DataFrame from data_loader
    fit            : True → fit new encoders/scaler (training time)
                     False → use provided encoders/scaler (inference time)
    scaler         : pre-fitted StandardScaler (required when fit=False)
    label_encoders : dict of pre-fitted LabelEncoders (required when fit=False)

    Returns
    -------
    X_train, X_test, y_train, y_test,
    fitted_scaler, fitted_label_encoders, feature_names
    """
    df = df.copy()

    # ── 1. Drop non-features ──────────────────────────────────────────────────
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # ── 2. Parse blood pressure ───────────────────────────────────────────────
    if "Blood Pressure" in df.columns:
        bp_df = _parse_blood_pressure(df["Blood Pressure"])
        df    = pd.concat([df.drop(columns=["Blood Pressure"]), bp_df], axis=1)

    # ── 3. Encode target  ─────────────────────────────────────────────────────
    target_le = LabelEncoder()
    y = target_le.fit_transform(df[TARGET_COL].fillna("None"))
    df.drop(columns=[TARGET_COL], inplace=True)

    # ── 4. Encode categorical columns ─────────────────────────────────────────
    if fit:
        label_encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            # Handle unseen labels gracefully
            df[col] = df[col].astype(str).map(
                lambda v, le=le: le.transform([v])[0]
                if v in le.classes_ else 0
            )

    # ── 5. Fill any remaining NaN with median ─────────────────────────────────
    df.fillna(df.median(numeric_only=True), inplace=True)

    feature_names = list(df.columns)
    X = df.values.astype(float)

    # Store target encoder on the label_encoders dict for re-use
    # Must be done BEFORE joblib.dump so it is persisted along with feature encoders
    label_encoders["__target__"] = target_le

    # ── 6. Scale features ─────────────────────────────────────────────────────
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        joblib.dump(scaler,         SCALER_PATH)
        joblib.dump(label_encoders, ENCODER_PATH)   # now includes __target__
    else:
        X = scaler.transform(X)

    # ── 7. Train / test split (only at training time) ─────────────────────────
    if fit:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    else:
        X_train = X_test = X
        y_train = y_test = y

    return X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names


def encode_single_record(record: dict,
                          scaler: StandardScaler,
                          label_encoders: dict) -> np.ndarray:
    """
    Convert a single user-input dict (from the GUI) into a scaled
    feature vector ready for model.predict().

    record keys must match COLUMNS minus Person ID and Sleep Disorder.
    """
    EXPECTED_FIELDS = [
        "Gender", "Age", "Occupation", "Sleep Duration",
        "Quality of Sleep", "Physical Activity Level",
        "Stress Level", "BMI Category", "Blood Pressure",
        "Heart Rate", "Daily Steps",
    ]
    missing = [f for f in EXPECTED_FIELDS if f not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    df = pd.DataFrame([record])

    if "Blood Pressure" in df.columns:
        bp_df = _parse_blood_pressure(df["Blood Pressure"])
        df    = pd.concat([df.drop(columns=["Blood Pressure"]), bp_df], axis=1)

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        le = label_encoders[col]
        val = str(df[col].iloc[0])
        df[col] = le.transform([val])[0] if val in le.classes_ else 0

    df.fillna(0, inplace=True)
    return scaler.transform(df.values.astype(float))


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_dataset
    df = load_dataset()
    X_tr, X_te, y_tr, y_te, sc, les, feats = preprocess(df)
    print("Feature names :", feats)
    print("Train shape   :", X_tr.shape)
    print("Test shape    :", X_te.shape)
    print("Classes       :", les["__target__"].classes_)