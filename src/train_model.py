#!/usr/bin/env python3
import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

FEATURES = [
    "ret_1d","ret_5d","ret_20d","vol_20d","drawdown_60d","rsi_14","volume_z",
    "macro_vix","macro_credit_spread","macro_rates"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--calibrate", action="store_true", help="Enable probability calibration (slower).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data, parse_dates=["date"])

    X = df[FEATURES].copy()
    y = df["label"].astype(int).values
    groups = df["asset_id"].astype(str).values  # keep assets separated

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    pre = ColumnTransformer(
        transformers=[("num", numeric_transformer, FEATURES)],
        remainder="drop"
    )

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=800,
        n_jobs=None,
        random_state=args.seed,
    )

    model = Pipeline(steps=[("pre", pre), ("clf", clf)])

    if args.calibrate:
        # Use small CV for speed; improves probability quality but optional
        model = CalibratedClassifierCV(estimator=model, method="sigmoid", cv=2)

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)
    pred = proba.argmax(axis=1)

    print("=== Classification report (test) ===")
    print(classification_report(y_test, pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print("Log loss:", log_loss(y_test, proba))

    joblib.dump({
        "model": model,
        "features": FEATURES,
        "label_names": ["Normal","Warning","Critical"],
        "calibrated": bool(args.calibrate)
    }, os.path.join(args.out_dir, "regime_model.joblib"))

    print(f"Saved model bundle to {os.path.join(args.out_dir,'regime_model.joblib')}")

if __name__ == "__main__":
    main()
