#!/usr/bin/env python3
import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd

# Allow "python src/run_inference.py" from project root
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from regime_logic import MarketMoodIndex, MoodConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--model", type=str, default="artifacts/regime_model.joblib")
    ap.add_argument("--out", type=str, default="artifacts/mood_index_output.csv")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    features = bundle["features"]

    df = pd.read_csv(args.data, parse_dates=["date"])

    X = df[features].copy()
    proba = model.predict_proba(X)

    out = df[["date","asset_id","label"]].copy()
    out["p_normal"] = proba[:,0]
    out["p_warning"] = proba[:,1]
    out["p_critical"] = proba[:,2]

    engine = MarketMoodIndex(MoodConfig())
    out2 = engine.run(out)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out2.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}  rows={len(out2):,}")
    print(out2.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
