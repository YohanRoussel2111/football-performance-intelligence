"""
One-shot build script for the demo.

Runs the full offline pipeline so the app (and the SQLite database) are ready:
    1. generate + clean + persist the synthetic season to SQLite
    2. export raw CSVs (to demonstrate the CSV-import path)
    3. train, temporally validate and calibrate the model, save the bundle
    4. score the squad and write the predictions table

Usage:  python build_demo.py
"""
from __future__ import annotations

import time

from src import config
from src.data_processing import pipeline as dp
from src.data_processing import database as db
from src.feature_engineering.features import build_features
from src.analytics.monitoring import compute_pmi
from src.ml import dataset as ds
from src.ml import train as ml_train
from src.ml import inference as ml_infer
from datetime import datetime, timezone


def main():
    t0 = time.time()
    print("① Pipeline: generate → validate → clean → aggregate → SQLite")
    out = dp.run_pipeline(persist=True)
    qr = out["quality_report"]
    print(f"   {qr.n_players} players · {qr.n_observations:,} obs · "
          f"completeness {qr.overall_completeness:.1f}% · flag {qr.quality_flag}")

    print("② Export raw CSVs → data/raw/")
    for name in ["players", "matches", "sessions", "gps_data", "wellness",
                 "physical_tests", "availability", "availability_episodes"]:
        df = out["raw"][name]
        df.to_csv(config.RAW_DIR / f"{name}.csv", index=False)
    print(f"   wrote {len(list(config.RAW_DIR.glob('*.csv')))} CSV files")

    print("③ Train, temporally validate & calibrate model")
    bundle = ml_train.train_and_select(verbose=True)
    ml_train.save_bundle(bundle)
    print(f"   selected {bundle['metadata']['selected_model']} → {ml_train.BUNDLE_PATH.name}")

    print("④ Score squad & write predictions table")
    feats = build_features(out["daily"])
    table = compute_pmi(ds.build_learning_table(feats))
    scores = ml_infer.score_table(bundle, table)
    latest = scores.sort_values("date").groupby("player_id").tail(1).copy()
    latest["model_version"] = bundle["metadata"]["model_version"]
    latest["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db.write_table("predictions", latest[["player_id", "date", "model_version",
                   "risk_probability", "monitoring_level", "generated_at"]])
    print(f"   wrote {len(latest)} predictions")

    print(f"\n✅ Build complete in {time.time() - t0:.1f}s. "
          f"Launch with:  python run_app.py")


if __name__ == "__main__":
    main()
