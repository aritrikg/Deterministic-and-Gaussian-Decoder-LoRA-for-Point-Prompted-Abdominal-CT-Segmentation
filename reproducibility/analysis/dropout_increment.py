"""Estimate the incremental value of decoder MC-dropout MI on fixed outcomes."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from reanalyze import descriptive_t_interval, risk_model


def dropout_increment(replay_path):
    with ZipFile(replay_path) as replay, replay.open("per_prompt.csv") as stream:
        predictions = pd.read_csv(stream)

    prompt = ["key", "volume", "slice_index", "organ", "x", "y"]
    baseline = predictions.loc[predictions.variant.eq("deterministic_single")]
    dropout = predictions.loc[
        predictions.variant.eq("decoder_mc_dropout"), prompt + ["mean_mi_full_nats"]
    ].rename(columns={"mean_mi_full_nats": "dropout_mi"})
    matched = baseline.merge(dropout, on=prompt, validate="one_to_one")
    assert len(matched) == len(baseline) == len(dropout) == 2466

    matched["failure"] = matched.dice.lt(0.5)
    matched["log_area"] = np.log1p(matched.predicted_area_pixels)
    assert matched.failure.sum() == 189 and matched.volume.nunique() == 6
    assert matched.groupby("volume").failure.nunique().eq(2).all()
    features = ["log_area", "mean_entropy_full_nats"]
    assert np.isfinite(matched[features + ["dropout_mi"]]).all().all()

    rows = []
    for volume, held_out in matched.groupby("volume", sort=True):
        fit = matched.loc[matched.volume.ne(volume)]
        base_score = risk_model(fit, held_out, features)
        with_mi_score = risk_model(fit, held_out, features + ["dropout_mi"])
        base_auc = roc_auc_score(held_out.failure, base_score)
        with_mi_auc = roc_auc_score(held_out.failure, with_mi_score)
        rows.append({
            "volume": volume,
            "targets": len(held_out),
            "failures": int(held_out.failure.sum()),
            "organ_area_entropy_auroc": base_auc,
            "organ_area_entropy_dropout_mi_auroc": with_mi_auc,
            "increment": with_mi_auc - base_auc,
            "dropout_mi_alone_auroc": roc_auc_score(held_out.failure, held_out.dropout_mi),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_zip", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    results = dropout_increment(args.replay_zip)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    print(results.to_string(index=False))
    print(results.mean(numeric_only=True).to_string())
    print("Descriptive paired 95% t interval:", descriptive_t_interval(results.increment))
