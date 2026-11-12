"""Predict the sampled GDSC test pairs, with optional PCC/RMSE evaluation."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cadrin import predict_dataset
from cadrin.config import load_feature_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, required=True,
        help="Complete CaDRiN bundle directory or ZIP archive",
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "config.json"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output", type=Path, help="Optional prediction CSV path")
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Read the separate test labels after prediction and report pooled PCC/RMSE",
    )
    args = parser.parse_args()

    prediction, _ = predict_dataset(args.bundle, args.config, device=args.device)
    frame = prediction.frame
    print(f"Predicted {len(frame)} GDSC pairs; first ten rows:")
    print(frame[["sample_id", "route", "prediction"]].head(10).to_string(index=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
    if args.evaluate:
        config = load_feature_config(args.config)
        labels = pd.read_csv(config["labels_path"], sep="\t", dtype={"sample_id": str})
        scored = labels.merge(
            frame[["sample_id", "prediction"]], on="sample_id",
            how="left", validate="one_to_one",
        )
        if len(scored) != len(frame) or scored["prediction"].isna().any():
            raise ValueError("Test labels and predicted pairs must match exactly")
        target = scored["response_value"].to_numpy(np.float64)
        predicted = scored["prediction"].to_numpy(np.float64)
        pcc = float(np.corrcoef(target, predicted)[0, 1])
        rmse = float(np.sqrt(np.mean(np.square(predicted - target))))
        print(f"n={len(scored)} pooled_PCC={pcc:.6f} RMSE={rmse:.6f}")


if __name__ == "__main__":
    main()
