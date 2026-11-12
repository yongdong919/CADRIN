#!/usr/bin/env python3
"""Select the fixed response-profile coefficient using validation data only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


from .train import regression_metrics
from .utils import write_json


def evaluate_beta(
    frame: pd.DataFrame,
    beta: float,
    trained_beta: float,
    source: str = "configured",
) -> tuple[np.ndarray, dict]:
    if trained_beta == 0:
        raise ValueError("A nonzero training beta is required for calibration")
    if source != "configured":
        raise ValueError(f"Unsupported response-profile source: {source}")
    profile = frame[
        "pred_response_profile_effect"
    ].to_numpy(np.float64)
    prediction = (
        frame["prediction"].to_numpy(np.float64)
        + (float(beta) / trained_beta - 1.0) * profile
    )
    metrics = regression_metrics(
        frame["response_value"].to_numpy(np.float64),
        prediction,
        frame["master_drug_id"].astype(str).to_numpy(),
    )
    return prediction, metrics


def calibrate_response_profile(run):
    """Select the response-profile coefficient on validation data."""

    run = Path(run).expanduser().resolve()
    manifest = json.loads(
        (run / "run_manifest.json").read_text(encoding="utf-8")
    )
    grid = [
        float(value)
        for value in manifest["config"]["response_profile"].get(
            "beta_grid",
            [0.0, 0.25, 0.5, 0.75, 1.0],
        )
    ]
    trained_beta = float(manifest["config"]["response_profile"]["beta"])
    validation = pd.read_csv(run / "validation_predictions.csv.gz")
    validation_rows = []
    for beta in grid:
        _, metrics = evaluate_beta(validation, beta, trained_beta)
        validation_rows.append(
            {"source": "configured", "beta": beta, **metrics}
        )
    selected = min(
        validation_rows,
        key=lambda row: (
            row["rmse"],
            -row["pearson"],
            row["beta"],
        ),
    )
    selected_beta = float(selected["beta"])
    selected_source = str(selected["source"])

    split_metrics = {}
    for split in ("validation", "test"):
        frame = pd.read_csv(run / f"{split}_predictions.csv.gz")
        prediction, metrics = evaluate_beta(
            frame,
            selected_beta,
            trained_beta,
            selected_source,
        )
        frame.insert(
            frame.columns.get_loc("prediction") + 1,
            "prediction_before_beta_calibration",
            frame["prediction"],
        )
        frame["prediction"] = prediction
        frame["selected_profile_source"] = selected_source
        frame["squared_error"] = np.square(
            prediction - frame["response_value"].to_numpy()
        )
        frame["absolute_error"] = np.abs(
            prediction - frame["response_value"].to_numpy()
        )
        frame.to_csv(
            run / f"calibrated_{split}_predictions.csv.gz",
            index=False,
        )
        raw_metrics = json.loads(
            (run / f"{split}_metrics.json").read_text(encoding="utf-8")
        )
        metrics.update(
            {
                "best_epoch": raw_metrics["best_epoch"],
                "selection_metric": (
                    "validation RMSE, then validation profile-beta RMSE"
                ),
                "model_parameters": raw_metrics["model_parameters"],
                "profile_beta": selected_beta,
                "profile_source": selected_source,
            }
        )
        write_json(run / f"calibrated_{split}_metrics.json", metrics)
        split_metrics[split] = metrics

    write_json(
        run / "profile_beta_calibration.json",
        {
            "grid": grid,
            "selected_beta": selected_beta,
            "selected_source": selected_source,
            "selected_by": (
                "minimum validation RMSE; PCC and smaller beta break exact ties"
            ),
            "validation_grid": validation_rows,
            "validation": split_metrics["validation"],
            "test": split_metrics["test"],
        },
    )
    print(
        json.dumps(
            {
                "scenario": manifest["scenario"],
                "selected_beta": selected_beta,
                "selected_source": selected_source,
                "validation": split_metrics["validation"],
                "test": split_metrics["test"],
            },
            indent=2,
        )
    )
    return selected_beta
