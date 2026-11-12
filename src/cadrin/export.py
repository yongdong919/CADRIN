"""Save a trained composite model as a self-contained inference bundle."""

from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import torch

from .mtr_externalizer import read_mtr_table, save_externalizer_model
from .predict import MTR_ENCODER_FIELDS
from .routing import canonical_structure_key
from .utils import read_json, require_file, write_json


COPIED_ARTIFACTS = (
    "scalers.npz",
    "training_drug_effects.csv",
)


def _prediction_checkpoint(checkpoint):
    """Keep fitted prediction state without training records or input paths."""

    state = {name: checkpoint[name] for name in (
        "model_state_dict", "model_config", "dimensions", "global_mean",
        "response_profile_state",
    )}
    state["response_profile_state"] = {
        name: value for name, value in state["response_profile_state"].items()
        if name != "singular_values"
    }
    correction = checkpoint.get("random_pair_correction_state")
    if correction is not None:
        state["random_pair_correction_state"] = {name: correction[name] for name in (
            "drug_ids", "cell_ids", "drug_factors", "cell_factors", "fusion_weights",
        )}
    return state


def _prediction_mtr_metadata(metadata):
    """Keep the encoder settings checked when loading query features."""

    return {
        "feature_spec_version": metadata["feature_spec_version"],
        "encoder": {name: metadata["encoder"][name] for name in MTR_ENCODER_FIELDS
                    if name in metadata["encoder"]},
    }


def export_bundle(run_dir, bundle_dir):
    """Export an eight-file prediction bundle to a new or empty directory."""

    run = Path(run_dir).expanduser().resolve()
    out = Path(bundle_dir).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Bundle output directory must be new or empty: {out}")
    manifest = read_json(require_file(run / "run_manifest.json", "training manifest"))
    config = manifest["config"]
    checkpoint = torch.load(
        require_file(run / "checkpoint.pt", "checkpoint"),
        map_location="cpu", weights_only=False,
    )
    if checkpoint["scenario"] != manifest["scenario"]:
        raise ValueError("Checkpoint scenario does not match the training manifest")
    profile = checkpoint["response_profile_state"]
    beta = float(profile["beta"])
    if manifest["scenario"] != "random_pair":
        calibration_path = require_file(
            run / "profile_beta_calibration.json", "cold-scenario profile calibration"
        )
        beta = float(read_json(calibration_path)["selected_beta"])

    out.mkdir(parents=True, exist_ok=True)
    torch.save(_prediction_checkpoint(checkpoint), out / "checkpoint.pt")
    for name in COPIED_ARTIFACTS:
        shutil.copy2(require_file(run / name, name), out / name)
    externalizer = joblib.load(
        require_file(run / "drug_externalizer.joblib", "drug externalizer")
    )
    save_externalizer_model(externalizer, out / "drug_externalizer.joblib")
    metadata = read_json(
        require_file(config["main_effect"]["mtr_audit"], "frozen MTR metadata")
    )
    write_json(out / "mtr_audit.json", _prediction_mtr_metadata(metadata))

    drug_ids = np.asarray(profile["drug_ids"]).astype(str)
    mtr, _ = read_mtr_table(config["main_effect"]["mtr_features"])
    mtr["master_drug_id"] = mtr["master_drug_id"].astype(str)
    drugs = mtr[mtr["master_drug_id"].isin(drug_ids)][
        ["master_drug_id", "canonical_smiles"]
    ].copy()
    drugs["structure_key"] = drugs["canonical_smiles"].map(canonical_structure_key)
    if set(drugs["master_drug_id"]) != set(drug_ids):
        raise ValueError("MTR structures do not cover the training drug inventory")
    if (
        drugs["structure_key"].isna().any()
        or drugs["structure_key"].str.startswith("invalid::", na=False).any()
        or drugs["structure_key"].duplicated().any()
    ):
        raise ValueError("Training canonical structure keys must be valid, complete and unique")
    drugs.sort_values("master_drug_id").to_csv(out / "source_drugs.csv", index=False)

    pairs = pd.read_csv(require_file(run / "training_pairs.csv.gz", "training pairs"), dtype=str)
    if pairs.empty or pairs.duplicated().any():
        raise ValueError("Training pair inventory must be nonempty and unique")
    if len(pairs) != int(manifest["data"]["rows"]["train"]):
        raise ValueError("Training pair inventory does not match the fitted data")
    pairs.to_csv(out / "source_pairs.csv.gz", index=False)
    write_json(out / "manifest.json", {"locked_parameters": {"profile_beta": beta}})
    return out
