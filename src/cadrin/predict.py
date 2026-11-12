"""Predict drug responses from feature files."""

from __future__ import annotations

from pathlib import Path

from .bundle import FrozenCaDRiNPredictor
from .config import load_feature_config
from .data import prepare_features, read_pair_keys
from .utils import read_json, require_file


MTR_ENCODER_FIELDS = (
    "encoder_key",
    "repo_id",
    "model_files_tree_sha256",
    "pooling",
    "source_tensor",
    "max_token_length",
    "truncation",
    "feature_width",
    "numeric_dtype",
    "feature_column_prefix",
    "feature_columns",
)


def validate_mtr_encoder(bundle_dir, dataset_config):
    """Check that query MTR features use the fitted model's encoder."""

    source = read_json(require_file(Path(bundle_dir) / "mtr_audit.json", "bundle MTR audit"))
    external_path = require_file(dataset_config["mtr_audit_path"], "dataset MTR audit")
    external = read_json(external_path)
    mismatches = [
        field
        for field in MTR_ENCODER_FIELDS
        if source["encoder"].get(field) != external["encoder"].get(field)
    ]
    if source.get("feature_spec_version") != external.get("feature_spec_version"):
        mismatches.append("feature_spec_version")
    if mismatches:
        raise ValueError(f"MTR encoder settings differ: {sorted(set(mismatches))}")
    return {
        "feature_spec_version": external["feature_spec_version"],
        "encoder_key": external["encoder"]["encoder_key"],
        "feature_width": int(external["encoder"]["feature_width"]),
    }


def predict_dataset(bundle_dir, dataset_config, pair_path=None, *, device="cpu", batch_size=4096):
    """Predict pairs with a complete bundle directory or ZIP archive.

    Features may be a dictionary or a JSON file whose paths are resolved
    relative to the configuration file. ``pair_path``
    can be omitted when that file includes ``pairs_path``. The return value is
    a ``(FrozenPrediction, metadata)`` tuple; predictions are in ``.frame``.
    """

    if isinstance(dataset_config, (str, Path)):
        dataset_config = load_feature_config(dataset_config)
    if pair_path is None:
        pair_path = dataset_config["pairs_path"]

    predictor = FrozenCaDRiNPredictor.load(bundle_dir, device=device)
    metadata = validate_mtr_encoder(predictor.bundle_dir, dataset_config)
    pairs = read_pair_keys(pair_path)
    prepared = prepare_features(dataset_config, predictor.scalers, pairs)
    invalid = sum(
        key is None or str(key).startswith("invalid::")
        for key in prepared.structure_keys
    )
    metadata["structure_identity"] = {
        "drugs": int(len(prepared.drug_ids)),
        "valid_canonical_structures": int(len(prepared.drug_ids) - invalid),
        "missing_or_invalid_structures": int(invalid),
    }
    prediction = predictor.predict(prepared, batch_size=batch_size)
    return prediction, metadata
