"""Drug and cell feature loading."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .mtr_externalizer import read_mtr_table
from .routing import canonical_structure_key
from .utils import require_file


PAIR_KEYS = ["cellosaurus_id", "master_drug_id"]


@dataclass
class PreparedFeatures:
    pairs: pd.DataFrame
    drug_ids: np.ndarray
    fingerprint: np.ndarray
    mtr: np.ndarray
    structure_keys: np.ndarray
    cell_ids: np.ndarray
    context: np.ndarray
    viper: np.ndarray


def read_pair_keys(path):
    path = require_file(path, "response key table")
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    required = ["sample_id", *PAIR_KEYS]
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"Pair table is missing columns: {missing}")
    columns = required + (["source_dataset"] if "source_dataset" in header else [])
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=columns,
        dtype={column: str for column in columns},
    )[columns]
    if frame[required].isna().any().any():
        raise ValueError("Pair keys contain missing values")
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be unique")
    if "source_dataset" not in frame:
        frame["source_dataset"] = "external"
    return frame


def pair_frame_from_reference(frame, source="GDSC"):
    required = set(PAIR_KEYS)
    if required - set(frame):
        raise ValueError("Reference predictions are missing pair keys")
    pairs = frame[[*PAIR_KEYS]].copy()
    pairs.insert(0, "sample_id", [f"{source}_{index}" for index in range(len(pairs))])
    pairs["source_dataset"] = (
        frame["source_dataset"].astype(str).to_numpy()
        if "source_dataset" in frame
        else source
    )
    return pairs


def _index_table(frame, key, label):
    if key not in frame:
        raise ValueError(f"{label} is missing {key}")
    frame = frame.copy()
    if frame[key].isna().any():
        raise ValueError(f"{label} identifiers must be present and unique")
    frame[key] = frame[key].astype(str)
    if frame[key].duplicated().any():
        raise ValueError(f"{label} identifiers must be present and unique")
    return frame.set_index(key)


def _scaler_columns(scalers, prefix):
    for key in (f"{prefix}_input_columns", f"{prefix}_columns"):
        if key in scalers.files:
            return scalers[key].astype(str).tolist()
    raise ValueError(f"Bundle scalers are missing {prefix} columns")


def transform_with_scaler(values, mask, median, minimum, scale):
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    median = np.asarray(median, dtype=np.float32)
    minimum = np.asarray(minimum, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if values.shape[1] == len(mask):
        values = values[:, mask]
    elif values.shape[1] != int(mask.sum()):
        raise ValueError("Feature width is incompatible with the frozen scaler")
    if values.shape[1] != len(median):
        raise ValueError("Frozen scaler arrays are not aligned")
    values = np.where(np.isfinite(values), values, median)
    values = (values - minimum) / scale
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def prepare_features(dataset_config, scalers, pairs):
    pairs = pairs.copy().reset_index(drop=True)
    pairs["master_drug_id"] = pairs["master_drug_id"].astype(str)
    pairs["cellosaurus_id"] = pairs["cellosaurus_id"].astype(str)
    drug_ids = np.asarray(sorted(pairs["master_drug_id"].unique()), dtype=str)
    cell_ids = np.asarray(sorted(pairs["cellosaurus_id"].unique()), dtype=str)

    fingerprint_columns = scalers["fingerprint_columns"].astype(str).tolist()
    fingerprint = pd.read_csv(
        require_file(dataset_config["fingerprint_path"], "Morgan table"),
        sep="\t",
        usecols=["master_drug_id", *fingerprint_columns],
        dtype={"master_drug_id": str},
    )
    fingerprint = _index_table(fingerprint, "master_drug_id", "Morgan table")
    missing = pd.Index(drug_ids).difference(fingerprint.index)
    if len(missing):
        raise ValueError(f"Morgan table is missing drugs: {missing[:5].tolist()}")
    fingerprint_values = fingerprint.loc[drug_ids, fingerprint_columns].to_numpy(
        np.float32
    )
    if not np.isfinite(fingerprint_values).all() or not np.isin(
        fingerprint_values, [0.0, 1.0]
    ).all():
        raise ValueError("Morgan features must be finite and binary")

    mtr_frame, mtr_columns = read_mtr_table(
        require_file(dataset_config["mtr_path"], "MTR table")
    )
    mtr_index = _index_table(mtr_frame, "master_drug_id", "MTR table")
    missing = pd.Index(drug_ids).difference(mtr_index.index)
    if len(missing):
        raise ValueError(f"MTR table is missing drugs: {missing[:5].tolist()}")
    mtr_values = mtr_index.loc[drug_ids, mtr_columns].to_numpy(np.float64)
    structure_keys = mtr_index.loc[drug_ids, "canonical_smiles"].map(
        canonical_structure_key
    ).to_numpy(object)

    context_columns = _scaler_columns(scalers, "context")
    context = pd.read_csv(
        require_file(dataset_config["context_path"], "Context table"),
        usecols=["sample_id", *context_columns],
        dtype={"sample_id": str},
    )
    context = _index_table(context, "sample_id", "Context table")
    missing = pd.Index(cell_ids).difference(context.index)
    if len(missing):
        raise ValueError(f"Context is missing cells: {missing[:5].tolist()}")
    context_values = transform_with_scaler(
        context.loc[cell_ids, context_columns].to_numpy(np.float32),
        scalers["context_mask"],
        scalers["context_median"],
        scalers["context_minimum"],
        scalers["context_scale"],
    )

    viper_columns = _scaler_columns(scalers, "viper")
    viper = pd.read_csv(
        require_file(dataset_config["viper_path"], "VIPER table"),
        sep="\t",
        usecols=["cellosaurus_id", *viper_columns],
        dtype={"cellosaurus_id": str},
    )
    viper = _index_table(viper, "cellosaurus_id", "VIPER table")
    missing = pd.Index(cell_ids).difference(viper.index)
    if len(missing):
        raise ValueError(f"VIPER is missing cells: {missing[:5].tolist()}")
    viper_values = transform_with_scaler(
        viper.loc[cell_ids, viper_columns].to_numpy(np.float32),
        scalers["viper_mask"],
        scalers["viper_median"],
        scalers["viper_minimum"],
        scalers["viper_scale"],
    )

    return PreparedFeatures(
        pairs=pairs,
        drug_ids=drug_ids,
        fingerprint=fingerprint_values,
        mtr=mtr_values,
        structure_keys=structure_keys,
        cell_ids=cell_ids,
        context=context_values,
        viper=viper_values,
    )
