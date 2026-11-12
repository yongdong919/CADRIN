"""Data loading for the fixed GDSC paper splits."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .training_mtr_externalizer import validate_training_mtr
from .utils import write_json


CONTEXT_PATTERN = re.compile(r"^PC(\d+)-PC(\d+)$")
PAIR_COLUMNS = ["cellosaurus_id", "master_drug_id"]
VIPER_METADATA = {
    "cellosaurus_id",
    "cell_line_name",
    "tissue",
    "has_expression",
    "has_mutation",
    "has_cnv_gistic",
    "has_expr_mut_cnv",
}


class DrugResponseDataset(Dataset):
    def __init__(self, frame, fingerprint, context, viper):
        self.frame = frame.reset_index(drop=True)
        self.fingerprint = fingerprint
        self.context = context
        self.viper = viper
        self.fingerprint_idx = torch.as_tensor(
            self.frame["fingerprint_idx"].to_numpy(np.int64)
        )
        self.context_idx = torch.as_tensor(
            self.frame["context_idx"].to_numpy(np.int64)
        )
        self.viper_idx = torch.as_tensor(
            self.frame["viper_idx"].to_numpy(np.int64)
        )
        self.drug_idx = torch.as_tensor(
            self.frame["drug_idx"].to_numpy(np.int64)
        )
        self.drug_mean = torch.as_tensor(
            self.frame["drug_mean_offset"].to_numpy(np.float32)
        )
        self.drug_mean_mask = torch.as_tensor(
            self.frame["drug_mean_mask"].to_numpy(np.float32)
        )
        self.response = torch.as_tensor(
            self.frame["response_value"].to_numpy(np.float32)
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return (
            self.fingerprint[self.fingerprint_idx[index]],
            self.context[self.context_idx[index]],
            self.viper[self.viper_idx[index]],
            self.drug_idx[index],
            self.drug_mean[index],
            self.drug_mean_mask[index],
            self.response[index],
        )


@dataclass
class PreparedData:
    train: DrugResponseDataset
    valid: DrugResponseDataset
    test: DrugResponseDataset
    dimensions: dict
    global_mean: float
    manifest: dict


def select_context_columns(frame, tolerance=1e-7):
    parsed = {}
    for column in frame.columns:
        if column in {"sample_id", "Set"}:
            continue
        match = CONTEXT_PATTERN.fullmatch(str(column))
        if match is None:
            raise ValueError(f"Invalid Context edge name: {column!r}")
        parsed[(int(match.group(1)), int(match.group(2)))] = column

    nodes = sorted({node for pair in parsed for node in pair})
    if nodes != list(range(len(nodes))):
        raise ValueError("Context PC indices must be contiguous and start at PC0")
    expected = {(i, j) for i in nodes for j in nodes}
    if set(parsed) != expected:
        raise ValueError("Context must contain the complete square network")

    square_columns = [parsed[(i, j)] for i in nodes for j in nodes]
    matrix = frame[square_columns].to_numpy(np.float64).reshape(
        len(frame),
        len(nodes),
        len(nodes),
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Context contains non-finite edge weights")
    asymmetry = np.max(np.abs(matrix - matrix.transpose(0, 2, 1)))
    if float(asymmetry) > tolerance:
        raise ValueError(f"Context is not symmetric: max difference={asymmetry}")
    return [parsed[(i, j)] for i in nodes for j in nodes if i < j]


def _read_tables(paths):
    context = pd.read_csv(paths["context"])
    fingerprint = pd.read_csv(paths["fingerprint"], sep="\t")
    viper = pd.read_csv(paths["viper"], sep="\t")
    response = pd.read_csv(paths["response"], sep="\t")

    keys = (
        (context, "sample_id", "Context"),
        (fingerprint, "master_drug_id", "fingerprint"),
        (viper, "cellosaurus_id", "VIPER"),
    )
    for frame, key, label in keys:
        if key not in frame:
            raise ValueError(f"{label} is missing {key!r}")
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"{label} identifiers must be present and unique")

    context_columns = select_context_columns(context)
    fingerprint_columns = sorted(
        [column for column in fingerprint if str(column).startswith("bit_")],
        key=lambda column: int(str(column).split("_")[1]),
    )
    expected_bits = [f"bit_{index}" for index in range(len(fingerprint_columns))]
    if fingerprint_columns != expected_bits:
        raise ValueError("Morgan columns must be contiguous from bit_0")
    viper_columns = [column for column in viper if column not in VIPER_METADATA]
    if not viper_columns:
        raise ValueError("No VIPER features were found")

    required_response = {
        "sample_id",
        "source_dataset",
        "cellosaurus_id",
        "master_drug_id",
        "response_value",
    }
    missing = required_response - set(response)
    if missing:
        raise ValueError(f"Response table is missing: {sorted(missing)}")
    return (
        context,
        fingerprint,
        viper,
        response,
        context_columns,
        fingerprint_columns,
        viper_columns,
    )


def _deduplicate_and_filter(response, context, fingerprint, viper):
    raw_response = response.copy()
    response = response.copy()
    response["response_value"] = pd.to_numeric(
        response["response_value"],
        errors="coerce",
    )
    source_rank = response["source_dataset"].map({"GDSC2": 0, "GDSC1": 1})
    response["_source_rank"] = source_rank.fillna(2).astype(int)
    response = response.sort_values(
        PAIR_COLUMNS + ["_source_rank", "sample_id"],
        kind="mergesort",
    ).drop_duplicates(PAIR_COLUMNS, keep="first")
    response = response.drop(columns="_source_rank")

    context_ids = set(context["sample_id"])
    fingerprint_ids = set(fingerprint["master_drug_id"])
    viper_ids = set(viper["cellosaurus_id"])
    response = response[
        response["cellosaurus_id"].isin(context_ids & viper_ids)
        & response["master_drug_id"].isin(fingerprint_ids)
        & np.isfinite(response["response_value"])
    ].copy()
    if response.empty:
        raise ValueError("No complete drug-response pairs remain")
    return raw_response, response


def _load_split(response, split_path, scenario):
    assignments = pd.read_csv(split_path)
    required = {*PAIR_COLUMNS, "split"}
    if required - set(assignments):
        raise ValueError(f"Split file is missing: {sorted(required-set(assignments))}")
    if assignments.duplicated(PAIR_COLUMNS).any():
        raise ValueError("Split file contains duplicate cell-drug pairs")
    if set(assignments["split"]) - {"train", "valid", "test", "excluded"}:
        raise ValueError("Split file contains unsupported labels")

    coverage = assignments[PAIR_COLUMNS].merge(
        response[PAIR_COLUMNS],
        on=PAIR_COLUMNS,
        how="outer",
        indicator=True,
    )
    if not coverage["_merge"].eq("both").all():
        raise ValueError(
            "Fixed split does not exactly match the available response pairs"
        )
    merged = response.merge(
        assignments,
        on=PAIR_COLUMNS,
        validate="one_to_one",
    )
    splits = {
        name: merged[merged["split"] == name].drop(columns="split").copy()
        for name in ("train", "valid", "test")
    }
    if any(frame.empty for frame in splits.values()):
        raise ValueError(f"{scenario} contains an empty split")
    return splits


def _assert_disjoint_entities(splits, column, label):
    entity_sets = {
        name: set(frame[column].astype(str))
        for name, frame in splits.items()
    }
    for left, right in (
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ):
        if entity_sets[left] & entity_sets[right]:
            raise RuntimeError(
                f"{label} overlap between {left} and {right}"
            )


def _assert_disjoint_drug_groups(splits, group_map, label):
    if not group_map:
        raise ValueError(f"Split metadata has no {label} mapping")
    group_sets = {}
    for name, frame in splits.items():
        groups = frame["master_drug_id"].astype(str).map(group_map)
        if groups.isna().any():
            raise ValueError(f"{label} mapping does not cover {name} drugs")
        group_sets[name] = set(groups)
    for left, right in (
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ):
        if group_sets[left] & group_sets[right]:
            raise RuntimeError(
                f"{label} overlap between {left} and {right}"
            )


def _fit_source_calibration(raw_response, train, min_pairs=20):
    train_pairs = train[PAIR_COLUMNS].drop_duplicates()
    paired = raw_response.merge(train_pairs, on=PAIR_COLUMNS, how="inner")
    paired = paired.copy()
    paired["response_value"] = pd.to_numeric(
        paired["response_value"],
        errors="coerce",
    )
    paired = paired[np.isfinite(paired["response_value"])]
    pair_means = paired.groupby(
        PAIR_COLUMNS + ["source_dataset"],
        as_index=False,
    )["response_value"].mean()
    wide = pair_means.pivot(
        index=PAIR_COLUMNS,
        columns="source_dataset",
        values="response_value",
    )
    if "GDSC2" not in wide:
        raise ValueError("GDSC2 is required as the calibration reference")

    coefficients = {"GDSC2": {"slope": 1.0, "intercept": 0.0}}
    sources = sorted(set(train["source_dataset"].astype(str)) | set(pair_means["source_dataset"].astype(str)))
    for source in sources:
        if source == "GDSC2":
            continue
        if source not in wide:
            coefficients[source] = {"slope": 1.0, "intercept": 0.0}
            continue
        values = wide[[source, "GDSC2"]].dropna()
        if len(values) < min_pairs:
            raise ValueError(
                f"Only {len(values)} paired responses are available for {source}"
            )
        intercept = float((values["GDSC2"] - values[source]).mean())
        coefficients[source] = {"slope": 1.0, "intercept": intercept}
    return coefficients


def _apply_source_calibration(frame, coefficients):
    frame = frame.copy()
    raw = frame["response_value"].to_numpy(np.float64)
    calibrated = raw.copy()
    sources = frame["source_dataset"].astype(str).to_numpy()
    for source, coefficient in coefficients.items():
        mask = sources == source
        calibrated[mask] = (
            coefficient["slope"] * raw[mask] + coefficient["intercept"]
        )
    frame["response_value_raw"] = raw
    frame["response_value"] = calibrated
    return frame


def _fit_minmax(values, names, label):
    values = np.asarray(values, dtype=np.float64)
    finite = np.where(np.isfinite(values), values, np.nan)
    if np.isnan(finite).all(axis=0).any():
        raise ValueError(f"{label} contains all-missing training features")
    full_medians = np.nanmedian(finite, axis=0)
    imputed = np.where(np.isfinite(values), values, full_medians)
    minimum = imputed.min(axis=0)
    maximum = imputed.max(axis=0)
    variable = (maximum - minimum) > 0
    if not variable.any():
        raise ValueError(f"All {label} features are constant")
    return {
        "columns": np.asarray(names, dtype=object)[variable].tolist(),
        "mask": variable,
        "median": full_medians[variable].astype(np.float32),
        "minimum": minimum[variable].astype(np.float32),
        "scale": (maximum[variable] - minimum[variable]).astype(np.float32),
    }


def _transform(values, scaler):
    values = np.asarray(values, dtype=np.float32)[:, scaler["mask"]]
    values = np.where(np.isfinite(values), values, scaler["median"])
    values = (values - scaler["minimum"]) / scaler["scale"]
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def _subsample(splits, maximum_rows, seed):
    if maximum_rows is None:
        return splits
    sampled = {}
    for offset, (name, frame) in enumerate(splits.items()):
        sampled[name] = frame.sample(
            n=min(int(maximum_rows), len(frame)),
            random_state=seed + offset,
        ).reset_index(drop=True)
    return sampled


def prepare_data(
    config,
    scenario,
    outdir,
    maximum_rows=None,
    seed=3078,
    split_seed=None,
):
    paths = config["paths"]
    (
        context,
        fingerprint,
        viper,
        raw_response,
        context_columns,
        fingerprint_columns,
        viper_columns,
    ) = _read_tables(paths)
    original_response, available_response = _deduplicate_and_filter(
        raw_response,
        context,
        fingerprint,
        viper,
    )
    requested_split_seed = (
        3078 if split_seed is None else int(split_seed)
    )
    split_root = Path(
        str(paths["split_root"]).format(
            split_seed=requested_split_seed
        )
    )
    split_path = split_root / f"{scenario}.csv.gz"
    split_meta_path = split_path.with_name(f"{scenario}.meta.json")
    if not split_meta_path.is_file():
        raise FileNotFoundError(f"Split metadata is missing: {split_meta_path}")
    split_meta = json.loads(split_meta_path.read_text(encoding="utf-8"))
    if split_meta.get("scenario") != scenario:
        raise ValueError("Split metadata scenario does not match the request")
    recorded_split_seed = int(split_meta["seed"])
    if split_seed is not None and int(split_seed) != recorded_split_seed:
        raise ValueError(
            f"Requested split_seed={split_seed}, but split metadata records "
            f"split_seed={recorded_split_seed}"
        )
    splits = _load_split(available_response, split_path, scenario)
    splits = _subsample(splits, maximum_rows, seed)
    validate_training_mtr(config["main_effect"], splits)

    calibration = _fit_source_calibration(original_response, splits["train"])
    splits = {
        name: _apply_source_calibration(frame, calibration)
        for name, frame in splits.items()
    }
    global_mean = float(splits["train"]["response_value"].mean())

    train_cells = set(splits["train"]["cellosaurus_id"])
    context_raw = context[context_columns].to_numpy(np.float32)
    viper_raw = viper[viper_columns].to_numpy(np.float32)
    context_scaler = _fit_minmax(
        context_raw[context["sample_id"].isin(train_cells)],
        context_columns,
        "Context",
    )
    viper_scaler = _fit_minmax(
        viper_raw[viper["cellosaurus_id"].isin(train_cells)],
        viper_columns,
        "VIPER",
    )
    context_values = _transform(context_raw, context_scaler)
    viper_values = _transform(viper_raw, viper_scaler)
    fingerprint_values = fingerprint[fingerprint_columns].to_numpy(np.float32)
    if (
        not np.isfinite(fingerprint_values).all()
        or (fingerprint_values < 0).any()
        or (fingerprint_values > 1).any()
    ):
        raise ValueError("Morgan fingerprint values must be finite and in [0,1]")

    fingerprint_index = {
        key: index
        for index, key in enumerate(fingerprint["master_drug_id"])
    }
    context_index = {
        key: index for index, key in enumerate(context["sample_id"])
    }
    viper_index = {
        key: index for index, key in enumerate(viper["cellosaurus_id"])
    }
    all_drugs = sorted(
        pd.concat(
            [frame["master_drug_id"] for frame in splits.values()]
        ).unique(),
        key=str,
    )
    drug_index = {key: index for index, key in enumerate(all_drugs)}
    train_drug_means = splits["train"].groupby("master_drug_id")[
        "response_value"
    ].mean()

    def add_indices(frame):
        frame = frame.copy()
        frame["fingerprint_idx"] = frame["master_drug_id"].map(fingerprint_index)
        frame["context_idx"] = frame["cellosaurus_id"].map(context_index)
        frame["viper_idx"] = frame["cellosaurus_id"].map(viper_index)
        frame["drug_idx"] = frame["master_drug_id"].map(drug_index)
        means = frame["master_drug_id"].map(train_drug_means)
        frame["drug_mean_mask"] = means.notna().astype(np.float32)
        frame["drug_mean_offset"] = (
            means.sub(global_mean).fillna(0.0).astype(np.float32)
        )
        if frame[
            ["fingerprint_idx", "context_idx", "viper_idx", "drug_idx"]
        ].isna().any().any():
            raise RuntimeError("Feature index alignment failed")
        return frame.reset_index(drop=True)

    splits = {name: add_indices(frame) for name, frame in splits.items()}
    if scenario in {
        "cold_cell",
        "cold_drug_cell",
        "cold_scaffold_cell",
    }:
        _assert_disjoint_entities(
            splits,
            "cellosaurus_id",
            f"{scenario} cells",
        )
    if scenario in {"cold_drug", "cold_drug_cell"}:
        _assert_disjoint_drug_groups(
            splits,
            split_meta.get("drug_fingerprint_groups"),
            f"{scenario} fingerprint groups",
        )
    if scenario in {"cold_scaffold", "cold_scaffold_cell"}:
        _assert_disjoint_drug_groups(
            splits,
            split_meta.get("drug_scaffold_groups"),
            f"{scenario} scaffolds",
        )

    tensors = {
        "fingerprint": torch.from_numpy(fingerprint_values),
        "context": torch.from_numpy(context_values),
        "viper": torch.from_numpy(viper_values),
    }
    dimensions = {
        "fingerprint": int(fingerprint_values.shape[1]),
        "context": int(context_values.shape[1]),
        "viper": int(viper_values.shape[1]),
    }
    expected_dimensions = {"fingerprint": 1024, "context": 1225, "viper": 277}
    if dimensions != expected_dimensions:
        raise ValueError(
            "CaDRiN feature dimensions changed: "
            f"{dimensions} != {expected_dimensions}"
        )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / "scalers.npz",
        context_columns=np.asarray(context_scaler["columns"], dtype=str),
        context_median=context_scaler["median"],
        context_minimum=context_scaler["minimum"],
        context_scale=context_scaler["scale"],
        context_mask=context_scaler["mask"],
        viper_columns=np.asarray(viper_scaler["columns"], dtype=str),
        viper_median=viper_scaler["median"],
        viper_minimum=viper_scaler["minimum"],
        viper_scale=viper_scaler["scale"],
        viper_mask=viper_scaler["mask"],
        fingerprint_columns=np.asarray(fingerprint_columns, dtype=str),
        global_mean=np.asarray([global_mean], dtype=np.float32),
    )
    manifest = {
        "scenario": scenario,
        "training_seed": int(seed),
        "split_seed": recorded_split_seed,
        "split_audit": split_meta.get("audit"),
        "dimensions": dimensions,
        "global_mean": global_mean,
        "rows": {name: len(frame) for name, frame in splits.items()},
        "cells": {
            name: frame["cellosaurus_id"].nunique()
            for name, frame in splits.items()
        },
        "drugs": {
            name: frame["master_drug_id"].nunique()
            for name, frame in splits.items()
        },
        "source_calibration": calibration,
        "context_selection": "ordered PCi-PCj upper triangle with i<j",
    }
    write_json(outdir / "data_manifest.json", manifest)
    datasets = {
        name: DrugResponseDataset(
            frame,
            tensors["fingerprint"],
            tensors["context"],
            tensors["viper"],
        )
        for name, frame in splits.items()
    }
    return PreparedData(
        train=datasets["train"],
        valid=datasets["valid"],
        test=datasets["test"],
        dimensions=dimensions,
        global_mean=global_mean,
        manifest=manifest,
    )
