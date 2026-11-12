"""Generate the six fixed repeated-holdout GDSC paper protocols."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SCENARIOS


PAIR_COLUMNS = ["cellosaurus_id", "master_drug_id"]
SPLIT_NAMES = ("train", "valid", "test")
CELL_COLD_SCENARIOS = {
    "cold_cell",
    "cold_drug_cell",
    "cold_scaffold_cell",
}
FINGERPRINT_COLD_SCENARIOS = {"cold_drug", "cold_drug_cell"}
SCAFFOLD_COLD_SCENARIOS = {"cold_scaffold", "cold_scaffold_cell"}


def _sample_holdout_keys(keys, fraction: float, seed: int) -> set:
    keys = pd.Series(sorted(pd.Series(keys).dropna().unique(), key=str))
    count = max(1, int(round(len(keys) * float(fraction))))
    if count >= len(keys):
        raise ValueError("Holdout fraction leaves no training groups")
    return set(keys.sample(n=count, random_state=int(seed)))


def _sample_holdout_pairs(pairs, fraction: float, seed: int) -> set:
    pairs = pairs[PAIR_COLUMNS].drop_duplicates().sort_values(PAIR_COLUMNS)
    count = max(1, int(round(len(pairs) * float(fraction))))
    if count >= len(pairs):
        raise ValueError("Holdout fraction leaves no training pairs")
    sampled = pairs.sample(n=count, random_state=int(seed))
    return set(sampled.itertuples(index=False, name=None))


def _pair_mask(frame, pair_keys) -> pd.Series:
    pairs = zip(frame["cellosaurus_id"], frame["master_drug_id"])
    return pd.Series((pair in pair_keys for pair in pairs), index=frame.index)


def _partition_keys(keys, fractions, seed: int) -> dict:
    keys = np.asarray(
        sorted(pd.Series(keys).dropna().unique(), key=str),
        dtype=object,
    )
    if len(keys) < len(SPLIT_NAMES):
        raise ValueError("Strict splitting requires at least three groups")
    rng = np.random.default_rng(int(seed))
    keys = keys[rng.permutation(len(keys))]
    counts = np.floor(np.asarray(fractions) * len(keys)).astype(int)
    counts = np.maximum(counts, 1)
    counts[0] += len(keys) - int(counts.sum())
    if counts[0] <= 0:
        raise ValueError("Strict fractions leave no training groups")
    values = np.concatenate(
        [np.repeat(name, count) for name, count in zip(SPLIT_NAMES, counts)]
    )
    return dict(zip(keys, values))


def _single_axis_assignment(pairs, group_column, seed, fractions):
    test_groups = _sample_holdout_keys(
        pairs[group_column],
        fractions["test"],
        seed,
    )
    train_pool = pairs[~pairs[group_column].isin(test_groups)]
    validation_fraction = fractions["valid"] / (
        fractions["train"] + fractions["valid"]
    )
    valid_groups = _sample_holdout_keys(
        train_pool[group_column],
        validation_fraction,
        seed + 1,
    )
    split = np.full(len(pairs), "train", dtype=object)
    split[pairs[group_column].isin(valid_groups)] = "valid"
    split[pairs[group_column].isin(test_groups)] = "test"
    partitions = {
        str(group): (
            "test"
            if group in test_groups
            else "valid"
            if group in valid_groups
            else "train"
        )
        for group in sorted(pairs[group_column].unique(), key=str)
    }
    return split, partitions


def build_assignments(
    samples,
    fingerprint_group_map,
    scaffold_group_map,
    scenario,
    *,
    seed,
    single_axis_fractions,
    strict_pair_fractions,
):
    """Build one complete pair assignment without reading response labels."""

    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    pairs = (
        samples[PAIR_COLUMNS]
        .drop_duplicates()
        .sort_values(PAIR_COLUMNS)
        .reset_index(drop=True)
    )
    pairs["fingerprint_group"] = pairs["master_drug_id"].map(
        fingerprint_group_map
    )
    pairs["scaffold_group"] = pairs["master_drug_id"].map(scaffold_group_map)
    if pairs[["fingerprint_group", "scaffold_group"]].isna().any().any():
        raise ValueError("Some response drugs have no chemical group")

    if scenario == "random_pair":
        test_pairs = _sample_holdout_pairs(
            pairs,
            single_axis_fractions["test"],
            seed,
        )
        test_mask = _pair_mask(pairs, test_pairs)
        remaining = pairs[~test_mask]
        validation_fraction = single_axis_fractions["valid"] / (
            single_axis_fractions["train"] + single_axis_fractions["valid"]
        )
        valid_pairs = _sample_holdout_pairs(
            remaining,
            validation_fraction,
            seed + 1,
        )
        valid_mask = _pair_mask(pairs, valid_pairs)
        pairs["split"] = "train"
        pairs.loc[valid_mask, "split"] = "valid"
        pairs.loc[test_mask, "split"] = "test"
        entity_partitions = {}
        method = (
            "10% unique pairs for test, then 1/9 of remaining pairs "
            "for validation"
        )
    elif scenario in {"cold_cell", "cold_drug", "cold_scaffold"}:
        group_column = {
            "cold_cell": "cellosaurus_id",
            "cold_drug": "fingerprint_group",
            "cold_scaffold": "scaffold_group",
        }[scenario]
        split, partitions = _single_axis_assignment(
            pairs,
            group_column,
            seed,
            single_axis_fractions,
        )
        pairs["split"] = split
        partition_name = {
            "cold_cell": "cells",
            "cold_drug": "fingerprint_groups",
            "cold_scaffold": "scaffold_groups",
        }[scenario]
        entity_partitions = {partition_name: partitions}
        method = (
            f"10% {partition_name} for test, then 1/9 of remaining "
            f"{partition_name} for validation"
        )
    else:
        target = np.asarray(
            [
                strict_pair_fractions["train"],
                strict_pair_fractions["valid"],
                strict_pair_fractions["test"],
            ],
            dtype=np.float64,
        )
        axis_fractions = np.sqrt(target) / np.sqrt(target).sum()
        drug_group_column = (
            "fingerprint_group"
            if scenario == "cold_drug_cell"
            else "scaffold_group"
        )
        cell_partitions = _partition_keys(
            pairs["cellosaurus_id"],
            axis_fractions,
            seed,
        )
        drug_partitions = _partition_keys(
            pairs[drug_group_column],
            axis_fractions,
            seed + 1,
        )
        pairs["cell_partition"] = pairs["cellosaurus_id"].map(
            cell_partitions
        )
        pairs["drug_partition"] = pairs[drug_group_column].map(
            drug_partitions
        )
        pairs["split"] = np.where(
            pairs["cell_partition"].eq(pairs["drug_partition"]),
            pairs["cell_partition"],
            "excluded",
        )
        partition_name = (
            "fingerprint_groups"
            if scenario == "cold_drug_cell"
            else "scaffold_groups"
        )
        entity_partitions = {
            "cells": {
                str(key): value for key, value in cell_partitions.items()
            },
            partition_name: {
                str(key): value for key, value in drug_partitions.items()
            },
            "axis_target_fractions": dict(
                zip(SPLIT_NAMES, axis_fractions.tolist())
            ),
        }
        method = (
            f"independent cell and {partition_name} partitions proportional "
            "to sqrt(80:10:10); retain only matching diagonal blocks"
        )

    audit = audit_assignments(pairs, scenario)
    return (
        pairs[PAIR_COLUMNS + ["split"]].copy(),
        audit,
        entity_partitions,
        method,
    )


def _overlap_count(frame, left, right, column):
    return int(
        len(
            set(frame.loc[frame["split"].eq(left), column])
            & set(frame.loc[frame["split"].eq(right), column])
        )
    )


def audit_assignments(frame, scenario):
    """Audit pair coverage and all scenario-specific entity constraints."""

    if frame.duplicated(PAIR_COLUMNS).any():
        raise ValueError("Split contains duplicate cell-drug pairs")
    invalid = set(frame["split"]) - {*SPLIT_NAMES, "excluded"}
    if invalid:
        raise ValueError(f"Invalid split labels: {sorted(invalid)}")
    for name in SPLIT_NAMES:
        if not frame["split"].eq(name).any():
            raise ValueError(f"{scenario} produced an empty {name} split")

    included = frame[frame["split"].isin(SPLIT_NAMES)]
    counts = {}
    for name in (*SPLIT_NAMES, "excluded"):
        subset = frame[frame["split"].eq(name)]
        counts[name] = {
            "pairs": int(len(subset)),
            "cells": int(subset["cellosaurus_id"].nunique()),
            "drugs": int(subset["master_drug_id"].nunique()),
            "fingerprint_groups": int(
                subset["fingerprint_group"].nunique()
            ),
            "scaffold_groups": int(subset["scaffold_group"].nunique()),
        }

    overlaps = {}
    for left, right in (
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ):
        prefix = f"{left}_{right}"
        for name, column in (
            ("cells", "cellosaurus_id"),
            ("drugs", "master_drug_id"),
            ("fingerprint_groups", "fingerprint_group"),
            ("scaffold_groups", "scaffold_group"),
        ):
            overlaps[f"{prefix}_{name}"] = _overlap_count(
                included,
                left,
                right,
                column,
            )

    if scenario in CELL_COLD_SCENARIOS and any(
        value for key, value in overlaps.items() if key.endswith("_cells")
    ):
        raise RuntimeError(f"{scenario} leaked cells")
    if scenario in FINGERPRINT_COLD_SCENARIOS and any(
        value
        for key, value in overlaps.items()
        if key.endswith("_fingerprint_groups")
    ):
        raise RuntimeError(f"{scenario} leaked fingerprint groups")
    if scenario in SCAFFOLD_COLD_SCENARIOS and any(
        value
        for key, value in overlaps.items()
        if key.endswith("_scaffold_groups")
    ):
        raise RuntimeError(f"{scenario} leaked scaffold groups")

    return {
        "available_pairs": int(len(frame)),
        "included_pairs": int(len(included)),
        "excluded_pairs": int(frame["split"].eq("excluded").sum()),
        "included_row_fractions": {
            name: float(included["split"].eq(name).mean())
            for name in SPLIT_NAMES
        },
        "counts": counts,
        "overlaps": overlaps,
    }
