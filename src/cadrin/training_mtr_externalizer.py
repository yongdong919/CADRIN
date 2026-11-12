"""Training-only Morgan/MTR externalizer for unseen drug main effects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold

from .mtr_externalizer import (
    MorganMTRExternalizer, predict_tanimoto_knn, read_mtr_table,
    save_externalizer_model,
)
from .routing import canonical_structure_key
from .utils import read_json, require_file


FEATURE_PREFIX = "chemberta::"


def validate_training_mtr(config, splits):
    """Check MTR inputs and exportable training identities before fitting."""

    metadata = read_json(require_file(config["mtr_audit"], "training MTR metadata"))
    frame, columns = read_mtr_table(
        require_file(config["mtr_features"], "training MTR table"),
        expected_width=int(config.get("mtr_width", 384)),
    )
    if metadata.get("outputs", {}).get("shape") != [len(frame), len(columns)]:
        raise ValueError("MTR feature shape does not match its encoder metadata")
    encoder = metadata.get("encoder", {})
    if not metadata.get("feature_spec_version"):
        raise ValueError("MTR metadata is missing feature_spec_version")
    for name in ("encoder_key", "model_files_tree_sha256"):
        if not encoder.get(name):
            raise ValueError(f"MTR encoder metadata is missing {name}")
    if encoder.get("feature_width") != len(columns):
        raise ValueError("MTR feature width does not match its encoder metadata")
    if encoder.get("feature_columns") != columns:
        raise ValueError("MTR feature columns do not match their encoder metadata")

    indexed = frame.set_index("master_drug_id")
    for split, pairs in splits.items():
        drug_ids = pd.Index(pairs["master_drug_id"].astype(str).unique())
        missing = drug_ids.difference(indexed.index)
        if len(missing):
            raise ValueError(f"MTR table is missing {split} drugs: {missing[:5].tolist()}")
    training_ids = splits["train"]["master_drug_id"].astype(str).unique()
    structures = indexed.loc[training_ids, "canonical_smiles"].map(canonical_structure_key)
    if structures.isna().any() or structures.str.startswith("invalid::", na=False).any():
        raise ValueError("Training drug structures must be present and valid SMILES")
    if structures.duplicated().any():
        raise ValueError("Training canonical structure keys must be unique")


def load_mtr_features(path, *, expected_width: int = 384):
    path = Path(path).expanduser().resolve()
    frame = pd.read_csv(path, sep="\t")
    if "master_drug_id" not in frame:
        raise ValueError("MTR feature table is missing master_drug_id")
    invalid_ids = (
        frame["master_drug_id"].isna().any()
        or frame["master_drug_id"].duplicated().any()
    )
    if invalid_ids:
        raise ValueError("MTR drug identifiers must be present and unique")
    columns = sorted(
        [column for column in frame if str(column).startswith(FEATURE_PREFIX)],
        key=lambda column: int(str(column).split("::")[1]),
    )
    expected = [f"{FEATURE_PREFIX}{index}" for index in range(expected_width)]
    if columns != expected:
        raise ValueError(
            f"MTR feature columns must be contiguous from 0 to {expected_width - 1}"
        )
    values = frame[columns].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("MTR feature table contains non-finite values")
    features = pd.DataFrame(
        values,
        index=frame["master_drug_id"].astype(str),
        columns=columns,
    )
    return features, {
        "rows": int(len(features)),
        "width": int(features.shape[1]),
    }


def _metrics(observed, predicted):
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    pearson = (
        float(np.corrcoef(observed, predicted)[0, 1])
        if len(observed) >= 2
        and np.std(observed) > 0
        and np.std(predicted) > 0
        else None
    )
    return {
        "rmse": float(np.sqrt(np.mean(np.square(observed - predicted)))),
        "pearson": pearson,
    }


def fit_morgan_mtr_externalizer(
    drug_ids,
    morgan_features,
    mtr_features,
    target,
    *,
    k: int,
    gamma: float,
    tau: float,
    outer_folds: int,
    seed: int,
    extra_trees,
    groups=None,
):
    drug_ids = np.asarray(drug_ids).astype(str)
    morgan_features = np.asarray(morgan_features, dtype=np.float64)
    mtr_features = np.asarray(mtr_features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (
        target.ndim != 1
        or len(drug_ids) != len(target)
        or len(morgan_features) != len(target)
        or len(mtr_features) != len(target)
    ):
        raise ValueError("Morgan/MTR entity arrays must be aligned")
    if len(np.unique(drug_ids)) != len(drug_ids):
        raise ValueError("Training drug identifiers must be unique")
    if not (
        np.isfinite(morgan_features).all()
        and np.isfinite(mtr_features).all()
        and np.isfinite(target).all()
    ):
        raise ValueError("Morgan/MTR externalizer inputs must be finite")
    n_estimators = int(extra_trees["n_estimators"])
    max_features = float(extra_trees["max_features"])
    min_samples_leaf = int(extra_trees["min_samples_leaf"])
    n_jobs = int(extra_trees.get("n_jobs", -1))
    if n_estimators <= 0:
        raise ValueError("ExtraTrees n_estimators must be positive")
    if not 0 < max_features <= 1:
        raise ValueError("ExtraTrees max_features must be in (0, 1]")
    if min_samples_leaf <= 0:
        raise ValueError("ExtraTrees min_samples_leaf must be positive")

    if groups is None:
        splitter = KFold(
            n_splits=int(outer_folds),
            shuffle=True,
            random_state=int(seed),
        )
        folds = splitter.split(drug_ids)
        protocol = "training-drug cross-fitting"
        group_count = int(len(drug_ids))
    else:
        groups = np.asarray(groups).astype(str)
        if len(groups) != len(drug_ids):
            raise ValueError("Cross-fitting groups are not aligned to drugs")
        group_count = int(len(np.unique(groups)))
        if group_count < int(outer_folds):
            raise ValueError("Too few groups for grouped cross-fitting")
        splitter = GroupKFold(n_splits=int(outer_folds))
        folds = splitter.split(drug_ids, groups=groups)
        protocol = "training-scaffold grouped cross-fitting"

    fold_index = np.empty(len(target), dtype=np.int64)
    morgan_oof = np.empty(len(target), dtype=np.float64)
    mtr_oof = np.empty(len(target), dtype=np.float64)
    neighbor_tables = []
    for fold, (train_index, valid_index) in enumerate(folds):
        if groups is not None and (
            set(groups[train_index]) & set(groups[valid_index])
        ):
            raise RuntimeError("Cross-fitting leaked a scaffold group")
        fold_index[valid_index] = fold
        morgan_oof[valid_index], neighbors = predict_tanimoto_knn(
            morgan_features[valid_index],
            drug_ids[valid_index],
            morgan_features[train_index],
            drug_ids[train_index],
            target[train_index],
            k=int(k),
            gamma=float(gamma),
            tau=float(tau),
        )
        neighbors.insert(0, "fold", fold)
        neighbor_tables.append(neighbors)
        estimator = _fit_mtr_extra_trees(
            mtr_features[train_index],
            target[train_index],
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            seed=int(seed) + fold,
            n_jobs=n_jobs,
        )
        mtr_oof[valid_index] = estimator.predict(mtr_features[valid_index])

    stack = LinearRegression(positive=True)
    stack_features = np.column_stack((morgan_oof, mtr_oof))
    stack.fit(stack_features, target)
    stack_oof = stack.predict(stack_features)
    full_mtr = _fit_mtr_extra_trees(
        mtr_features,
        target,
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        seed=int(seed),
        n_jobs=n_jobs,
    )
    externalizer = MorganMTRExternalizer(
        library_ids=drug_ids,
        library_morgan=morgan_features,
        library_target=target,
        mtr_estimator=full_mtr,
        stack_intercept=float(stack.intercept_),
        stack_coefficients=np.asarray(stack.coef_, dtype=np.float64),
        k=int(k),
        gamma=float(gamma),
        tau=float(tau),
    )
    oof = pd.DataFrame(
        {
            "master_drug_id": drug_ids,
            "fold": fold_index,
            "observed_drug_effect": target,
            "morgan_knn_prediction": morgan_oof,
            "mtr_extra_trees_prediction": mtr_oof,
            "stack_prediction": stack_oof,
        }
    )
    if groups is not None:
        oof["cv_group"] = groups
    audit = {
        "protocol": protocol,
        "outer_folds": int(outer_folds),
        "groups": group_count,
        "seed": int(seed),
        "morgan_knn": {
            "k": int(k),
            "gamma": float(gamma),
            "tau": float(tau),
            "oof": _metrics(target, morgan_oof),
        },
        "nonnegative_stack": {
            "intercept": externalizer.stack_intercept,
            "morgan_weight": float(externalizer.stack_coefficients[0]),
            "mtr_weight": float(externalizer.stack_coefficients[1]),
            "oof": _metrics(target, stack_oof),
        },
    }
    audit["mtr_extra_trees"] = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "min_samples_leaf": min_samples_leaf,
        "oof": _metrics(target, mtr_oof),
    }
    neighbors = pd.concat(neighbor_tables, ignore_index=True)
    return externalizer, oof, neighbors, audit


def _fit_mtr_extra_trees(
    features,
    target,
    *,
    n_estimators: int,
    max_features: float,
    min_samples_leaf: int,
    seed: int,
    n_jobs: int,
):
    estimator = ExtraTreesRegressor(
        n_estimators=int(n_estimators),
        max_features=float(max_features),
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(seed),
        n_jobs=int(n_jobs),
    )
    estimator.fit(features, target)
    return estimator


def externalizer_state(externalizer: MorganMTRExternalizer):
    estimator = externalizer.mtr_estimator
    return {
        "method": (
            "training_oof_morgan_3nn_mtr_extra_trees_nonnegative_stack"
        ),
        "artifact": "drug_externalizer.joblib",
        "library_ids": externalizer.library_ids.tolist(),
        "library_morgan": externalizer.library_morgan.astype(np.float32),
        "library_target": externalizer.library_target.astype(np.float32),
        "k": externalizer.k,
        "gamma": externalizer.gamma,
        "tau": externalizer.tau,
        "n_estimators": int(estimator.n_estimators),
        "max_features": float(estimator.max_features),
        "min_samples_leaf": int(estimator.min_samples_leaf),
        "random_state": int(estimator.random_state),
        "stack_intercept": externalizer.stack_intercept,
        "stack_coefficients": externalizer.stack_coefficients.astype(np.float32),
    }


def save_externalizer(path, externalizer, state):
    path = Path(path)
    np.savez_compressed(
        path,
        library_ids=np.asarray(state["library_ids"], dtype=str),
        library_morgan=state["library_morgan"],
        library_target=state["library_target"],
        k=np.asarray([state["k"]], dtype=np.int64),
        gamma=np.asarray([state["gamma"]], dtype=np.float32),
        tau=np.asarray([state["tau"]], dtype=np.float32),
        n_estimators=np.asarray([state["n_estimators"]], dtype=np.int64),
        max_features=np.asarray([state["max_features"]], dtype=np.float32),
        min_samples_leaf=np.asarray(
            [state["min_samples_leaf"]], dtype=np.int64
        ),
        random_state=np.asarray([state["random_state"]], dtype=np.int64),
        stack_intercept=np.asarray([state["stack_intercept"]], dtype=np.float32),
        stack_coefficients=state["stack_coefficients"],
    )
    save_externalizer_model(
        externalizer,
        path.with_name("drug_externalizer.joblib"),
    )
