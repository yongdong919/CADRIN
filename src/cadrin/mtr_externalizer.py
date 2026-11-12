"""Morgan and ChemBERTa-MTR drug-effect prediction."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd


FEATURE_PREFIX = "chemberta::"


def save_externalizer_model(externalizer, path):
    """Save fitted trees with unused node padding cleared."""

    for estimator in externalizer.mtr_estimator.estimators_:
        nodes = estimator.tree_.__getstate__()["nodes"]
        used = np.zeros(nodes.dtype.itemsize, dtype=bool)
        for dtype, offset, *_ in nodes.dtype.fields.values():
            used[offset:offset + dtype.itemsize] = True
        raw = nodes.view(np.uint8).reshape(len(nodes), nodes.dtype.itemsize)
        raw[:, ~used] = 0
    joblib.dump(externalizer, path, compress=3)


def binary_tanimoto(query, library):
    query = np.asarray(query, dtype=np.float64)
    library = np.asarray(library, dtype=np.float64)
    intersection = query @ library.T
    denominator = (
        query.sum(axis=1, keepdims=True)
        + library.sum(axis=1, keepdims=True).T
        - intersection
    )
    return np.divide(
        intersection,
        denominator,
        out=np.zeros_like(intersection),
        where=denominator > 0,
    )


def predict_tanimoto_knn(
    query_features,
    query_ids,
    library_features,
    library_ids,
    library_targets,
    *,
    k,
    gamma,
    tau,
):
    query_ids = np.asarray(query_ids).astype(str)
    library_ids = np.asarray(library_ids).astype(str)
    targets = np.asarray(library_targets, dtype=np.float64)
    scalar_target = targets.ndim == 1
    if scalar_target:
        targets = targets[:, None]
    if targets.ndim != 2 or len(targets) != len(library_ids):
        raise ValueError("KNN targets must align with the neighbor library")
    if not 1 <= int(k) <= len(library_ids):
        raise ValueError(f"k={k} is incompatible with library size={len(library_ids)}")
    if float(gamma) < 0 or float(tau) < 0:
        raise ValueError("gamma and tau must be non-negative")

    similarities = binary_tanimoto(query_features, library_features)
    predictions = np.empty((len(query_ids), targets.shape[1]), dtype=np.float64)
    rows = []
    for query_index, query_id in enumerate(query_ids):
        order = np.lexsort((library_ids, -similarities[query_index]))[: int(k)]
        selected = similarities[query_index, order]
        weights = (
            np.ones(int(k), dtype=np.float64)
            if float(gamma) == 0
            else np.power(selected, float(gamma))
        )
        local = (
            np.zeros(targets.shape[1], dtype=np.float64)
            if float(weights.sum()) <= 0
            else np.dot(weights, targets[order]) / weights.sum()
        )
        maximum = float(selected[0])
        confidence = (
            1.0 if float(tau) == 0 else maximum / (maximum + float(tau))
        )
        predictions[query_index] = confidence * local
        for rank, library_index in enumerate(order, start=1):
            rows.append(
                {
                    "query_id": query_id,
                    "neighbor_id": library_ids[library_index],
                    "rank": rank,
                    "similarity": float(similarities[query_index, library_index]),
                    "weight_before_normalization": float(weights[rank - 1]),
                    "confidence": confidence,
                }
            )
    prediction = predictions[:, 0] if scalar_target else predictions
    return prediction, pd.DataFrame(rows)


@dataclass
class MorganMTRExternalizer:
    library_ids: np.ndarray
    library_morgan: np.ndarray
    library_target: np.ndarray
    mtr_estimator: object
    stack_intercept: float
    stack_coefficients: np.ndarray
    k: int
    gamma: float
    tau: float

    def predict(self, query_ids, query_morgan, query_mtr):
        morgan_prediction, neighbors = predict_tanimoto_knn(
            query_morgan,
            query_ids,
            self.library_morgan,
            self.library_ids,
            self.library_target,
            k=self.k,
            gamma=self.gamma,
            tau=self.tau,
        )
        mtr_prediction = self.mtr_estimator.predict(query_mtr)
        components = np.column_stack((morgan_prediction, mtr_prediction))
        prediction = self.stack_intercept + components @ self.stack_coefficients
        return prediction, neighbors, components


def read_mtr_table(path, expected_width=384):
    frame = pd.read_csv(path, sep="\t", dtype={"master_drug_id": str})
    required = {"master_drug_id", "canonical_smiles"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"MTR table is missing columns: {sorted(missing)}")
    if frame["master_drug_id"].isna().any() or frame[
        "master_drug_id"
    ].duplicated().any():
        raise ValueError("MTR drug identifiers must be present and unique")
    columns = sorted(
        [column for column in frame if str(column).startswith(FEATURE_PREFIX)],
        key=lambda column: int(str(column).split("::")[1]),
    )
    expected = [f"{FEATURE_PREFIX}{index}" for index in range(int(expected_width))]
    if columns != expected:
        raise ValueError(
            f"MTR columns must be contiguous from 0 to {int(expected_width) - 1}"
        )
    values = frame[columns].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("MTR features contain non-finite values")
    return frame, columns
