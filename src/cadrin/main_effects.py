"""Training-only main-effect decomposition and externalization utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PAIR_COLUMNS = ["cellosaurus_id", "master_drug_id"]


@dataclass
class TwoWayEffects:
    global_mean: float
    drug: pd.DataFrame
    cell: pd.DataFrame
    residual: np.ndarray
    iterations: int
    converged: bool
    tolerance: float
    maximum_drug_residual_mean: float
    maximum_cell_residual_mean: float
    overall_residual_mean: float


def _factorize(values) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(values).astype(str)
    categories = np.asarray(sorted(np.unique(labels)), dtype=object)
    lookup = {label: index for index, label in enumerate(categories)}
    codes = np.asarray([lookup[label] for label in labels], dtype=np.int64)
    return codes, categories


def fit_two_way_effects(
    frame: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> TwoWayEffects:
    """Fit unregularized two-way least squares on an observed training graph."""

    required = {*PAIR_COLUMNS, "response_value"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Two-way frame is missing columns: {sorted(missing)}")
    y = frame["response_value"].to_numpy(np.float64)
    if not np.isfinite(y).all():
        raise ValueError("Two-way response contains non-finite values")
    drug_code, drug_ids = _factorize(frame["master_drug_id"])
    cell_code, cell_ids = _factorize(frame["cellosaurus_id"])
    drug_count = np.bincount(drug_code, minlength=len(drug_ids)).astype(np.float64)
    cell_count = np.bincount(cell_code, minlength=len(cell_ids)).astype(np.float64)

    global_mean = float(np.mean(y))
    drug_effect = np.zeros(len(drug_ids), dtype=np.float64)
    cell_effect = np.zeros(len(cell_ids), dtype=np.float64)
    converged = False
    for iteration in range(1, max_iterations + 1):
        previous = global_mean + drug_effect[drug_code] + cell_effect[cell_code]
        global_mean = float(
            np.mean(y - drug_effect[drug_code] - cell_effect[cell_code])
        )
        drug_effect = np.bincount(
            drug_code,
            weights=y - global_mean - cell_effect[cell_code],
            minlength=len(drug_ids),
        ) / drug_count
        shift = float(np.dot(drug_count, drug_effect) / drug_count.sum())
        drug_effect -= shift
        global_mean += shift
        cell_effect = np.bincount(
            cell_code,
            weights=y - global_mean - drug_effect[drug_code],
            minlength=len(cell_ids),
        ) / cell_count
        shift = float(np.dot(cell_count, cell_effect) / cell_count.sum())
        cell_effect -= shift
        global_mean += shift
        current = global_mean + drug_effect[drug_code] + cell_effect[cell_code]
        if float(np.max(np.abs(current - previous))) <= tolerance:
            converged = True
            break

    residual = y - global_mean - drug_effect[drug_code] - cell_effect[cell_code]
    drug_residual_mean = np.bincount(
        drug_code, weights=residual, minlength=len(drug_ids)
    ) / drug_count
    cell_residual_mean = np.bincount(
        cell_code, weights=residual, minlength=len(cell_ids)
    ) / cell_count
    raw_drug = (
        np.bincount(drug_code, weights=y, minlength=len(drug_ids)) / drug_count
        - global_mean
    )
    raw_cell = (
        np.bincount(cell_code, weights=y, minlength=len(cell_ids)) / cell_count
        - global_mean
    )
    return TwoWayEffects(
        global_mean=global_mean,
        drug=pd.DataFrame(
            {
                "master_drug_id": drug_ids,
                "effect": drug_effect,
                "raw_marginal_effect": raw_drug,
                "observations": drug_count.astype(np.int64),
            }
        ),
        cell=pd.DataFrame(
            {
                "cellosaurus_id": cell_ids,
                "effect": cell_effect,
                "raw_marginal_effect": raw_cell,
                "observations": cell_count.astype(np.int64),
            }
        ),
        residual=residual,
        iterations=iteration,
        converged=converged,
        tolerance=float(tolerance),
        maximum_drug_residual_mean=float(np.max(np.abs(drug_residual_mean))),
        maximum_cell_residual_mean=float(np.max(np.abs(cell_residual_mean))),
        overall_residual_mean=float(np.mean(residual)),
    )


def build_entity_features(dataset, entity: str, ids) -> np.ndarray:
    ids = np.asarray(ids).astype(str)
    frame = dataset.frame
    if entity == "drug":
        unique = frame.drop_duplicates("master_drug_id").set_index("master_drug_id")
        indices = unique.loc[ids, "fingerprint_idx"].to_numpy(np.int64)
        return dataset.fingerprint.numpy()[indices]
    if entity == "cell":
        unique = frame.drop_duplicates("cellosaurus_id").set_index("cellosaurus_id")
        context = dataset.context.numpy()[
            unique.loc[ids, "context_idx"].to_numpy(np.int64)
        ]
        viper = dataset.viper.numpy()[
            unique.loc[ids, "viper_idx"].to_numpy(np.int64)
        ]
        return np.concatenate((context, viper), axis=1)
    raise ValueError(f"Unsupported entity type: {entity}")


def fit_ridge_externalizer(features, target, alpha: float):
    estimator = make_pipeline(
        StandardScaler(),
        Ridge(alpha=float(alpha), solver="lsqr"),
    )
    estimator.fit(
        np.asarray(features, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
    )
    return estimator


def binary_tanimoto(query: np.ndarray, library: np.ndarray) -> np.ndarray:
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
    k: int,
    gamma: float,
    tau: float,
) -> tuple[np.ndarray, pd.DataFrame]:
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
    neighbor_rows = []
    for query_index, query_id in enumerate(query_ids):
        order = np.lexsort(
            (library_ids, -similarities[query_index])
        )[: int(k)]
        selected_similarity = similarities[query_index, order]
        weights = (
            np.ones(int(k), dtype=np.float64)
            if float(gamma) == 0
            else np.power(selected_similarity, float(gamma))
        )
        local_prediction = (
            np.zeros(targets.shape[1], dtype=np.float64)
            if float(weights.sum()) <= 0
            else np.dot(weights, targets[order]) / weights.sum()
        )
        maximum_similarity = float(selected_similarity[0])
        confidence = (
            1.0
            if float(tau) == 0
            else maximum_similarity / (maximum_similarity + float(tau))
        )
        predictions[query_index] = confidence * local_prediction
        for rank, library_index in enumerate(order, start=1):
            neighbor_rows.append(
                {
                    "query_id": query_id,
                    "neighbor_id": library_ids[library_index],
                    "rank": rank,
                    "similarity": float(
                        similarities[query_index, library_index]
                    ),
                    "weight_before_normalization": float(weights[rank - 1]),
                    "confidence": confidence,
                }
            )
    return (
        predictions[:, 0] if scalar_target else predictions,
        pd.DataFrame(neighbor_rows),
    )
