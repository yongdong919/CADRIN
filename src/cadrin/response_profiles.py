"""Training-only low-rank response profiles and side-information externalizers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .main_effects import (
    build_entity_features,
    fit_ridge_externalizer,
    predict_tanimoto_knn,
)
from .utils import write_json


@dataclass
class ResponseProfileFactors:
    drug_ids: np.ndarray
    cell_ids: np.ndarray
    drug_factors: np.ndarray
    cell_factors: np.ndarray
    singular_values: np.ndarray
    observed_rmse: float
    explained_observed_variance: float


def fit_response_profile_factors(frame, effects, rank: int) -> ResponseProfileFactors:
    """Factorize the zero-filled, strictly centered training residual matrix."""

    if len(frame) != len(effects.residual):
        raise ValueError("ANOVA residual is not aligned to the training pairs")
    if frame.duplicated(["master_drug_id", "cellosaurus_id"]).any():
        raise ValueError("Response-profile training pairs must be unique")

    drug_ids = effects.drug["master_drug_id"].astype(str).to_numpy()
    cell_ids = effects.cell["cellosaurus_id"].astype(str).to_numpy()
    if not 1 <= int(rank) <= min(len(drug_ids), len(cell_ids)):
        raise ValueError("Response-profile rank is incompatible with the matrix")
    drug_lookup = {entity: index for index, entity in enumerate(drug_ids)}
    cell_lookup = {entity: index for index, entity in enumerate(cell_ids)}
    drug_code = (
        frame["master_drug_id"].astype(str).map(drug_lookup).to_numpy(np.int64)
    )
    cell_code = (
        frame["cellosaurus_id"].astype(str).map(cell_lookup).to_numpy(np.int64)
    )

    matrix = np.zeros((len(drug_ids), len(cell_ids)), dtype=np.float64)
    observed = np.zeros_like(matrix, dtype=bool)
    matrix[drug_code, cell_code] = effects.residual
    observed[drug_code, cell_code] = True
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    scale = np.sqrt(singular_values[:rank])
    drug_factors = left[:, :rank] * scale
    cell_factors = right[:rank].T * scale

    for component in range(rank):
        pivot = int(np.argmax(np.abs(drug_factors[:, component])))
        if drug_factors[pivot, component] < 0:
            drug_factors[:, component] *= -1
            cell_factors[:, component] *= -1

    reconstruction = drug_factors @ cell_factors.T
    residual = matrix[observed] - reconstruction[observed]
    observed_mse = float(np.mean(np.square(matrix[observed])))
    reconstruction_mse = float(np.mean(np.square(residual)))
    return ResponseProfileFactors(
        drug_ids=drug_ids,
        cell_ids=cell_ids,
        drug_factors=drug_factors,
        cell_factors=cell_factors,
        singular_values=singular_values[:rank],
        observed_rmse=float(np.sqrt(reconstruction_mse)),
        explained_observed_variance=(
            float(1.0 - reconstruction_mse / observed_mse)
            if observed_mse > 0
            else 0.0
        ),
    )


def _entity_factor_frame(ids, factors, key):
    return pd.DataFrame(
        {
            key: ids,
            **{
                f"factor_{index}": factors[:, index]
                for index in range(factors.shape[1])
            },
        }
    )


def prepare_response_profile_priors(prepared, effects, config, outdir):
    rank = int(config["rank"])
    beta = float(config["beta"])
    cell_main_weight = float(config["cell_main_weight"])
    cell_ridge_alpha = float(config["cell_ridge_alpha"])
    factors = fit_response_profile_factors(prepared.train.frame, effects, rank)

    drug_factor_map = dict(zip(factors.drug_ids, factors.drug_factors))
    cell_effect = effects.cell.set_index("cellosaurus_id")["effect"]
    cell_target = np.column_stack(
        (
            cell_effect.loc[factors.cell_ids].to_numpy(np.float64),
            factors.cell_factors,
        )
    )
    cell_features = build_entity_features(
        prepared.train, "cell", factors.cell_ids
    )
    cell_externalizer = fit_ridge_externalizer(
        cell_features,
        cell_target,
        cell_ridge_alpha,
    )

    drug_features = build_entity_features(
        prepared.train, "drug", factors.drug_ids
    )
    k = int(config["drug_knn"]["k"])
    gamma = float(config["drug_knn"]["gamma"])
    tau = float(config["drug_knn"]["tau"])
    components = {}
    cell_rows = []
    neighbor_tables = []
    for split, dataset in (
        ("train", prepared.train),
        ("valid", prepared.valid),
        ("test", prepared.test),
    ):
        frame = dataset.frame
        unique_drugs = np.asarray(
            sorted(frame["master_drug_id"].astype(str).unique())
        )
        split_drug_factors = dict(drug_factor_map)
        unseen_drugs = np.asarray(
            [drug for drug in unique_drugs if drug not in split_drug_factors],
            dtype=str,
        )
        if len(unseen_drugs):
            query = build_entity_features(dataset, "drug", unseen_drugs)
            prediction, neighbors = predict_tanimoto_knn(
                query,
                unseen_drugs,
                drug_features,
                factors.drug_ids,
                factors.drug_factors,
                k=k,
                gamma=gamma,
                tau=tau,
            )
            split_drug_factors.update(dict(zip(unseen_drugs, prediction)))
            neighbors.insert(0, "split", split)
            neighbor_tables.append(neighbors)

        unique_cells = np.asarray(
            sorted(frame["cellosaurus_id"].astype(str).unique())
        )
        split_cell_output = dict(zip(factors.cell_ids, cell_target))
        unseen_cells = np.asarray(
            [cell for cell in unique_cells if cell not in split_cell_output],
            dtype=str,
        )
        if len(unseen_cells):
            query = build_entity_features(dataset, "cell", unseen_cells)
            prediction = cell_externalizer.predict(query)
            split_cell_output.update(dict(zip(unseen_cells, prediction)))

        pair_drug_factor = np.stack(
            [
                split_drug_factors[drug]
                for drug in frame["master_drug_id"].astype(str)
            ]
        )
        pair_cell_output = np.stack(
            [
                split_cell_output[cell]
                for cell in frame["cellosaurus_id"].astype(str)
            ]
        )
        cell_main = cell_main_weight * pair_cell_output[:, 0]
        profile = beta * np.sum(
            pair_drug_factor * pair_cell_output[:, 1:],
            axis=1,
        )
        components[split] = {
            "cell_main_effect": cell_main.astype(np.float32),
            "response_profile_effect": profile.astype(np.float32),
        }
        for cell in unique_cells:
            output = split_cell_output[cell]
            cell_rows.append(
                {
                    "split": split,
                    "cellosaurus_id": cell,
                    "cell_main_effect": float(output[0]),
                    "source": (
                        "seen_anova_profile"
                        if cell in cell_effect.index
                        else "unseen_context_viper_ridge"
                    ),
                }
            )

    drug_frame = _entity_factor_frame(
        factors.drug_ids,
        factors.drug_factors,
        "master_drug_id",
    )
    cell_frame = _entity_factor_frame(
        factors.cell_ids,
        factors.cell_factors,
        "cellosaurus_id",
    )
    drug_frame.to_csv(outdir / "training_drug_response_factors.csv", index=False)
    cell_frame.to_csv(outdir / "training_cell_response_factors.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(outdir / "cell_main_priors.csv", index=False)
    neighbors = (
        pd.concat(neighbor_tables, ignore_index=True)
        if neighbor_tables
        else pd.DataFrame(
            columns=[
                "split",
                "query_id",
                "neighbor_id",
                "rank",
                "similarity",
                "weight_before_normalization",
                "confidence",
            ]
        )
    )
    neighbors.to_csv(outdir / "drug_profile_neighbors.csv", index=False)

    scaler = cell_externalizer.named_steps["standardscaler"]
    ridge = cell_externalizer.named_steps["ridge"]
    np.savez_compressed(
        outdir / "response_profile_externalizers.npz",
        drug_ids=factors.drug_ids,
        drug_factors=factors.drug_factors.astype(np.float32),
        drug_fingerprints=drug_features.astype(np.float32),
        cell_ids=factors.cell_ids,
        cell_outputs=cell_target.astype(np.float32),
        cell_feature_mean=scaler.mean_.astype(np.float32),
        cell_feature_scale=scaler.scale_.astype(np.float32),
        cell_coefficient=np.asarray(ridge.coef_, dtype=np.float32),
        cell_intercept=np.asarray(ridge.intercept_, dtype=np.float32),
    )
    state = {
        "rank": rank,
        "beta": beta,
        "cell_main_weight": cell_main_weight,
        "singular_values": factors.singular_values.astype(np.float32),
        "drug_knn": {"k": k, "gamma": gamma, "tau": tau},
        "drug_ids": factors.drug_ids,
        "drug_factors": factors.drug_factors.astype(np.float32),
        "drug_fingerprints": drug_features.astype(np.float32),
        "cell_ids": factors.cell_ids,
        "cell_outputs": cell_target.astype(np.float32),
        "cell_feature_mean": scaler.mean_.astype(np.float32),
        "cell_feature_scale": scaler.scale_.astype(np.float32),
        "cell_coefficient": np.asarray(ridge.coef_, dtype=np.float32),
        "cell_intercept": np.asarray(ridge.intercept_, dtype=np.float32),
    }
    manifest = {
        "method": "zero-filled SVD of training-only ANOVA residual",
        "rank": rank,
        "beta": beta,
        "cell_main_weight": cell_main_weight,
        "singular_values": factors.singular_values,
        "training_observed_rmse": factors.observed_rmse,
        "training_explained_observed_variance": (
            factors.explained_observed_variance
        ),
        "seen_drug_rule": "training response-factor lookup",
        "unseen_drug_rule": "raw Morgan Tanimoto weighted factor average",
        "drug_knn": {"k": k, "gamma": gamma, "tau": tau},
        "seen_cell_rule": "training ANOVA cell effect and response-factor lookup",
        "unseen_cell_rule": "Context+VIPER multi-output Ridge",
        "cell_ridge_alpha": cell_ridge_alpha,
    }
    write_json(outdir / "response_profile_manifest.json", manifest)
    return components, state, manifest
