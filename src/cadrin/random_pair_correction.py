"""Multiresolution response correction."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.optimize import nnls
from torch import nn


PAIR_COLUMNS = ["master_drug_id", "cellosaurus_id"]


@dataclass
class ObservedProfileTeacher:
    drug_ids: np.ndarray
    cell_ids: np.ndarray
    drug_factors: np.ndarray
    cell_factors: np.ndarray
    rank: int
    regularization: float
    best_epoch: int
    observed_rmse: float
    explained_observed_variance: float
    selection: pd.DataFrame
    history: pd.DataFrame


class _ObservedFactorization(nn.Module):
    def __init__(self, drug_initial, cell_initial):
        super().__init__()
        self.drug_raw = nn.Parameter(
            torch.as_tensor(drug_initial, dtype=torch.float32)
        )
        self.cell_raw = nn.Parameter(
            torch.as_tensor(cell_initial, dtype=torch.float32)
        )

    def factors(self):
        return (
            self.drug_raw - self.drug_raw.mean(dim=0, keepdim=True),
            self.cell_raw - self.cell_raw.mean(dim=0, keepdim=True),
        )

    def forward(self, drug_index, cell_index):
        drug, cell = self.factors()
        return torch.sum(drug[drug_index] * cell[cell_index], dim=1)


def _pair_codes(frame, drug_ids, cell_ids):
    drug_lookup = {entity: index for index, entity in enumerate(drug_ids)}
    cell_lookup = {entity: index for index, entity in enumerate(cell_ids)}
    drug_code = (
        frame["master_drug_id"].astype(str).map(drug_lookup).to_numpy(np.int64)
    )
    cell_code = (
        frame["cellosaurus_id"].astype(str).map(cell_lookup).to_numpy(np.int64)
    )
    return drug_code, cell_code


def _initial_factors(
    residual,
    drug_code,
    cell_code,
    n_drugs,
    n_cells,
    rank,
    fit_index,
):
    matrix = np.zeros((n_drugs, n_cells), dtype=np.float64)
    fit_index = np.asarray(fit_index, dtype=np.int64)
    matrix[drug_code[fit_index], cell_code[fit_index]] = residual[fit_index]
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    scale = np.sqrt(np.maximum(singular_values[:rank], 1e-8))
    drug = left[:, :rank] * scale
    cell = right[:rank].T * scale
    return (
        drug - drug.mean(axis=0, keepdims=True),
        cell - cell.mean(axis=0, keepdims=True),
    )


def _entry_holdout(drug_code, cell_code, fraction, seed):
    rows = len(drug_code)
    desired = int(round(float(fraction) * rows))
    if desired <= 0:
        return np.arange(rows, dtype=np.int64), np.asarray([], dtype=np.int64)
    drug_count = np.bincount(drug_code).astype(np.int64)
    cell_count = np.bincount(cell_code).astype(np.int64)
    generator = np.random.default_rng(seed)
    selected = []
    for index in generator.permutation(rows):
        drug = drug_code[index]
        cell = cell_code[index]
        if drug_count[drug] <= 1 or cell_count[cell] <= 1:
            continue
        selected.append(int(index))
        drug_count[drug] -= 1
        cell_count[cell] -= 1
        if len(selected) >= desired:
            break
    holdout = np.asarray(sorted(selected), dtype=np.int64)
    fit = np.setdiff1d(np.arange(rows, dtype=np.int64), holdout)
    return fit, holdout


def _fit_factorization(
    residual,
    drug_code,
    cell_code,
    n_drugs,
    n_cells,
    *,
    rank,
    regularization,
    learning_rate,
    epochs,
    patience,
    min_delta,
    fit_index,
    score_index,
    seed,
    device,
):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    drug_initial, cell_initial = _initial_factors(
        residual,
        drug_code,
        cell_code,
        n_drugs,
        n_cells,
        rank,
        fit_index,
    )
    model = _ObservedFactorization(drug_initial, cell_initial).to(device)
    drug_tensor = torch.as_tensor(drug_code, dtype=torch.long, device=device)
    cell_tensor = torch.as_tensor(cell_code, dtype=torch.long, device=device)
    target = torch.as_tensor(residual, dtype=torch.float32, device=device)
    fit_tensor = torch.as_tensor(fit_index, dtype=torch.long, device=device)
    score_tensor = torch.as_tensor(
        score_index if len(score_index) else fit_index,
        dtype=torch.long,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    best_rmse = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(drug_tensor, cell_tensor)
        drug_factor, cell_factor = model.factors()
        fit_loss = torch.mean(
            torch.square(prediction[fit_tensor] - target[fit_tensor])
        )
        objective = fit_loss + float(regularization) * (
            torch.mean(torch.square(drug_factor))
            + torch.mean(torch.square(cell_factor))
        )
        objective.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            prediction = model(drug_tensor, cell_tensor)
            fit_rmse = torch.sqrt(
                torch.mean(
                    torch.square(
                        prediction[fit_tensor] - target[fit_tensor]
                    )
                )
            ).item()
            score_rmse = torch.sqrt(
                torch.mean(
                    torch.square(
                        prediction[score_tensor] - target[score_tensor]
                    )
                )
            ).item()
        history.append(
            {
                "epoch": epoch,
                "fit_rmse": fit_rmse,
                "score_rmse": score_rmse,
                "objective": float(objective.detach()),
            }
        )
        if score_rmse < best_rmse - float(min_delta):
            best_rmse = score_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if patience and stale >= int(patience):
            break
    if best_state is None:
        raise RuntimeError("Observed-entry factorization did not fit")
    model.load_state_dict(best_state)
    with torch.no_grad():
        drug_factor, cell_factor = model.factors()
    return (
        drug_factor.cpu().numpy().astype(np.float64),
        cell_factor.cpu().numpy().astype(np.float64),
        best_epoch,
        best_rmse,
        pd.DataFrame(history),
    )


def _canonicalize(drug_factors, cell_factors):
    drug_factors = np.asarray(drug_factors, dtype=np.float64).copy()
    cell_factors = np.asarray(cell_factors, dtype=np.float64).copy()
    for component in range(drug_factors.shape[1]):
        pivot = int(np.argmax(np.abs(drug_factors[:, component])))
        if drug_factors[pivot, component] < 0:
            drug_factors[:, component] *= -1
            cell_factors[:, component] *= -1
    return drug_factors, cell_factors


def fit_observed_profile_teacher(
    frame,
    effects,
    config,
    *,
    seed,
    device,
):
    """Select on a training-entry holdout, then refit on all training pairs."""

    if len(frame) != len(effects.residual):
        raise ValueError("ANOVA residual is not aligned to teacher pairs")
    if frame.duplicated(PAIR_COLUMNS).any():
        raise ValueError("Observed-profile training pairs must be unique")
    drug_ids = effects.drug["master_drug_id"].astype(str).to_numpy()
    cell_ids = effects.cell["cellosaurus_id"].astype(str).to_numpy()
    drug_code, cell_code = _pair_codes(frame, drug_ids, cell_ids)
    residual = np.asarray(effects.residual, dtype=np.float64)
    fit_index, holdout_index = _entry_holdout(
        drug_code,
        cell_code,
        config["holdout_fraction"],
        seed,
    )
    if not len(holdout_index):
        raise ValueError("Teacher selection requires a nonempty entry holdout")

    selection_rows = []
    histories = []
    candidate = 0
    maximum_rank = min(len(drug_ids), len(cell_ids))
    for rank_value in config["rank_grid"]:
        rank = int(rank_value)
        if rank < 1 or rank > maximum_rank:
            continue
        for regularization_value in config["regularization_grid"]:
            regularization = float(regularization_value)
            candidate += 1
            _, _, best_epoch, holdout_rmse, history = _fit_factorization(
                residual,
                drug_code,
                cell_code,
                len(drug_ids),
                len(cell_ids),
                rank=rank,
                regularization=regularization,
                learning_rate=float(config["learning_rate"]),
                epochs=int(config["selection_epochs"]),
                patience=int(config["patience"]),
                min_delta=float(config["min_delta"]),
                fit_index=fit_index,
                score_index=holdout_index,
                seed=seed + candidate,
                device=device,
            )
            selection_rows.append(
                {
                    "rank": rank,
                    "regularization": regularization,
                    "best_epoch": best_epoch,
                    "holdout_rmse": holdout_rmse,
                    "fit_pairs": int(len(fit_index)),
                    "holdout_pairs": int(len(holdout_index)),
                }
            )
            history.insert(0, "rank", rank)
            history.insert(1, "regularization", regularization)
            history["stage"] = "selection"
            histories.append(history)
    if not selection_rows:
        raise ValueError("No observed-profile candidate is compatible with data")

    selection = pd.DataFrame(selection_rows).sort_values(
        ["holdout_rmse", "rank", "regularization"],
        kind="mergesort",
    )
    selected = selection.iloc[0]
    selected_epoch = max(1, int(selected["best_epoch"]))
    all_index = np.arange(len(frame), dtype=np.int64)
    drug_factors, cell_factors, _, _, refit_history = _fit_factorization(
        residual,
        drug_code,
        cell_code,
        len(drug_ids),
        len(cell_ids),
        rank=int(selected["rank"]),
        regularization=float(selected["regularization"]),
        learning_rate=float(config["learning_rate"]),
        epochs=selected_epoch,
        patience=0,
        min_delta=0.0,
        fit_index=all_index,
        score_index=all_index,
        seed=seed + 10_000,
        device=device,
    )
    drug_factors, cell_factors = _canonicalize(
        drug_factors, cell_factors
    )
    prediction = np.sum(
        drug_factors[drug_code] * cell_factors[cell_code], axis=1
    )
    reconstruction_mse = float(np.mean(np.square(residual - prediction)))
    residual_mse = float(np.mean(np.square(residual)))
    refit_history.insert(0, "rank", int(selected["rank"]))
    refit_history.insert(
        1, "regularization", float(selected["regularization"])
    )
    refit_history["stage"] = "all_observed_refit"
    histories.append(refit_history)
    return ObservedProfileTeacher(
        drug_ids=drug_ids,
        cell_ids=cell_ids,
        drug_factors=drug_factors.astype(np.float32),
        cell_factors=cell_factors.astype(np.float32),
        rank=int(selected["rank"]),
        regularization=float(selected["regularization"]),
        best_epoch=selected_epoch,
        observed_rmse=float(np.sqrt(reconstruction_mse)),
        explained_observed_variance=(
            float(1.0 - reconstruction_mse / residual_mse)
            if residual_mse > 0
            else 0.0
        ),
        selection=selection.reset_index(drop=True),
        history=pd.concat(histories, ignore_index=True),
    )


def observed_profile_for_pairs(frame, teacher):
    """Return profile values and the training-visibility routing mask."""

    drug_lookup = {
        entity: index for index, entity in enumerate(teacher.drug_ids)
    }
    cell_lookup = {
        entity: index for index, entity in enumerate(teacher.cell_ids)
    }
    drug_index = frame["master_drug_id"].astype(str).map(drug_lookup)
    cell_index = frame["cellosaurus_id"].astype(str).map(cell_lookup)
    visible = drug_index.notna().to_numpy() & cell_index.notna().to_numpy()
    profile = np.zeros(len(frame), dtype=np.float64)
    profile[visible] = np.sum(
        teacher.drug_factors[drug_index[visible].to_numpy(np.int64)]
        * teacher.cell_factors[cell_index[visible].to_numpy(np.int64)],
        axis=1,
    )
    return profile, visible


def fusion_design(conditional, svd_profile, observed_profile):
    conditional = np.asarray(conditional, dtype=np.float64)
    svd_profile = np.asarray(svd_profile, dtype=np.float64)
    observed_profile = np.asarray(observed_profile, dtype=np.float64)
    return np.column_stack(
        (
            conditional,
            svd_profile,
            observed_profile - svd_profile,
        )
    )


def fit_nonnegative_fusion(
    response,
    main_effect,
    conditional,
    svd_profile,
    observed_profile,
    visible,
):
    """Fit three validation-only weights while keeping the main weight at one."""

    response = np.asarray(response, dtype=np.float64)
    main_effect = np.asarray(main_effect, dtype=np.float64)
    visible = np.asarray(visible, dtype=bool)
    design = fusion_design(conditional, svd_profile, observed_profile)
    if not visible.any():
        raise ValueError("Fusion fitting requires at least one both-seen pair")
    if not (
        np.isfinite(response).all()
        and np.isfinite(main_effect).all()
        and np.isfinite(design).all()
    ):
        raise ValueError("Fusion inputs must be finite")
    weights, residual_norm = nnls(
        design[visible],
        response[visible] - main_effect[visible],
    )
    return weights, float(residual_norm)


def apply_multiresolution_correction(
    base_prediction,
    main_effect,
    conditional,
    svd_profile,
    observed_profile,
    visible,
    weights,
):
    """Use the correction only when both entities occurred in training."""

    base_prediction = np.asarray(base_prediction, dtype=np.float64)
    main_effect = np.asarray(main_effect, dtype=np.float64)
    visible = np.asarray(visible, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (3,) or (weights < 0).any():
        raise ValueError("Fusion requires three nonnegative weights")
    corrected = main_effect + fusion_design(
        conditional,
        svd_profile,
        observed_profile,
    ) @ weights
    return np.where(visible, corrected, base_prediction)
