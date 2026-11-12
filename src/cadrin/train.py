"""Train one CaDRiN evaluation scenario from scratch."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .chemical_groups import load_scaffold_groups
from .config import TRAINING_SCENARIOS as SCENARIOS, load_training_config as load_config
from .training_data import prepare_data
from .loss import CaDRiNLoss
from .main_effects import build_entity_features, fit_two_way_effects
from .model import build_model
from .training_mtr_externalizer import (
    externalizer_state,
    fit_morgan_mtr_externalizer,
    load_mtr_features,
    save_externalizer,
)
from .random_pair_correction import (
    apply_multiresolution_correction,
    fit_nonnegative_fusion,
    fit_observed_profile_teacher,
    observed_profile_for_pairs,
)
from .response_profiles import prepare_response_profile_priors
from .training_utils import resolve_device, seed_everything
from .utils import write_json


def _correlation(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def regression_metrics(y_true, y_pred, drug_ids):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = y_pred - y_true
    total_variance = float(np.sum(np.square(y_true - y_true.mean())))
    frame = pd.DataFrame(
        {"true": y_true, "pred": y_pred, "drug": drug_ids}
    )
    within_drug = [
        value
        for _, group in frame.groupby("drug")
        if len(group) >= 5
        for value in [_correlation(group["true"], group["pred"])]
        if np.isfinite(value)
    ]
    means = frame.groupby("drug")[["true", "pred"]].mean()
    drug_mean_error = means["pred"].to_numpy() - means["true"].to_numpy()
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "r2": (
            float(1.0 - np.sum(np.square(error)) / total_variance)
            if total_variance > 0
            else float("nan")
        ),
        "pearson": _correlation(y_true, y_pred),
        "spearman": _correlation(
            pd.Series(y_true).rank(method="average"),
            pd.Series(y_pred).rank(method="average"),
        ),
        "within_drug_pearson_median": (
            float(np.median(within_drug)) if within_drug else float("nan")
        ),
        "within_drug_groups": int(len(within_drug)),
        "drug_mean_rmse": float(
            np.sqrt(np.mean(np.square(drug_mean_error)))
        ),
        "drug_mean_pearson": _correlation(means["true"], means["pred"]),
    }


class FixedPriorDataset(Dataset):
    """Dataset with training-derived additive priors."""

    def __init__(
        self,
        base,
        drug_main_effect,
        cell_main_effect=None,
        response_profile_effect=None,
    ):
        self.base = base
        self.frame = base.frame
        self.base_drug_main_effect = torch.as_tensor(
            np.asarray(drug_main_effect, dtype=np.float32)
        )
        self.cell_main_effect = torch.as_tensor(
            np.zeros(len(base), dtype=np.float32)
            if cell_main_effect is None
            else np.asarray(cell_main_effect, dtype=np.float32)
        )
        self.response_profile_effect = torch.as_tensor(
            np.zeros(len(base), dtype=np.float32)
            if response_profile_effect is None
            else np.asarray(response_profile_effect, dtype=np.float32)
        )
        if not (
            len(self.base_drug_main_effect)
            == len(self.cell_main_effect)
            == len(self.response_profile_effect)
            == len(base)
        ):
            raise ValueError("Fixed priors are not aligned to the pair dataset")
        self.fixed_additive_effect = (
            self.base_drug_main_effect
            + self.cell_main_effect
            + self.response_profile_effect
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        fingerprint, context, viper, drug_idx, _, _, response = self.base[index]
        return (
            fingerprint,
            context,
            viper,
            drug_idx,
            self.fixed_additive_effect[index],
            response,
        )


def make_loader(dataset, batch_size, shuffle, seed, num_workers):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def move_batch(batch, device):
    fingerprint, context, viper, drug_idx, fixed_prior, response = batch
    return (
        fingerprint.to(device),
        context.to(device),
        viper.to(device),
        drug_idx.to(device),
        fixed_prior.to(device).reshape(-1, 1),
        response.to(device).reshape(-1, 1),
    )


def prepare_drug_main_effects(prepared, config, outdir, scenario):
    """Fit ANOVA lookup and the training-only unseen-drug externalizer."""

    effects = fit_two_way_effects(prepared.train.frame)
    if not effects.converged:
        raise RuntimeError("Training-only two-way ANOVA did not converge")
    maximum_marginal = max(
        effects.maximum_drug_residual_mean,
        effects.maximum_cell_residual_mean,
        abs(effects.overall_residual_mean),
    )
    if maximum_marginal > 1e-8:
        raise RuntimeError("Training-only two-way ANOVA is not orthogonal")

    drug_ids = effects.drug["master_drug_id"].astype(str).to_numpy()
    target = effects.drug["effect"].to_numpy(np.float64)
    morgan = build_entity_features(prepared.train, "drug", drug_ids)
    mtr_table, mtr_audit = load_mtr_features(
        config["mtr_features"],
        expected_width=int(config.get("mtr_width", 384)),
    )
    if config.get("mtr_audit"):
        mtr_audit_path = Path(config["mtr_audit"]).expanduser().resolve()
        encoder_audit = json.loads(mtr_audit_path.read_text(encoding="utf-8"))
        expected_shape = encoder_audit.get("outputs", {}).get("shape")
        if expected_shape != [mtr_audit["rows"], mtr_audit["width"]]:
            raise ValueError("MTR feature shape does not match its encoder audit")
        mtr_audit["encoder_key"] = encoder_audit["encoder"]["encoder_key"]
        mtr_audit["model_files_tree_sha256"] = encoder_audit["encoder"][
            "model_files_tree_sha256"
        ]
    missing = pd.Index(drug_ids).difference(mtr_table.index)
    if len(missing):
        raise ValueError(
            "MTR feature table is missing training drugs: "
            f"{missing[:5].tolist()}"
        )

    cv_groups = None
    if scenario in {"cold_scaffold", "cold_scaffold_cell"}:
        scaffold_groups, _ = load_scaffold_groups(config["mtr_features"])
        cv_groups = scaffold_groups.loc[drug_ids].to_numpy()

    externalizer, oof, oof_neighbors, cv_audit = (
        fit_morgan_mtr_externalizer(
            drug_ids,
            morgan,
            mtr_table.loc[drug_ids].to_numpy(np.float64),
            target,
            k=int(config["k"]),
            gamma=float(config["gamma"]),
            tau=float(config["tau"]),
            outer_folds=int(config["outer_folds"]),
            seed=int(config["cv_seed"]),
            extra_trees=config["extra_trees"],
            groups=cv_groups,
        )
    )
    oof.to_csv(outdir / "drug_externalizer_oof.csv", index=False)
    oof_neighbors.to_csv(
        outdir / "drug_externalizer_oof_neighbors.csv", index=False
    )
    write_json(
        outdir / "drug_externalizer_audit.json",
        {**cv_audit, "mtr_features": mtr_audit},
    )

    seen = dict(zip(drug_ids, target))
    datasets = {}
    prior_rows = []
    neighbor_tables = []
    component_tables = []
    routing = {}
    for split, base in (
        ("train", prepared.train),
        ("valid", prepared.valid),
        ("test", prepared.test),
    ):
        unique_ids = np.asarray(
            sorted(base.frame["master_drug_id"].astype(str).unique())
        )
        unseen_ids = np.asarray(
            [drug for drug in unique_ids if drug not in seen], dtype=str
        )
        prior = dict(seen)
        nearest_similarity = {}
        if len(unseen_ids):
            query_morgan = build_entity_features(base, "drug", unseen_ids)
            missing = pd.Index(unseen_ids).difference(mtr_table.index)
            if len(missing):
                raise ValueError(
                    f"MTR feature table is missing {split} drugs: "
                    f"{missing[:5].tolist()}"
                )
            prediction, neighbors, components = externalizer.predict(
                unseen_ids,
                query_morgan,
                mtr_table.loc[unseen_ids].to_numpy(np.float64),
            )
            neighbors.insert(0, "split", split)
            neighbor_tables.append(neighbors)
            component_tables.append(
                pd.DataFrame(
                    {
                        "split": split,
                        "master_drug_id": unseen_ids,
                        "morgan_knn_prediction": components[:, 0],
                        "mtr_extra_trees_prediction": components[:, 1],
                        "stack_prediction": prediction,
                    }
                )
            )
            nearest_similarity = (
                neighbors[neighbors["rank"].eq(1)]
                .set_index("query_id")["similarity"]
                .to_dict()
            )
            prior.update(dict(zip(unseen_ids, prediction)))

        if any(
            prior[drug] != seen[drug]
            for drug in unique_ids
            if drug in seen
        ):
            raise RuntimeError("Seen-drug ANOVA lookup was modified")
        values = base.frame["master_drug_id"].astype(str).map(prior)
        if values.isna().any():
            raise RuntimeError(f"Drug-main alignment failed for {split}")
        datasets[split] = FixedPriorDataset(base, values.to_numpy())
        routing[split] = {
            "seen_drugs": int(len(unique_ids) - len(unseen_ids)),
            "unseen_drugs": int(len(unseen_ids)),
        }
        for drug in unique_ids:
            prior_rows.append(
                {
                    "split": split,
                    "master_drug_id": drug,
                    "drug_main_effect": float(prior[drug]),
                    "source": (
                        "seen_training_anova"
                        if drug in seen
                        else "unseen_morgan_mtr_stack"
                    ),
                    "maximum_tanimoto": (
                        None
                        if drug in seen
                        else float(nearest_similarity[drug])
                    ),
                }
            )

    effects.drug.to_csv(outdir / "training_drug_effects.csv", index=False)
    effects.cell.to_csv(outdir / "training_cell_effects.csv", index=False)
    pd.DataFrame(prior_rows).to_csv(outdir / "drug_main_priors.csv", index=False)
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
    neighbors.to_csv(outdir / "drug_neighbors.csv", index=False)
    components = (
        pd.concat(component_tables, ignore_index=True)
        if component_tables
        else pd.DataFrame(
            columns=[
                "split",
                "master_drug_id",
                "morgan_knn_prediction",
                "mtr_extra_trees_prediction",
                "stack_prediction",
            ]
        )
    )
    components.to_csv(outdir / "drug_externalizer_components.csv", index=False)

    state = externalizer_state(externalizer)
    state["mtr_features"] = mtr_audit
    state["cross_validation"] = cv_audit
    save_externalizer(outdir / "drug_externalizer.npz", externalizer, state)
    manifest = {
        "decomposition": "training-only unregularized two-way ANOVA",
        "global_mean": effects.global_mean,
        "iterations": effects.iterations,
        "maximum_marginal_residual_mean": maximum_marginal,
        "seen_drug_rule": "exact training ANOVA lookup",
        "unseen_drug_rule": (
            "training-OOF nonnegative stack of Morgan Tanimoto 3-NN "
            "and MTR ExtraTrees"
        ),
        "externalizer": {
            "cross_validation": cv_audit,
            "mtr_features": mtr_audit,
        },
        "routing": routing,
        "selected_by": config["selected_by"],
        "entities": {
            "training_drugs": int(len(drug_ids)),
            "training_cells": int(len(effects.cell)),
        },
    }
    write_json(outdir / "main_effect_manifest.json", manifest)
    return datasets, effects.global_mean, state, manifest, effects


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    totals = {}
    rows = 0
    truths, predictions, drug_ids = [], [], []
    for batch in loader:
        (
            fingerprint,
            context,
            viper,
            drug_idx,
            fixed_prior,
            response,
        ) = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            fingerprint, context, viper, fixed_prior
        )
        losses = criterion(outputs, fingerprint, context, response)
        losses["objective"].backward()
        optimizer.step()
        batch_rows = int(response.shape[0])
        rows += batch_rows
        for name, value in losses.items():
            totals[name] = (
                totals.get(name, 0.0) + float(value.detach()) * batch_rows
            )
        truths.append(response.detach().cpu().numpy().reshape(-1))
        predictions.append(outputs[0].detach().cpu().numpy().reshape(-1))
        drug_ids.append(drug_idx.detach().cpu().numpy().reshape(-1))
    metrics = regression_metrics(
        np.concatenate(truths),
        np.concatenate(predictions),
        np.concatenate(drug_ids),
    )
    losses = {name: value / rows for name, value in totals.items()}
    return losses, metrics


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    totals, rows = {}, 0
    payload = {
        name: []
        for name in (
            "true",
            "pred",
            "drug_idx",
            "fixed_additive_effect",
            "conditional_response",
        )
    }
    for batch in loader:
        (
            fingerprint,
            context,
            viper,
            drug_idx,
            fixed_prior,
            response,
        ) = move_batch(batch, device)
        outputs = model(
            fingerprint, context, viper, fixed_prior
        )
        losses = criterion(outputs, fingerprint, context, response)
        batch_rows = int(response.shape[0])
        rows += batch_rows
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch_rows
        values = {
            "true": response,
            "pred": outputs[0],
            "drug_idx": drug_idx,
            "fixed_additive_effect": outputs[1],
            "conditional_response": outputs[2],
        }
        for name, value in values.items():
            payload[name].append(value.detach().cpu().numpy().reshape(-1))
    payload = {
        name: np.concatenate(parts) for name, parts in payload.items()
    }
    losses = {name: value / rows for name, value in totals.items()}
    metrics = regression_metrics(
        payload["true"], payload["pred"], payload["drug_idx"]
    )
    return losses, metrics, payload


def save_predictions(path, dataset, payload, global_mean):
    frame = dataset.frame[
        [
            "cellosaurus_id",
            "master_drug_id",
            "source_dataset",
            "response_value",
        ]
    ].copy()
    frame["prediction"] = payload["pred"]
    frame["pred_global_mean"] = float(global_mean)
    frame["pred_drug_main_effect"] = dataset.base_drug_main_effect.numpy()
    frame["pred_cell_main_effect"] = dataset.cell_main_effect.numpy()
    frame["pred_response_profile_effect"] = (
        dataset.response_profile_effect.numpy()
    )
    frame["pred_fixed_additive_effect"] = payload[
        "fixed_additive_effect"
    ]
    frame["pred_conditional_response"] = payload["conditional_response"]
    frame["squared_error"] = np.square(
        payload["pred"] - payload["true"]
    )
    frame["absolute_error"] = np.abs(
        payload["pred"] - payload["true"]
    )
    frame.to_csv(path, index=False)


def random_pair_components(dataset, payload, global_mean, teacher):
    observed_profile, visible = observed_profile_for_pairs(
        dataset.frame, teacher
    )
    svd_profile = dataset.response_profile_effect.numpy().astype(np.float64)
    main_effect = (
        float(global_mean)
        + dataset.base_drug_main_effect.numpy().astype(np.float64)
        + dataset.cell_main_effect.numpy().astype(np.float64)
    )
    return {
        "main_effect": main_effect,
        "conditional": payload["conditional_response"].astype(np.float64),
        "svd_profile": svd_profile,
        "observed_profile": observed_profile,
        "visible": visible,
    }


def apply_random_pair_correction(payload, components, weights):
    corrected = {name: value.copy() for name, value in payload.items()}
    corrected["base_pred"] = payload["pred"].copy()
    corrected["pred"] = apply_multiresolution_correction(
        payload["pred"],
        components["main_effect"],
        components["conditional"],
        components["svd_profile"],
        components["observed_profile"],
        components["visible"],
        weights,
    )
    return corrected


def save_corrected_predictions(
    path,
    dataset,
    payload,
    components,
    global_mean,
):
    frame = dataset.frame[
        [
            "cellosaurus_id",
            "master_drug_id",
            "source_dataset",
            "response_value",
        ]
    ].copy()
    frame["prediction"] = payload["pred"]
    frame["prediction_base"] = payload["base_pred"]
    frame["pred_global_mean"] = float(global_mean)
    frame["pred_drug_main_effect"] = dataset.base_drug_main_effect.numpy()
    frame["pred_cell_main_effect"] = dataset.cell_main_effect.numpy()
    frame["pred_main_effect"] = components["main_effect"]
    frame["pred_conditional_response"] = components["conditional"]
    frame["pred_svd_profile_effect"] = components["svd_profile"]
    frame["pred_observed_profile_effect"] = components["observed_profile"]
    frame["pred_observed_profile_correction"] = (
        components["observed_profile"] - components["svd_profile"]
    )
    frame["both_entities_seen"] = components["visible"]
    frame["squared_error"] = np.square(
        payload["pred"] - payload["true"]
    )
    frame["absolute_error"] = np.abs(
        payload["pred"] - payload["true"]
    )
    frame.to_csv(path, index=False)


def corrected_metrics(dataset, payload, base_metrics):
    metrics = regression_metrics(
        payload["true"],
        payload["pred"],
        dataset.frame["master_drug_id"].astype(str).to_numpy(),
    )
    metrics["base_rmse"] = base_metrics["rmse"]
    metrics["base_pearson"] = base_metrics["pearson"]
    metrics["delta_rmse"] = metrics["rmse"] - base_metrics["rmse"]
    metrics["delta_pearson"] = (
        metrics["pearson"] - base_metrics["pearson"]
    )
    return metrics


def train_from_config(
    config_path,
    *,
    scenario,
    outdir,
    training_seed=3078,
    split_seed=None,
    device="auto",
    epochs=None,
):
    """Fit every composite component and return the complete bundle directory.

    Inputs and fixed splits are specified by the training JSON. ``outdir`` is
    explicit and must be new or empty so previous runs are preserved.
    Cold scenarios include validation-only profile calibration. ``epochs``
    can shorten a development run without changing the template configuration.
    """

    if scenario not in SCENARIOS:
        raise ValueError(f"Unsupported training scenario: {scenario}")
    args = argparse.Namespace(
        config=config_path,
        scenario=scenario,
        outdir=Path(outdir),
        training_seed=int(training_seed),
        split_seed=int(training_seed if split_seed is None else split_seed),
        device=device,
        epochs=epochs,
        maximum_rows=None,
    )
    _train(args)
    if scenario != "random_pair":
        from .calibration import calibrate_response_profile

        calibrate_response_profile(args.outdir)
    from .export import export_bundle

    return export_bundle(args.outdir, args.outdir / "bundle")


def _train(args):
    config = load_config(args.config)
    training = dict(config["training"])
    if args.epochs is not None:
        training["epochs"] = args.epochs
    config["training"] = training
    experiment = config["experiment"]
    outdir = Path(args.outdir).expanduser().resolve()
    if outdir.exists() and (not outdir.is_dir() or any(outdir.iterdir())):
        raise FileExistsError(
            f"Training output directory must be new or empty: {outdir}"
        )
    seed_everything(args.training_seed)
    device = resolve_device(args.device)

    prepared = prepare_data(
        config,
        args.scenario,
        outdir,
        maximum_rows=args.maximum_rows,
        seed=args.training_seed,
        split_seed=args.split_seed,
    )
    prepared.train.frame[["master_drug_id", "cellosaurus_id"]].to_csv(
        outdir / "training_pairs.csv.gz", index=False
    )
    (
        datasets,
        global_mean,
        drug_externalizer_state,
        main_effect_manifest,
        effects,
    ) = prepare_drug_main_effects(
        prepared,
        config["main_effect"],
        outdir,
        args.scenario,
    )
    components, response_profile_state, response_profile_manifest = (
        prepare_response_profile_priors(
            prepared,
            effects,
            config["response_profile"],
            outdir,
        )
    )
    datasets = {
        split: FixedPriorDataset(
            dataset.base,
            dataset.base_drug_main_effect.numpy(),
            components[split]["cell_main_effect"],
            components[split]["response_profile_effect"],
        )
        for split, dataset in datasets.items()
    }

    train_loader = make_loader(
        datasets["train"],
        training["batch_size"],
        True,
        args.training_seed,
        training["num_workers"],
    )
    valid_loader = make_loader(
        datasets["valid"],
        training["evaluation_batch_size"],
        False,
        args.training_seed,
        training["num_workers"],
    )
    test_loader = make_loader(
        datasets["test"],
        training["evaluation_batch_size"],
        False,
        args.training_seed,
        training["num_workers"],
    )

    model = build_model(
        config["model"], prepared.dimensions, global_mean
    ).to(device)
    criterion = CaDRiNLoss(config["loss"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=training["lr_patience"],
        min_lr=training["min_learning_rate"],
        threshold=training["min_delta"],
        threshold_mode="abs",
    )

    best_rmse, best_epoch, best_state, stale_epochs = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, training["epochs"] + 1):
        started = time.time()
        train_losses, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        valid_losses, valid_metrics, _ = evaluate_model(
            model, valid_loader, criterion, device
        )
        scheduler.step(valid_metrics["rmse"])
        improved = (
            valid_metrics["rmse"] < best_rmse - training["min_delta"]
        )
        if improved:
            best_rmse = float(valid_metrics["rmse"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
            torch.save(
                {
                    "model": experiment["model_name"],
                    "scenario": args.scenario,
                    "training_seed": args.training_seed,
                    "split_seed": args.split_seed,
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "model_config": config["model"],
                    "dimensions": prepared.dimensions,
                    "global_mean": global_mean,
                    "drug_externalizer_state": drug_externalizer_state,
                    "response_profile_state": response_profile_state,
                },
                outdir / "checkpoint.pt",
            )
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - started,
                **{
                    f"train_{key}": value
                    for key, value in train_losses.items()
                },
                "train_rmse": train_metrics["rmse"],
                "train_pearson": train_metrics["pearson"],
                **{
                    f"valid_{key}": value
                    for key, value in valid_losses.items()
                },
                "valid_rmse": valid_metrics["rmse"],
                "valid_pearson": valid_metrics["pearson"],
            }
        )
        print(
            f"scenario={args.scenario} epoch={epoch:03d} "
            f"valid_rmse={valid_metrics['rmse']:.6f} "
            f"valid_pearson={valid_metrics['pearson']:.6f}",
            flush=True,
        )
        if stale_epochs >= training["patience"]:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    valid_losses, valid_metrics, valid_payload = evaluate_model(
        model, valid_loader, criterion, device
    )
    test_losses, test_metrics, test_payload = evaluate_model(
        model, test_loader, criterion, device
    )
    model_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    for metrics, losses in (
        (valid_metrics, valid_losses),
        (test_metrics, test_losses),
    ):
        metrics["best_epoch"] = best_epoch
        metrics["selection_metric"] = "validation_rmse"
        metrics["model_parameters"] = model_parameters
        metrics["losses"] = losses

    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)
    correction_manifest = None
    if args.scenario == "random_pair":
        write_json(outdir / "base_validation_metrics.json", valid_metrics)
        write_json(outdir / "base_test_metrics.json", test_metrics)
        save_predictions(
            outdir / "base_validation_predictions.csv.gz",
            datasets["valid"],
            valid_payload,
            global_mean,
        )
        save_predictions(
            outdir / "base_test_predictions.csv.gz",
            datasets["test"],
            test_payload,
            global_mean,
        )

        correction_config = config["random_pair_correction"]
        teacher = fit_observed_profile_teacher(
            prepared.train.frame,
            effects,
            correction_config["teacher"],
            seed=args.training_seed,
            device=device,
        )
        teacher.selection.to_csv(
            outdir / "observed_profile_selection.csv", index=False
        )
        teacher.history.to_csv(
            outdir / "observed_profile_history.csv", index=False
        )
        valid_components = random_pair_components(
            datasets["valid"], valid_payload, global_mean, teacher
        )
        test_components = random_pair_components(
            datasets["test"], test_payload, global_mean, teacher
        )
        weights, residual_norm = fit_nonnegative_fusion(
            valid_payload["true"],
            valid_components["main_effect"],
            valid_components["conditional"],
            valid_components["svd_profile"],
            valid_components["observed_profile"],
            valid_components["visible"],
        )
        corrected_valid = apply_random_pair_correction(
            valid_payload, valid_components, weights
        )
        corrected_test = apply_random_pair_correction(
            test_payload, test_components, weights
        )
        final_valid_metrics = corrected_metrics(
            datasets["valid"], corrected_valid, valid_metrics
        )
        final_test_metrics = corrected_metrics(
            datasets["test"], corrected_test, test_metrics
        )
        weight_record = {
            "conditional_response": float(weights[0]),
            "rank8_svd_profile": float(weights[1]),
            "observed_profile_correction": float(weights[2]),
        }
        for metrics, components in (
            (final_valid_metrics, valid_components),
            (final_test_metrics, test_components),
        ):
            metrics.update(
                {
                    "best_epoch": best_epoch,
                    "selection_metric": (
                        "validation RMSE for neural checkpoint; validation "
                        "nonnegative least squares for response fusion"
                    ),
                    "model_parameters": model_parameters,
                    "profile_beta": float(
                        config["response_profile"]["beta"]
                    ),
                    "profile_source": "multiresolution_fusion",
                    "teacher_rank": teacher.rank,
                    "teacher_regularization": teacher.regularization,
                    "fusion_weights": weight_record,
                    "both_seen_pairs": int(components["visible"].sum()),
                }
            )
        valid_metrics, test_metrics = (
            final_valid_metrics,
            final_test_metrics,
        )
        valid_payload, test_payload = corrected_valid, corrected_test
        write_json(outdir / "validation_metrics.json", valid_metrics)
        write_json(outdir / "test_metrics.json", test_metrics)
        save_corrected_predictions(
            outdir / "validation_predictions.csv.gz",
            datasets["valid"],
            valid_payload,
            valid_components,
            global_mean,
        )
        save_corrected_predictions(
            outdir / "test_predictions.csv.gz",
            datasets["test"],
            test_payload,
            test_components,
            global_mean,
        )
        write_json(
            outdir / "fusion_weights.json",
            {
                "method": correction_config["fusion"]["method"],
                "main_effect_weight": 1.0,
                "weights": weight_record,
                "validation_residual_norm": residual_norm,
            },
        )
        correction_state = {
            "method": correction_config["method"],
            "drug_ids": teacher.drug_ids,
            "cell_ids": teacher.cell_ids,
            "drug_factors": teacher.drug_factors,
            "cell_factors": teacher.cell_factors,
            "rank": teacher.rank,
            "regularization": teacher.regularization,
            "best_epoch": teacher.best_epoch,
            "fusion_weights": weights.astype(np.float32),
            "visibility_rule": (
                "apply only when both drug and cell occur in training"
            ),
        }
        torch.save(
            {
                "model": experiment["model_name"],
                "scenario": args.scenario,
                "training_seed": args.training_seed,
                "split_seed": args.split_seed,
                "epoch": best_epoch,
                "model_state_dict": best_state,
                "model_config": config["model"],
                "dimensions": prepared.dimensions,
                "global_mean": global_mean,
                "drug_externalizer_state": drug_externalizer_state,
                "response_profile_state": response_profile_state,
                "random_pair_correction_state": correction_state,
            },
            outdir / "checkpoint.pt",
        )
        correction_manifest = {
            "method": correction_config["method"],
            "training_location": "inside train.py after neural checkpoint",
            "teacher": {
                "method": (
                    "training-only observed-entry weighted factorization of "
                    "strictly centered ANOVA residual"
                ),
                "rank": teacher.rank,
                "regularization": teacher.regularization,
                "best_epoch": teacher.best_epoch,
                "observed_rmse": teacher.observed_rmse,
                "explained_observed_variance": (
                    teacher.explained_observed_variance
                ),
                "selection": teacher.selection.to_dict("records"),
            },
            "fusion": {
                "method": correction_config["fusion"]["method"],
                "main_effect_weight": 1.0,
                "weights": weight_record,
                "selected_on": "validation only",
            },
            "visibility_rule": (
                "apply only when both drug and cell occur in training; "
                "otherwise retain the base prediction"
            ),
        }
    else:
        write_json(outdir / "validation_metrics.json", valid_metrics)
        write_json(outdir / "test_metrics.json", test_metrics)
        save_predictions(
            outdir / "validation_predictions.csv.gz",
            datasets["valid"],
            valid_payload,
            global_mean,
        )
        save_predictions(
            outdir / "test_predictions.csv.gz",
            datasets["test"],
            test_payload,
            global_mean,
        )

    selection = (
        "minimum validation RMSE for the neural checkpoint; training-only "
        "observed-entry teacher and validation nonnegative response fusion"
        if correction_manifest is not None
        else (
            "minimum validation RMSE; response-profile coefficient selected "
            "on validation after checkpoint selection"
        )
    )
    write_json(
        outdir / "run_manifest.json",
        {
            "model": experiment["model_name"],
            "scenario": args.scenario,
            "training_seed": args.training_seed,
            "split_seed": args.split_seed,
            "device": str(device),
            "config": config,
            "data": prepared.manifest,
            "main_effect": main_effect_manifest,
            "response_profile": response_profile_manifest,
            "random_pair_correction": correction_manifest,
            "best_epoch": best_epoch,
            "selection": selection,
        },
    )
    print(
        json.dumps(
            {
                "scenario": args.scenario,
                "validation": valid_metrics,
                "test": test_metrics,
            },
            indent=2,
        )
    )
