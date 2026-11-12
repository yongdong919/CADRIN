"""Inference with the fitted CaDRiN composite model."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
import zipfile

import joblib
import numpy as np
import pandas as pd
import torch

from .data import PreparedFeatures
from .model import build_model
from .mtr_externalizer import predict_tanimoto_knn
from .routing import assign_routes
from .utils import read_json, require_file


@dataclass
class FrozenPrediction:
    frame: pd.DataFrame
    drug_main_neighbors: pd.DataFrame
    drug_profile_neighbors: pd.DataFrame


def _resolve_bundle(path):
    """Resolve a directory or unpack one complete ZIP bundle temporarily."""

    path = Path(path).expanduser().resolve()
    if path.is_dir():
        return path, None
    require_file(path, "bundle")
    if not zipfile.is_zipfile(path):
        raise ValueError("Expected a complete CaDRiN bundle directory or ZIP archive")
    temporary = TemporaryDirectory(prefix="cadrin-bundle-")
    try:
        with zipfile.ZipFile(path) as archive:
            roots = []
            for member in archive.infolist():
                name = PurePosixPath(member.filename)
                if (
                    name.is_absolute() or ".." in name.parts
                    or "\\" in member.filename or ":" in member.filename
                ):
                    raise ValueError(f"Invalid bundle archive path: {member.filename}")
                if name.name == "manifest.json" and not member.is_dir():
                    roots.append(name.parent)
            if len(roots) != 1:
                raise ValueError("Bundle archive must contain exactly one manifest.json")
            archive.extractall(temporary.name)
        return Path(temporary.name) / roots[0], temporary
    except Exception:
        temporary.cleanup()
        raise


def _ridge_predict(features, state):
    mean = np.asarray(state["cell_feature_mean"], dtype=np.float64)
    scale = np.asarray(state["cell_feature_scale"], dtype=np.float64)
    coefficient = np.asarray(state["cell_coefficient"], dtype=np.float64)
    intercept = np.asarray(state["cell_intercept"], dtype=np.float64)
    standardized = (np.asarray(features, dtype=np.float64) - mean) / scale
    return standardized @ coefficient.T + intercept


def fuse_predictions(
    main,
    conditional,
    profile8,
    profile32,
    both_seen,
    weights,
    base_prediction=None,
):
    main = np.asarray(main, dtype=np.float64)
    conditional = np.asarray(conditional, dtype=np.float64)
    profile8 = np.asarray(profile8, dtype=np.float64)
    profile32 = np.asarray(profile32, dtype=np.float64)
    both_seen = np.asarray(both_seen, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (3,) or (weights < 0).any():
        raise ValueError("Multiresolution fusion requires three nonnegative weights")
    correction = np.where(both_seen, profile32 - profile8, 0.0)
    prediction = (
        main + conditional + profile8
        if base_prediction is None
        else np.asarray(base_prediction, dtype=np.float64).copy()
    )
    prediction[both_seen] = (
        main[both_seen]
        + weights[0] * conditional[both_seen]
        + weights[1] * profile8[both_seen]
        + weights[2] * correction[both_seen]
    )
    return prediction, correction


class FrozenCaDRiNPredictor:
    """Load fitted components from a directory or ZIP and predict responses."""

    def __init__(self, bundle_dir, device="cpu"):
        self.bundle_dir, self._temporary_bundle = _resolve_bundle(bundle_dir)
        self.manifest = read_json(self.bundle_dir / "manifest.json")
        self.checkpoint = torch.load(
            require_file(self.bundle_dir / "checkpoint.pt", "checkpoint"),
            map_location="cpu",
            weights_only=False,
        )
        self.scalers = np.load(
            BytesIO(require_file(self.bundle_dir / "scalers.npz", "scalers").read_bytes()),
            allow_pickle=False,
        )
        self.externalizer = joblib.load(
            require_file(
                self.bundle_dir / "drug_externalizer.joblib",
                "drug externalizer",
            )
        )
        self.device = self._resolve_device(device)
        self.model = build_model(
            self.checkpoint["model_config"],
            self.checkpoint["dimensions"],
            self.checkpoint["global_mean"],
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

        source_drugs = pd.read_csv(
            require_file(self.bundle_dir / "source_drugs.csv", "source drugs"),
            dtype=str,
        )
        required_drugs = {"master_drug_id", "structure_key"}
        if required_drugs - set(source_drugs):
            raise ValueError("source_drugs.csv has an invalid schema")
        if source_drugs[list(required_drugs)].isna().any().any():
            raise ValueError("Source drug identity must be complete")
        if source_drugs["structure_key"].duplicated().any():
            raise ValueError("Source structure keys must be unique")
        self.source_by_structure = source_drugs.set_index("structure_key")[
            "master_drug_id"
        ].astype(str).to_dict()

        source_pairs = pd.read_csv(
            require_file(self.bundle_dir / "source_pairs.csv.gz", "source pairs"),
            dtype=str,
        )
        pair_columns = ["master_drug_id", "cellosaurus_id"]
        if set(pair_columns) - set(source_pairs):
            raise ValueError("source_pairs.csv.gz has an invalid schema")
        if source_pairs.duplicated(pair_columns).any():
            raise ValueError("Source training pairs must be unique")
        self.source_pair_set = set(
            source_pairs[pair_columns].itertuples(index=False, name=None)
        )

        drug_effects = pd.read_csv(
            require_file(
                self.bundle_dir / "training_drug_effects.csv",
                "training drug effects",
            ),
            dtype={"master_drug_id": str},
        )
        if drug_effects["master_drug_id"].duplicated().any():
            raise ValueError("Training drug effects must be unique")
        self.drug_effects = drug_effects.set_index("master_drug_id")["effect"]

    @classmethod
    def load(cls, bundle_dir, device="cpu"):
        return cls(bundle_dir, device=device)

    @staticmethod
    def _resolve_device(device):
        if str(device) == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(device) == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device(str(device))

    def _profile_beta(self, routes):
        locked = self.manifest.get("locked_parameters", {}).get("profile_beta")
        if isinstance(locked, dict):
            return np.asarray([float(locked[route]) for route in routes])
        if locked is not None:
            return np.full(len(routes), float(locked), dtype=np.float64)
        locked = self.checkpoint.get("locked_profile_beta")
        if isinstance(locked, dict):
            return np.asarray([float(locked[route]) for route in routes])
        value = float(self.checkpoint["response_profile_state"].get("beta", 1.0))
        return np.full(len(routes), value, dtype=np.float64)

    @torch.no_grad()
    def _conditional(
        self,
        prepared,
        pair_drug_index,
        pair_cell_index,
        fixed_prior,
        batch_size,
    ):
        result = np.empty(len(pair_drug_index), dtype=np.float32)
        for start in range(0, len(result), int(batch_size)):
            stop = min(start + int(batch_size), len(result))
            drug = pair_drug_index[start:stop]
            cell = pair_cell_index[start:stop]
            outputs = self.model(
                torch.as_tensor(
                    prepared.fingerprint[drug], dtype=torch.float32, device=self.device
                ),
                torch.as_tensor(
                    prepared.context[cell], dtype=torch.float32, device=self.device
                ),
                torch.as_tensor(
                    prepared.viper[cell], dtype=torch.float32, device=self.device
                ),
                torch.as_tensor(
                    fixed_prior[start:stop, None],
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
            result[start:stop] = outputs[2].detach().cpu().numpy().reshape(-1)
        return result.astype(np.float64)

    def predict(self, prepared: PreparedFeatures, batch_size=4096):
        pairs = prepared.pairs.copy().reset_index(drop=True)
        unique_drugs = prepared.drug_ids
        unique_cells = prepared.cell_ids

        mapped = pd.Series(prepared.structure_keys, index=unique_drugs).map(
            self.source_by_structure
        )
        drug_seen = mapped.notna().to_numpy()
        mapped_source_ids = mapped.fillna("").astype(str).to_numpy()

        profile_state = self.checkpoint["response_profile_state"]
        source_cell_ids = np.asarray(profile_state["cell_ids"]).astype(str)
        source_cell_output = {
            entity: np.asarray(value, dtype=np.float64)
            for entity, value in zip(source_cell_ids, profile_state["cell_outputs"])
        }
        cell_seen = np.asarray(
            [entity in source_cell_output for entity in unique_cells], dtype=bool
        )

        drug_main = np.empty(len(unique_drugs), dtype=np.float64)
        morgan_main = np.empty(len(unique_drugs), dtype=np.float64)
        mtr_main = np.empty(len(unique_drugs), dtype=np.float64)
        maximum_tanimoto = np.full(len(unique_drugs), np.nan, dtype=np.float64)
        if drug_seen.any():
            values = self.drug_effects.loc[
                mapped_source_ids[drug_seen]
            ].to_numpy(np.float64)
            drug_main[drug_seen] = values
            morgan_main[drug_seen] = values
            mtr_main[drug_seen] = values

        unseen_drug_index = np.flatnonzero(~drug_seen)
        main_neighbors = pd.DataFrame()
        if len(unseen_drug_index):
            prediction, main_neighbors, components = self.externalizer.predict(
                unique_drugs[unseen_drug_index],
                prepared.fingerprint[unseen_drug_index],
                prepared.mtr[unseen_drug_index],
            )
            drug_main[unseen_drug_index] = prediction
            morgan_main[unseen_drug_index] = components[:, 0]
            mtr_main[unseen_drug_index] = components[:, 1]
            maximum = (
                main_neighbors[main_neighbors["rank"].eq(1)]
                .set_index("query_id")["similarity"]
            )
            maximum_tanimoto[unseen_drug_index] = maximum.loc[
                unique_drugs[unseen_drug_index]
            ].to_numpy(np.float64)

        profile_source_drugs = np.asarray(profile_state["drug_ids"]).astype(str)
        profile_factor_map = {
            entity: np.asarray(value, dtype=np.float64)
            for entity, value in zip(
                profile_source_drugs, profile_state["drug_factors"]
            )
        }
        rank = int(profile_state["rank"])
        drug_factors = np.empty((len(unique_drugs), rank), dtype=np.float64)
        if drug_seen.any():
            drug_factors[drug_seen] = np.stack(
                [profile_factor_map[entity] for entity in mapped_source_ids[drug_seen]]
            )
        profile_neighbors = pd.DataFrame()
        if len(unseen_drug_index):
            knn = profile_state["drug_knn"]
            prediction, profile_neighbors = predict_tanimoto_knn(
                prepared.fingerprint[unseen_drug_index],
                unique_drugs[unseen_drug_index],
                np.asarray(profile_state["drug_fingerprints"], dtype=np.float64),
                profile_source_drugs,
                np.asarray(profile_state["drug_factors"], dtype=np.float64),
                k=int(knn["k"]),
                gamma=float(knn["gamma"]),
                tau=float(knn["tau"]),
            )
            drug_factors[unseen_drug_index] = prediction

        cell_outputs = np.empty((len(unique_cells), 1 + rank), dtype=np.float64)
        if cell_seen.any():
            cell_outputs[cell_seen] = np.stack(
                [source_cell_output[entity] for entity in unique_cells[cell_seen]]
            )
        if (~cell_seen).any():
            features = np.concatenate((prepared.context, prepared.viper), axis=1)
            cell_outputs[~cell_seen] = _ridge_predict(features[~cell_seen], profile_state)

        drug_lookup = dict(zip(unique_drugs, range(len(unique_drugs))))
        cell_lookup = dict(zip(unique_cells, range(len(unique_cells))))
        pair_drug_index = pairs["master_drug_id"].map(drug_lookup).to_numpy(np.int64)
        pair_cell_index = pairs["cellosaurus_id"].map(cell_lookup).to_numpy(np.int64)
        pair_drug_seen = drug_seen[pair_drug_index]
        pair_cell_seen = cell_seen[pair_cell_index]
        pair_source_drug = mapped_source_ids[pair_drug_index]
        pair_seen = np.asarray(
            [
                bool(seen_drug and seen_cell)
                and (source_drug, cell) in self.source_pair_set
                for source_drug, cell, seen_drug, seen_cell in zip(
                    pair_source_drug,
                    pairs["cellosaurus_id"].astype(str),
                    pair_drug_seen,
                    pair_cell_seen,
                )
            ],
            dtype=bool,
        )
        routes = assign_routes(pair_drug_seen, pair_cell_seen, pair_seen)

        pair_drug_main = drug_main[pair_drug_index]
        cell_main_weight = float(profile_state.get("cell_main_weight", 1.0))
        pair_cell_main = cell_main_weight * cell_outputs[pair_cell_index, 0]
        profile8 = np.sum(
            drug_factors[pair_drug_index] * cell_outputs[pair_cell_index, 1:],
            axis=1,
        )
        prediction_beta = self._profile_beta(routes)
        trained_beta = float(profile_state.get("beta", 1.0))
        fixed_prior = pair_drug_main + pair_cell_main + trained_beta * profile8
        conditional = self._conditional(
            prepared,
            pair_drug_index,
            pair_cell_index,
            fixed_prior,
            batch_size,
        )

        global_mean = float(self.checkpoint["global_mean"])
        main = global_mean + pair_drug_main + pair_cell_main
        prediction_base = main + conditional + prediction_beta * profile8
        prediction = prediction_base.copy()
        prediction_without_conditional = main + prediction_beta * profile8
        profile32 = np.zeros(len(pairs), dtype=np.float64)
        correction32 = np.zeros(len(pairs), dtype=np.float64)
        both_seen = pair_drug_seen & pair_cell_seen

        correction_state = self.checkpoint.get("random_pair_correction_state")
        if correction_state is not None:
            correction_drugs = {
                entity: index
                for index, entity in enumerate(
                    np.asarray(correction_state["drug_ids"]).astype(str)
                )
            }
            correction_cells = {
                entity: index
                for index, entity in enumerate(
                    np.asarray(correction_state["cell_ids"]).astype(str)
                )
            }
            both_index = np.flatnonzero(both_seen)
            if len(both_index):
                try:
                    drug_index32 = np.asarray(
                        [correction_drugs[value] for value in pair_source_drug[both_index]],
                        dtype=np.int64,
                    )
                    cell_index32 = np.asarray(
                        [
                            correction_cells[value]
                            for value in pairs.iloc[both_index][
                                "cellosaurus_id"
                            ].astype(str)
                        ],
                        dtype=np.int64,
                    )
                except KeyError as error:
                    raise ValueError(
                        "Both-seen pair is absent from rank-32 entity maps"
                    ) from error
                profile32[both_index] = np.sum(
                    np.asarray(correction_state["drug_factors"])[drug_index32]
                    * np.asarray(correction_state["cell_factors"])[cell_index32],
                    axis=1,
                )
            prediction, correction32 = fuse_predictions(
                main,
                conditional,
                profile8,
                profile32,
                both_seen,
                correction_state["fusion_weights"],
                base_prediction=prediction_base,
            )
            weights = np.asarray(correction_state["fusion_weights"], dtype=np.float64)
            prediction_without_conditional[both_seen] = (
                main[both_seen]
                + weights[1] * profile8[both_seen]
                + weights[2] * correction32[both_seen]
            )

        result = pairs.copy()
        result["source_master_drug_id"] = np.where(
            pair_drug_seen, pair_source_drug, ""
        )
        result["drug_seen"] = pair_drug_seen
        result["cell_seen"] = pair_cell_seen
        result["pair_seen"] = pair_seen
        result["route"] = routes
        result["maximum_tanimoto"] = maximum_tanimoto[pair_drug_index]
        result["pred_global_mean"] = global_mean
        result["pred_drug_main_effect"] = pair_drug_main
        result["pred_cell_main_effect"] = pair_cell_main
        result["pred_main_effect"] = main
        result["pred_svd_profile_effect"] = profile8
        result["pred_observed_profile_effect"] = profile32
        result["pred_observed_profile_correction"] = correction32
        result["pred_conditional_response"] = conditional
        result["pred_neural_fixed_additive_effect"] = fixed_prior
        result["pred_profile_beta"] = prediction_beta
        result["prediction_base"] = prediction_base
        result["prediction_no_rp"] = prediction_base
        result["prediction_without_conditional"] = prediction_without_conditional
        result["prediction_main_effects_only"] = main
        result["prediction_global_mean"] = global_mean
        result["prediction_morgan_main_only"] = (
            global_mean + morgan_main[pair_drug_index] + pair_cell_main
        )
        result["prediction_mtr_main_only"] = (
            global_mean + mtr_main[pair_drug_index] + pair_cell_main
        )
        result["prediction"] = prediction
        forbidden = {"response_value", "LN_IC50", "GI50_uM"} & set(result)
        if forbidden:
            raise RuntimeError(f"Response labels leaked into prediction: {forbidden}")
        return FrozenPrediction(result, main_neighbors, profile_neighbors)
