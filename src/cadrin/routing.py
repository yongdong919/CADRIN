"""Deterministic identity and visibility routing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem


ROUTES = (
    "pair_replication",
    "random_pair_transfer",
    "cold_drug",
    "cold_cell",
    "double_cold",
)


def canonical_structure_key(smiles):
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    text = str(smiles).strip()
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return f"invalid::{text}"
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def assign_routes(drug_seen, cell_seen, pair_seen):
    drug_seen = np.asarray(drug_seen, dtype=bool)
    cell_seen = np.asarray(cell_seen, dtype=bool)
    pair_seen = np.asarray(pair_seen, dtype=bool)
    if not (drug_seen.shape == cell_seen.shape == pair_seen.shape):
        raise ValueError("Visibility masks are not aligned")
    if np.any(pair_seen & ~(drug_seen & cell_seen)):
        raise ValueError("A replicated pair must contain two seen entities")
    route = np.full(len(drug_seen), "double_cold", dtype=object)
    route[drug_seen & ~cell_seen] = "cold_cell"
    route[~drug_seen & cell_seen] = "cold_drug"
    route[drug_seen & cell_seen & ~pair_seen] = "random_pair_transfer"
    route[pair_seen] = "pair_replication"
    return route.astype(str)
