"""Stable chemical grouping used by cold-drug and cold-scaffold protocols."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold



def fingerprint_groups(frame: pd.DataFrame) -> pd.Series:
    """Return exact Morgan-bit equivalence groups indexed by drug ID."""

    bit_columns = sorted(
        [column for column in frame if str(column).startswith("bit_")],
        key=lambda column: int(str(column).split("_", 1)[1]),
    )
    if not bit_columns:
        raise ValueError("Fingerprint table has no bit_* columns")
    expected = [f"bit_{index}" for index in range(len(bit_columns))]
    if bit_columns != expected:
        raise ValueError("Fingerprint bit columns must be contiguous from bit_0")
    values = frame[bit_columns].to_numpy()
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("Fingerprint values must be finite binary values")
    groups = [
        hashlib.sha256(
            np.packbits(row.astype(np.uint8, copy=False)).tobytes()
        ).hexdigest()
        for row in values
    ]
    return pd.Series(
        groups,
        index=frame["master_drug_id"].astype(str),
        name="fingerprint_group",
    )


def bemis_murcko_scaffold(smiles: str) -> str:
    """Return the exact achiral Bemis-Murcko scaffold or ACYCLIC."""

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid canonical SMILES: {smiles!r}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule,
        includeChirality=False,
    )
    return scaffold or "ACYCLIC"


def load_scaffold_groups(path: str | Path) -> tuple[pd.Series, dict]:
    """Group drugs by their Bemis-Murcko scaffolds."""

    path = Path(path).expanduser().resolve()
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=["master_drug_id", "canonical_smiles"],
    )
    if (
        frame["master_drug_id"].isna().any()
        or frame["master_drug_id"].duplicated().any()
        or frame["canonical_smiles"].isna().any()
    ):
        raise ValueError("SMILES table identifiers and structures must be complete")
    groups = frame["canonical_smiles"].map(bemis_murcko_scaffold)
    groups.index = frame["master_drug_id"].astype(str)
    groups.name = "scaffold_group"
    return groups, {
        "method": "exact achiral Bemis-Murcko scaffold",
        "acyclic_policy": "single ACYCLIC group",
        "drugs": int(len(groups)),
        "groups": int(groups.nunique()),
    }
