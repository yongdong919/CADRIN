"""Configuration loading for CaDRiN training and frozen inference."""

from __future__ import annotations

from pathlib import Path

from .utils import read_json, require_file


BASE_SCENARIOS = ("random_pair", "cold_cell", "cold_drug")
TRAINING_SCENARIOS = BASE_SCENARIOS + (
    "cold_scaffold",
    "cold_drug_cell",
    "cold_scaffold_cell",
)
# Supported training scenarios.
SCENARIOS = TRAINING_SCENARIOS


def load_feature_config(path):
    """Read a feature-path JSON file relative to its own directory."""

    path = require_file(path, "feature config")
    config = read_json(path)
    for name, value in config.items():
        if name.endswith("_path"):
            value = Path(value).expanduser()
            if not value.is_absolute():
                value = path.parent / value
            config[name] = str(value.resolve())
    return config


def load_training_config(path):
    """Load the fixed CaDRiN paper-training configuration."""

    path = require_file(path, "training config")
    config = read_json(path)
    for section in (
        "experiment",
        "paths",
        "model",
        "main_effect",
        "response_profile",
        "random_pair_correction",
        "loss",
        "training",
    ):
        if section not in config:
            raise ValueError(f"Configuration is missing the {section!r} section")

    for name in ("context", "fingerprint", "viper", "response", "split_root"):
        value = Path(config["paths"][name]).expanduser()
        if not value.is_absolute():
            value = (path.parent / value).resolve()
        config["paths"][name] = str(value)
    mtr = Path(config["main_effect"]["mtr_features"]).expanduser()
    if not mtr.is_absolute():
        mtr = (path.parent / mtr).resolve()
    config["main_effect"]["mtr_features"] = str(mtr)
    if not config["main_effect"].get("mtr_audit"):
        raise ValueError("Training requires main_effect.mtr_audit for bundle export")
    mtr_audit = Path(config["main_effect"]["mtr_audit"]).expanduser()
    if not mtr_audit.is_absolute():
        mtr_audit = (path.parent / mtr_audit).resolve()
    config["main_effect"]["mtr_audit"] = str(mtr_audit)

    if config["model"]["hidden_dims"] != [300, 100]:
        raise ValueError("CaDRiN requires hidden_dims=[300, 100]")
    expected_codes = {
        "fingerprint_code_dim": 30,
        "context_code_dim": 30,
        "viper_code_dim": 32,
    }
    for name, expected in expected_codes.items():
        if int(config["model"][name]) != expected:
            raise ValueError(f"CaDRiN requires {name}={expected}")
    if config["main_effect"]["method"] != "visibility_aware_anova_morgan_mtr":
        raise ValueError("Unsupported drug main-effect method")
    if config["response_profile"]["method"] != "training_anova_residual_svd":
        raise ValueError("Unsupported response-profile method")
    correction = config["random_pair_correction"]
    if correction["method"] != "both_seen_multiresolution_response_correction":
        raise ValueError("Unsupported random-pair correction method")
    if (
        correction["fusion"]["method"]
        != "validation_nonnegative_least_squares"
        or float(correction["fusion"]["main_effect_weight"]) != 1.0
    ):
        raise ValueError("Unsupported random-pair fusion")
    return config


