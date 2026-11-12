# CaDRiN usage guide

[Back to the project overview](../README.md)

## Local prediction

Install CaDRiN following the [quickstart](../README.md#quickstart). The tested
environment uses Python 3.9.20 and PyTorch 2.1.1. CPU inference is supported;
GPU inference and training require a CUDA-compatible PyTorch installation.

Run the example and save predictions:

```bash
python examples/run_example.py --bundle examples/cadrin_pretrained_bundle.zip \
  --output predictions.csv
```

Use `--device cuda` or `--device auto` to select a GPU. Use `--config` to
provide your own prediction configuration. The loader accepts a complete
model bundle as a ZIP archive or an unpacked directory.

Predictions are expressed as natural-log IC50 in micromolar units. The output
includes the final prediction, visibility route and component predictions.

### Python API

```python
from cadrin import predict_dataset

prediction, metadata = predict_dataset(
    bundle_dir="examples/cadrin_pretrained_bundle.zip",
    dataset_config="examples/config.json",
    device="cpu",
)
print(prediction.frame[["sample_id", "route", "prediction"]])
```

`predict_dataset()` loads the complete model, validates the MTR encoder
settings and prepares the input features before prediction.

## Input data

Use [examples/config.json](../examples/config.json) as a template. Paths are
resolved relative to the configuration file. Tabular inputs support gzip
compression with `.csv.gz` or `.tsv.gz` extensions.

| Prediction input | Format | Required identifiers and features |
| --- | --- | --- |
| Drug-cell pairs | TSV | `sample_id`, `cellosaurus_id`, `master_drug_id`; optional `source_dataset` |
| Morgan fingerprints | TSV | `master_drug_id`, binary `bit_0` through `bit_1023` |
| ChemBERTa-MTR features | TSV | `master_drug_id`, `canonical_smiles`, `chemberta::0` through `chemberta::383` |
| Context networks | CSV | `sample_id` and the Context columns recorded in the bundle scalers |
| VIPER activities | TSV | `cellosaurus_id` and the transcription factor columns recorded in the bundle scalers |
| MTR metadata | JSON | Frozen encoder identity, pooling and feature column definitions |

In the pairs table, `sample_id` uniquely identifies each prediction row.
In the Context table, `sample_id` identifies a cell and must match the pairs
table's `cellosaurus_id`. Feature tables must contain one row per drug or cell
identifier and cover all requested pairs. Morgan and MTR values must be finite;
MTR features must follow the bundle's frozen encoder settings.

Reuse the established, dataset-specific Context inputs and their upstream
design. CaDRiN applies its fitted scalers to those values. Drug identity is
matched by canonical isomeric SMILES and cell identity by Cellosaurus ID.

## Training

Training starts from prepared GDSC1/GDSC2 responses, fixed splits and the
Morgan, MTR, Context and VIPER inputs. The paper experiments use seeds
`3078`, `1103` and `2027` across `random_pair`, `cold_cell`, `cold_drug`,
`cold_scaffold` and `cold_scaffold_cell`.

Copy the [training configuration](../src/cadrin/configs/cadrin.json) to
`training.json` and set the paths to your prepared inputs and splits:

```python
from cadrin import train_from_config, FrozenCaDRiNPredictor

bundle_dir = train_from_config(
    "training.json",
    scenario="random_pair",
    outdir="my_model",
    training_seed=3078,
    split_seed=3078,
    device="auto",
)
model = FrozenCaDRiNPredictor.load(bundle_dir)
```

Use a separate, new or empty output directory for each scenario and seed.
Training fits all model components, selects the neural checkpoint on validation
data and exports a complete bundle for the same prediction API.

### Training inputs

In addition to the prediction features above, provide:

| Training input | Format | Required contents |
| --- | --- | --- |
| Response table | TSV | `sample_id`, `source_dataset`, `cellosaurus_id`, `master_drug_id`, `response_value` |
| Split assignments | CSV.gz | `{scenario}.csv.gz` with `cellosaurus_id`, `master_drug_id`, `split` |
| Split metadata | JSON | `{scenario}.meta.json` with `scenario`, `seed` and applicable drug group mappings |

Response values are natural-log IC50 in micromolar units. Preserve the original
GDSC1/GDSC2 labels and measurements. Training prefers GDSC2 for overlapping
pairs and fits source offsets using training pairs; each additional source
requires at least 20 paired training responses with GDSC2.

The split must cover the available pairs after deduplication and feature
filtering. Split labels are `train`, `valid`, `test` or `excluded`.
Cold-drug metadata includes `drug_fingerprint_groups`; scaffold-held-out
metadata includes `drug_scaffold_groups`.

Training uses the existing complete, symmetric 50-by-50 Context network with
columns `PC0-PC0` through `PC49-PC49`, a cell identifier in `sample_id` and
optional `Set` metadata. It selects the ordered upper-triangle edges from
these supplied values. Feature dimensions are fixed at 1,024 Morgan bits,
1,225 Context edges and 277 VIPER activities. Use the established VIPER
column and metadata schema shown in the example.

Both `main_effect.mtr_features` and `main_effect.mtr_audit` are required.
The MTR table must cover train, validation and test drugs; the metadata
must include `feature_spec_version`, the frozen `encoder` settings and
`outputs.shape` as `[number_of_drugs, 384]`. Training drug structures must
be valid and unique after canonical isomeric SMILES conversion.

### Model export

Distribute the complete bundle so that weights, scalers, externalizers,
response priors, calibration parameters and entity mappings load together:

```bash
python -m zipfile -c my_model.zip my_model/bundle
```

Pass `my_model.zip` to `FrozenCaDRiNPredictor.load()` or `predict_dataset()`.
