"""CaDRiN neural conditional-response model."""

from __future__ import annotations

import torch
from torch import nn


class FeatureAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, code_dim, dropout=0.2):
        super().__init__()
        encoder = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for in_dim, out_dim in zip(hidden_dims, hidden_dims[1:]):
            encoder.extend(
                [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
        encoder.extend([nn.Linear(hidden_dims[-1], code_dim), nn.ReLU()])
        self.encoder = nn.Sequential(*encoder)

        decoder = [
            nn.Linear(code_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for in_dim, out_dim in zip(hidden_dims[::-1], hidden_dims[-2::-1]):
            decoder.extend(
                [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
        decoder.extend([nn.Linear(hidden_dims[0], input_dim), nn.Sigmoid()])
        self.decoder = nn.Sequential(*decoder)

    def forward(self, features):
        code = self.encoder(features)
        return code, self.decoder(code)


class ViperProjector(nn.Module):
    def __init__(self, input_dim, code_dim, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, features):
        return self.encoder(features)


class ResponsePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, features):
        return self.layers(features)


class CaDRiN(nn.Module):
    def __init__(
        self,
        fingerprint_dim,
        context_dim,
        viper_dim,
        *,
        hidden_dims=(300, 100),
        fingerprint_code_dim=30,
        context_code_dim=30,
        viper_code_dim=32,
        predictor_hidden_dims=(128, 128),
        dropout=0.2,
        global_mean=0.0,
    ):
        super().__init__()
        self.fingerprint_autoencoder = FeatureAutoencoder(
            fingerprint_dim, hidden_dims, fingerprint_code_dim, dropout
        )
        self.context_autoencoder = FeatureAutoencoder(
            context_dim, hidden_dims, context_code_dim, dropout
        )
        self.viper_autoencoder = ViperProjector(viper_dim, viper_code_dim, dropout)
        cell_dim = context_code_dim + viper_code_dim
        self.cell_to_drug = nn.Linear(cell_dim, fingerprint_code_dim)
        self.forward_network = ResponsePredictor(
            3 * fingerprint_code_dim + cell_dim,
            predictor_hidden_dims,
        )
        self.register_buffer(
            "global_mean", torch.tensor([float(global_mean)], dtype=torch.float32)
        )

    def forward(self, fingerprint, context, viper, fixed_additive_effect):
        drug_code, fingerprint_reconstruction = self.fingerprint_autoencoder(
            fingerprint
        )
        context_code, context_reconstruction = self.context_autoencoder(context)
        viper_code = self.viper_autoencoder(viper)
        cell_code = torch.cat((context_code, viper_code), dim=1)
        cell_projection = self.cell_to_drug(cell_code)
        fusion = torch.cat(
            (
                drug_code,
                cell_code,
                drug_code * cell_projection,
                torch.abs(drug_code - cell_projection),
            ),
            dim=1,
        )
        conditional_response = self.forward_network(fusion)
        prediction = self.global_mean + fixed_additive_effect + conditional_response
        return (
            prediction,
            fixed_additive_effect,
            conditional_response,
            fingerprint_reconstruction,
            context_reconstruction,
        )


def build_model(config, dimensions, global_mean):
    return CaDRiN(
        dimensions["fingerprint"],
        dimensions["context"],
        dimensions["viper"],
        hidden_dims=tuple(config["hidden_dims"]),
        fingerprint_code_dim=config["fingerprint_code_dim"],
        context_code_dim=config["context_code_dim"],
        viper_code_dim=config["viper_code_dim"],
        predictor_hidden_dims=tuple(config["predictor_hidden_dims"]),
        dropout=config["dropout"],
        global_mean=global_mean,
    )
