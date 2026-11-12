"""CaDRiN training objective."""

from __future__ import annotations

from torch import nn


class CaDRiNLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fingerprint_weight = float(config["fingerprint_reconstruction"])
        self.context_weight = float(config["context_reconstruction"])
        self.response_criterion = nn.MSELoss()
        self.fingerprint_criterion = nn.BCELoss()
        self.context_criterion = nn.MSELoss()

    def forward(self, outputs, fingerprint, context, response):
        (
            prediction,
            _,
            _,
            fingerprint_reconstruction,
            context_reconstruction,
        ) = outputs
        response_loss = self.response_criterion(prediction, response)
        fingerprint_loss = self.fingerprint_criterion(
            fingerprint_reconstruction, fingerprint
        )
        context_loss = self.context_criterion(
            context_reconstruction, context
        )
        objective = (
            response_loss
            + self.fingerprint_weight * fingerprint_loss
            + self.context_weight * context_loss
        )
        return {
            "objective": objective,
            "response": response_loss,
            "fingerprint_reconstruction": fingerprint_loss,
            "context_reconstruction": context_loss,
        }
