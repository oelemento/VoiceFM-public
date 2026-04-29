"""
Tabular transformer encoder for clinical profile features.

Tokenizes mixed-type tabular data (binary, continuous, categorical) into a
sequence of learned embeddings, processes them with a small Transformer, and
projects the CLS representation into a shared embedding space for contrastive
learning with audio recordings.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTokenizer(nn.Module):
    """Convert mixed-type tabular features into a sequence of token embeddings.

    Each feature is independently projected into a ``hidden_dim``-sized vector.
    A learned ``[CLS]`` token is prepended to the sequence.

    Missing continuous values are replaced by a per-feature learned mask token.

    Args:
        feature_config: Dictionary describing the feature schema::

            {
                "binary": ["feat_a", "feat_b", ...],
                "continuous": ["feat_c", "feat_d", ...],
                "categorical": {"feat_e": 5, "feat_f": 12, ...}
            }

        hidden_dim: Dimensionality of each token embedding.
    """

    def __init__(self, feature_config: dict[str, Any], hidden_dim: int = 256) -> None:
        super().__init__()
        self.feature_config = feature_config
        self.hidden_dim = hidden_dim

        # Deterministic ordering of features so that sequence position is stable
        self.binary_names: list[str] = sorted(feature_config.get("binary", []))
        self.continuous_names: list[str] = sorted(feature_config.get("continuous", []))
        self.categorical_names: list[str] = sorted(
            feature_config.get("categorical", {}).keys()
        )
        categorical_sizes: dict[str, int] = feature_config.get("categorical", {})

        self.num_features = (
            len(self.binary_names)
            + len(self.continuous_names)
            + len(self.categorical_names)
        )

        # --- Per-feature embeddings ---
        # Binary features: each gets an Embedding(2, hidden_dim)
        self.binary_embeddings = nn.ModuleDict(
            {name: nn.Embedding(2, hidden_dim) for name in self.binary_names}
        )

        # Continuous features: each gets a Linear(1, hidden_dim)
        self.continuous_projections = nn.ModuleDict(
            {name: nn.Linear(1, hidden_dim) for name in self.continuous_names}
        )

        # Learned mask tokens for missing continuous values (one per feature)
        if self.continuous_names:
            self.continuous_mask_tokens = nn.ParameterDict(
                {
                    name: nn.Parameter(torch.randn(hidden_dim))
                    for name in self.continuous_names
                }
            )
        else:
            self.continuous_mask_tokens = nn.ParameterDict()

        # Categorical features: each gets an Embedding(num_categories, hidden_dim)
        self.categorical_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(categorical_sizes[name], hidden_dim)
                for name in self.categorical_names
            }
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Tokenize a batch of tabular features.

        Args:
            features: Dictionary mapping feature names to tensors.
                - Binary features: ``(batch,)`` long tensors with values in {0, 1}.
                - Continuous features: ``(batch,)`` float tensors. ``NaN`` values
                  are treated as missing and replaced with a learned mask token.
                - Categorical features: ``(batch,)`` long tensors with values in
                  ``[0, num_categories)``.

        Returns:
            Token embeddings of shape ``(batch, num_features + 1, hidden_dim)``
            where the first token is ``[CLS]``.
        """
        batch_size = next(iter(features.values())).shape[0]
        tokens: list[torch.Tensor] = []

        # Binary
        for name in self.binary_names:
            # (batch,) long -> (batch, hidden_dim)
            tokens.append(self.binary_embeddings[name](features[name].long()))

        # Continuous (with NaN masking)
        for name in self.continuous_names:
            vals = features[name].float()  # (batch,)
            missing = torch.isnan(vals)

            # Project valid values: (batch, 1) -> (batch, hidden_dim)
            projected = self.continuous_projections[name](vals.nan_to_num(0.0).unsqueeze(-1))

            if missing.any():
                # Replace missing positions with the learned mask token
                mask_token = self.continuous_mask_tokens[name].unsqueeze(0)  # (1, hidden_dim)
                projected = torch.where(missing.unsqueeze(-1), mask_token, projected)

            tokens.append(projected)

        # Categorical
        for name in self.categorical_names:
            tokens.append(self.categorical_embeddings[name](features[name].long()))

        # Stack feature tokens -> (batch, num_features, hidden_dim)
        token_seq = torch.stack(tokens, dim=1)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)
        token_seq = torch.cat([cls, token_seq], dim=1)  # (batch, num_features+1, hidden_dim)

        return token_seq


class ClinicalEncoder(nn.Module):
    """Tabular transformer over clinical features.

    Tokenizes mixed-type features, processes them through a stack of Transformer
    encoder layers, and projects the ``[CLS]`` output into the shared contrastive
    embedding space.

    Args:
        feature_config: Feature schema (see :class:`FeatureTokenizer`).
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        hidden_dim: Dimensionality of token embeddings and transformer hidden size.
        dropout: Dropout rate in the Transformer.
        projection_dim: Output embedding dimensionality.
    """

    def __init__(
        self,
        feature_config: dict[str, Any],
        num_layers: int = 4,
        num_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        projection_dim: int = 256,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim

        # Feature tokenizer (includes CLS token)
        self.tokenizer = FeatureTokenizer(feature_config, hidden_dim)

        # Learned positional embeddings for each token position
        # +1 for CLS token
        num_positions = self.tokenizer.num_features + 1
        self.position_embedding = nn.Embedding(num_positions, hidden_dim)

        # Layer norm before transformer (pre-norm style)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Projection head: hidden_dim -> projection_dim with L2 normalization
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode clinical features into L2-normalized embeddings.

        Args:
            features: Dictionary of feature tensors (see
                :class:`FeatureTokenizer.forward`).

        Returns:
            L2-normalized embeddings of shape ``(batch, projection_dim)``.
        """
        # Tokenize: (batch, num_features + 1, hidden_dim)
        token_seq = self.tokenizer(features)

        # Add positional embeddings
        seq_len = token_seq.shape[1]
        positions = torch.arange(seq_len, device=token_seq.device)
        token_seq = token_seq + self.position_embedding(positions).unsqueeze(0)

        # Input normalization
        token_seq = self.input_norm(token_seq)

        # Transformer: (batch, seq_len, hidden_dim)
        transformed = self.transformer(token_seq)

        # Extract CLS token (position 0)
        cls_output = transformed[:, 0, :]  # (batch, hidden_dim)

        # Project and L2-normalize
        embeddings = self.projection(cls_output)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings
