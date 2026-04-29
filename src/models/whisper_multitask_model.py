"""Multi-task supervised clinical voice model with Whisper backbone.

Fine-tunes Whisper large-v2 encoder with multiple clinical task heads
(binary classification and regression). The fine-tuned encoder produces
1280-dim embeddings that serve as the foundation model.

No contrastive loss, no projection bottleneck — task heads operate
directly on the 1280-dim encoder output.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

WHISPER_SAMPLE_RATE = 16000
WHISPER_MAX_SAMPLES = WHISPER_SAMPLE_RATE * 30


class WhisperClinicalModel(nn.Module):
    """Whisper encoder with multi-task clinical prediction heads.

    Architecture:
        Raw audio → WhisperFeatureExtractor (mel) → Whisper encoder
        → masked mean-pool → 1280-dim foundation embedding → task heads

    Args:
        config: Dict with keys:
            backbone.name: HuggingFace model identifier
            backbone.freeze_layers: Number of layers to freeze from bottom
            backbone.unfreeze_last_n: Number of layers to unfreeze from top
            task_conditioning.num_task_types: Number of recording task types
            tasks: List of task dicts with {name, type, input_key, weight}
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        backbone_cfg = config["backbone"]
        model_name = backbone_cfg.get("name", "openai/whisper-large-v2")
        unfreeze_last_n = backbone_cfg.get("unfreeze_last_n", 4)

        # --- Whisper encoder ---
        from transformers import WhisperModel, WhisperFeatureExtractor

        whisper_model = WhisperModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.encoder = whisper_model.encoder.float()  # ensure FP32 for training
        del whisper_model.decoder

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

        hidden_dim = self.encoder.config.d_model  # 1280 for large-v2
        self.hidden_dim = hidden_dim

        # Freeze all, then selectively unfreeze top N layers
        for param in self.encoder.parameters():
            param.requires_grad = False

        if unfreeze_last_n > 0:
            for layer in self.encoder.layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
            # Also unfreeze layer norm
            if hasattr(self.encoder, 'layer_norm'):
                for param in self.encoder.layer_norm.parameters():
                    param.requires_grad = True

        self._any_unfrozen = unfreeze_last_n > 0

        n_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.encoder.parameters())
        print(f"Whisper encoder: {n_total/1e6:.0f}M total, {n_trainable/1e6:.0f}M trainable "
              f"(top {unfreeze_last_n} layers unfrozen)")

        # --- Task-type conditioning ---
        num_task_types = config.get("task_conditioning", {}).get("num_task_types", 80)
        self.task_embedding = nn.Embedding(num_task_types, hidden_dim)

        # --- Task heads (operate on 1280-dim) ---
        self.task_configs = config.get("tasks", [])
        self.task_heads = nn.ModuleDict()
        for task in self.task_configs:
            self.task_heads[task["name"]] = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        audio_input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        task_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict]:
        """Forward pass through Whisper encoder and task heads.

        Args:
            audio_input_values: Raw waveform (batch, samples) at 16kHz.
            attention_mask: Binary mask (batch, samples).
            task_type_ids: Integer tensor (batch,) for recording task type.

        Returns:
            Dict with:
                pooled: Foundation embedding (batch, 1280)
                task_outputs: {task_name: (batch, 1)} predictions
        """
        device = audio_input_values.device

        # Convert raw audio to mel spectrograms
        audio_np = audio_input_values.cpu().numpy()
        mel_list = []
        lengths = []
        for i in range(len(audio_np)):
            wav = audio_np[i]
            if attention_mask is not None:
                valid_len = int(attention_mask[i].sum().item())
                wav = wav[:valid_len]
            lengths.append(len(wav))
            mel_list.append(
                self.feature_extractor(
                    wav, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt",
                ).input_features.squeeze(0)
            )
        mel_features = torch.stack(mel_list).to(device)  # (B, 80, 3000)

        # Run encoder
        ctx = torch.no_grad() if not self._any_unfrozen else contextlib.nullcontext()
        with ctx:
            encoder_out = self.encoder(mel_features.float(), return_dict=True)

        hidden_states = encoder_out.last_hidden_state  # (B, 1500, 1280)

        # Add task-type embedding
        if task_type_ids is not None:
            task_emb = self.task_embedding(task_type_ids).unsqueeze(1)  # (B, 1, 1280)
            hidden_states = hidden_states + task_emb

        # Masked mean-pooling based on actual audio length
        seq_len = hidden_states.shape[1]  # 1500
        pooled = torch.zeros(len(lengths), self.hidden_dim, device=device)
        for i, length in enumerate(lengths):
            token_len = max(1, int(length / WHISPER_MAX_SAMPLES * seq_len))
            token_len = min(token_len, seq_len)
            pooled[i] = hidden_states[i, :token_len, :].mean(dim=0)

        # Task head predictions
        task_outputs = {}
        for task in self.task_configs:
            name = task["name"]
            task_outputs[name] = self.task_heads[name](pooled)  # (B, 1)

        return {"pooled": pooled, "task_outputs": task_outputs}

    def get_foundation_state_dict(self) -> dict:
        """Return state dict for the foundation model (encoder + pooling, no task heads)."""
        foundation_prefixes = ("encoder.", "task_embedding.")
        return {
            k: v for k, v in self.state_dict().items()
            if any(k.startswith(p) for p in foundation_prefixes)
        }

    @torch.no_grad()
    def extract_embeddings(
        self,
        audio_input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        task_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract 1280-dim foundation embeddings (for downstream eval).

        Returns:
            L2-normalized pooled representation (batch, 1280).
        """
        self.eval()
        output = self.forward(audio_input_values, attention_mask, task_type_ids)
        return F.normalize(output["pooled"], p=2, dim=-1)
