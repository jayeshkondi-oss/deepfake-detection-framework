import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model


class AudioEncoder(nn.Module):
    """
    Wav2Vec2-based audio encoder.

    Input:
        waveform: [B, samples]

    Output:
        audio features: [B, 768]
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if pretrained:
            self.model = Wav2Vec2Model.from_pretrained(
                model_name,
                attn_implementation="eager",
            )
        else:
            # Random-init Wav2Vec2 with the wav2vec2-base
            # architecture, with no network access required.
            # Used for fast/offline tests and architecture checks
            # where pretrained weights don't matter.
            config = Wav2Vec2Config(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                attn_implementation="eager",
            )
            self.model = Wav2Vec2Model(config)

        self.feature_dim = (
            self.model.config.hidden_size
        )

        if freeze_backbone:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract one fixed-size feature vector
        from each audio waveform.

        Args:
            waveform:
                [B, samples]

        Returns:
            [B, 768]
        """

        if not isinstance(waveform, torch.Tensor):
            raise TypeError(
                "waveform must be a PyTorch tensor."
            )

        if waveform.ndim != 2:
            raise ValueError(
                "waveform must have shape [B, samples]."
            )

        outputs = self.model(
            input_values=waveform
        )

        hidden_states = outputs.last_hidden_state

        # Mean pooling across the temporal dimension.
        features = hidden_states.mean(dim=1)

        return features