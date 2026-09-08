import torch
import torch.nn as nn

from .visual_encoder import VisualEncoder
from .audio_encoder import AudioEncoder
from .temporal_transformer import TemporalTransformer
from .cross_attention_fusion import CrossAttentionFusion
from .classification_head import ClassificationHead


class DeepfakeDetector(nn.Module):
    """
    Complete multimodal deepfake detection model.

    Visual branch:
        Video frames
            -> ViT
            -> Temporal Transformer

    Audio branch:
        Audio waveform
            -> Wav2Vec2

    Fusion:
        Visual + Audio
            -> Cross-Attention Fusion

    Classification:
        Fused features
            -> Classification Head
            -> REAL / FAKE logits

    Inputs:
        frames:
            [B, T, 3, 224, 224]

        waveform:
            [B, samples]

    Output:
        logits:
            [B, 2]

        0 = REAL
        1 = FAKE
    """

    def __init__(
        self,
        visual_model_name: str = "google/vit-base-patch16-224",
        audio_model_name: str = "facebook/wav2vec2-base",
        feature_dim: int = 768,
        num_heads: int = 8,
        temporal_layers: int = 2,
        max_sequence_length: int = 16,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 384,
        pretrained: bool = True,
        freeze_visual_backbone: bool = False,
        freeze_audio_backbone: bool = False,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = 2

        # --------------------------------------------------
        # Visual encoder
        # --------------------------------------------------

        self.visual_encoder = VisualEncoder(
            model_name=visual_model_name,
            pretrained=pretrained,
            freeze_backbone=freeze_visual_backbone,
        )

        if self.visual_encoder.feature_dim != feature_dim:
            raise ValueError(
                "Visual encoder feature dimension does not "
                "match feature_dim."
            )

        # --------------------------------------------------
        # Temporal Transformer
        # --------------------------------------------------

        self.temporal_transformer = TemporalTransformer(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=temporal_layers,
            dropout=dropout,
            max_sequence_length=max_sequence_length,
        )

        # --------------------------------------------------
        # Audio encoder
        # --------------------------------------------------

        self.audio_encoder = AudioEncoder(
            model_name=audio_model_name,
            pretrained=pretrained,
            freeze_backbone=freeze_audio_backbone,
        )

        if self.audio_encoder.feature_dim != feature_dim:
            raise ValueError(
                "Audio encoder feature dimension does not "
                "match feature_dim."
            )

        # --------------------------------------------------
        # Cross-attention fusion
        # --------------------------------------------------

        self.fusion = CrossAttentionFusion(
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # --------------------------------------------------
        # Classification head
        # --------------------------------------------------

        self.classifier = ClassificationHead(
            input_dim=feature_dim,
            hidden_dim=classifier_hidden_dim,
            num_classes=2,
            dropout=0.5,   # increased from 0.3 to fight overfitting
        )

    def forward(
        self,
        frames: torch.Tensor,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the complete model.

        Args:
            frames:
                [B, T, 3, H, W]

            waveform:
                [B, samples]

        Returns:
            logits:
                [B, 2]
        """

        # --------------------------------------------------
        # Validate video input
        # --------------------------------------------------

        if not isinstance(frames, torch.Tensor):
            raise TypeError(
                "frames must be a PyTorch tensor."
            )

        if frames.ndim != 5:
            raise ValueError(
                "frames must have shape "
                "[B, T, 3, H, W]."
            )

        batch_size = frames.shape[0]
        sequence_length = frames.shape[1]

        if frames.shape[2] != 3:
            raise ValueError(
                "frames must have 3 color channels."
            )

        if sequence_length == 0:
            raise ValueError(
                "frames sequence cannot be empty."
            )

        # --------------------------------------------------
        # Validate audio input
        # --------------------------------------------------

        if not isinstance(waveform, torch.Tensor):
            raise TypeError(
                "waveform must be a PyTorch tensor."
            )

        if waveform.ndim != 2:
            raise ValueError(
                "waveform must have shape [B, samples]."
            )

        if waveform.shape[0] != batch_size:
            raise ValueError(
                "Video and audio batch sizes must match."
            )

        # --------------------------------------------------
        # Visual feature extraction
        # --------------------------------------------------
        #
        # [B, T, 3, H, W]
        #       ↓ reshape
        # [B*T, 3, H, W]
        #
        # ViT
        #
        # [B*T, 768]
        #
        #       ↓ reshape
        #
        # [B, T, 768]
        # --------------------------------------------------

        frames_flat = frames.reshape(
            batch_size * sequence_length,
            frames.shape[2],
            frames.shape[3],
            frames.shape[4],
        )

        visual_features = self.visual_encoder(
            frames_flat
        )

        visual_features = visual_features.reshape(
            batch_size,
            sequence_length,
            self.feature_dim,
        )

        # --------------------------------------------------
        # Temporal modelling
        # --------------------------------------------------

        # The temporal transformer produces [B, 768].
        temporal_features = self.temporal_transformer(
            visual_features
        )

        # --------------------------------------------------
        # Audio feature extraction
        # --------------------------------------------------

        audio_features = self.audio_encoder(
            waveform
        )

        # [B, 768] -> [B, 1, 768]
        audio_features = audio_features.unsqueeze(1)

        # --------------------------------------------------
        # Cross-modal fusion
        # --------------------------------------------------

        # The existing fusion module expects:
        #
        # visual: [B, T, 768]
        # audio : [B, T_audio, 768]
        #
        # We use the temporal visual representation as
        # one visual token.
        #
        # [B, 768] -> [B, 1, 768]

        temporal_features = temporal_features.unsqueeze(1)

        fused_features = self.fusion(
            temporal_features,
            audio_features,
        )

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        logits = self.classifier(
            fused_features
        )

        return logits

    def predict_proba(
        self,
        frames: torch.Tensor,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return REAL / FAKE probabilities.

        Output:
            [B, 2]

        Column 0 = REAL probability
        Column 1 = FAKE probability
        """

        logits = self.forward(
            frames,
            waveform,
        )

        return torch.softmax(
            logits,
            dim=-1,
        )

    def predict(
        self,
        frames: torch.Tensor,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return predicted class.

        0 = REAL
        1 = FAKE
        """

        probabilities = self.predict_proba(
            frames,
            waveform,
        )

        return torch.argmax(
            probabilities,
            dim=-1,
        )