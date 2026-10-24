---
title: DeepForensics AI - Deepfake Detection Framework
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Multi-Modal Deepfake Detection Framework (DeepForensics AI)

A robust multi-modal deepfake detection and explainable AI (XAI) framework that analyzes both facial visual dynamics and acoustic speech signals to detect manipulated media with high confidence.

## Core Architecture
- **Vision Backbone**: Pretrained Vision Transformer (ViT-Base/16) for spatial patch attention.
- **Audio Backbone**: Meta Wav2Vec 2.0 for phoneme-level acoustic frequency modeling.
- **Cross-Modal Attention Fusion**: Bidirectional cross-attention aligning visual facial speech movements with voice spectrograms.
- **Temporal Transformer**: Sequence aggregation across 16 uniformly sampled video frames.
- **Explainability (XAI)**:
  - 14×14 patch attention rollout heatmap.
  - Per-frame manipulation timeline localization.
  - Acoustic cross-modal synchronization spectrogram.
  - MediaPipe landmark face tracking with +20% margin expansion.
