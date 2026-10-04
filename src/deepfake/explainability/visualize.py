"""
visualize.py
Save visual/audio/temporal explanations as actual image files: a
heatmap overlay for visual attention, and line/bar plots for audio
attention and the per-frame fake-probability timeline. A picture of
the model's attention is far more convincing than a sentence saying
an explanation exists.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Must match FramePreprocessor's normalization exactly, so frames can
# be converted back into a viewable image.
_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
_STD = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)


def _denormalize_frame(frame: torch.Tensor) -> np.ndarray:
    """Convert a preprocessed [3, H, W] tensor back to a viewable
    [H, W, 3] uint8 image."""
    frame = frame.detach().cpu() * _STD + _MEAN
    frame = frame.clamp(0, 1)
    image = frame.permute(1, 2, 0).numpy()
    return (image * 255).astype(np.uint8)


def save_visual_saliency(
    frame: torch.Tensor,
    saliency: torch.Tensor,
    output_path: str | Path,
) -> Path:
    """
    Overlay an attention-rollout saliency map on its source frame and
    save as a PNG.

    Args:
        frame: preprocessed frame tensor [3, H, W].
        saliency: saliency map [grid_h, grid_w], values in [0, 1].
        output_path: where to save the PNG.

    Returns:
        output_path, as a Path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = _denormalize_frame(frame)
    height, width = image.shape[:2]

    saliency_resized = (
        F.interpolate(
            saliency.detach().cpu().unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .numpy()
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image)
    ax.imshow(saliency_resized, cmap="jet", alpha=0.45)
    ax.set_title("Visual attention (brighter = more attended)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def save_audio_importance_plot(
    importance: torch.Tensor,
    times_seconds: torch.Tensor,
    output_path: str | Path,
) -> Path:
    """Line plot of audio attention importance over time."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    times = times_seconds.detach().cpu().numpy()
    values = importance.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times, values, color="#d1495b")
    ax.fill_between(times, values, alpha=0.2, color="#d1495b")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Attention importance")
    ax.set_title("Audio attention over time")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def save_temporal_timeline_plot(
    fake_probability_per_frame,
    times_seconds,
    output_path: str | Path,
    threshold: float = 0.5,
) -> Path:
    """Bar chart of per-frame fake probability -- the 'when is it
    fake' timeline."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    probs = np.asarray(fake_probability_per_frame)
    times = (
        np.asarray(times_seconds)
        if times_seconds is not None
        else np.arange(len(probs))
    )

    colors = ["#ef4444" if p >= threshold else "#10b981" for p in probs]
    bar_width = (times[1] - times[0]) * 0.8 if len(times) > 1 else 0.5

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(times, probs, width=bar_width, color=colors)
    ax.axhline(threshold, color="#a855f7", linestyle="--", linewidth=1.5, label=f"Threshold ({int(threshold*100)}%)")
    ax.set_xlabel(
        "Time (seconds)" if times_seconds is not None else "Frame index"
    )
    ax.set_ylabel("Fake probability")
    ax.set_title(f"Per-frame fake-probability timeline (Threshold: {int(threshold*100)}%)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
