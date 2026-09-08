"""
predict.py
Run the full deepfake detection + explanation pipeline on a single
video file from the command line.

Usage:
    python predict.py --video path/to/video.mp4 --checkpoint models/checkpoints/best_model.pt

    # Quick check without a trained checkpoint (random weights --
    # useful only for verifying the pipeline runs, not real detection):
    python predict.py --video path/to/video.mp4
"""

import argparse
from pathlib import Path

import torch

from deepfake.inference.predictor import DeepfakePredictor



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to the video file to analyze.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a trained checkpoint (e.g. models/checkpoints/best_model.pt). "
        "If omitted, runs with random (untrained) weights -- only useful for "
        "verifying the pipeline itself runs, not for real predictions.",
    )
    parser.add_argument(
        "--no_explanations", action="store_true",
        help="Skip the visual/audio attention explanations for a faster run "
        "(classification + temporal localization still run).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Confidence threshold for FAKE verdict (default: 0.65). "
        "The model's fake-probability must exceed this value to label "
        "a video as FAKE. Lower = more sensitive, higher = more specific. "
        "Range: 0.0 to 1.0.",
    )
    parser.add_argument(
        "--save_visualizations", type=str, default="outputs",
        help="Directory to save the visual attention heatmap, audio "
        "attention plot, and fake-probability timeline as PNG images "
        "(default: outputs). Pass --no_explanations to skip saving.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.checkpoint is None:
        print(
            "WARNING: no --checkpoint given. Running with random, untrained "
            "weights -- this will NOT produce meaningful predictions, only "
            "confirms the pipeline runs end to end."
        )

    predictor = DeepfakePredictor(
        checkpoint_path=args.checkpoint,
        device=device,
        # A checkpoint (when given) overwrites every weight anyway, so
        # there's no reason to spend time downloading pretrained
        # encoder weights first just to immediately replace them.
        pretrained=False,
        fake_threshold=args.threshold,
    )

    print(f"Analyzing {args.video} ...")
    report = predictor.predict(
        args.video,
        compute_explanations=not args.no_explanations,
        save_visualizations_dir=(
            None if args.no_explanations else args.save_visualizations
        ),
    )
    predictor.close()

    print()
    print(report.summary())

    if not args.no_explanations and args.save_visualizations:
        html_path = Path(args.save_visualizations) / "report.html"
        generated_html = report.generate_html_report(str(html_path))
        abs_html = Path(generated_html).resolve()
        url_path = str(abs_html).replace("\\", "/")
        print(f"\n[+] Interactive Visual Report generated:")
        print(f"    file:///{url_path}")
        print(f"    (Open this file in your browser to view heatmaps & interactive charts)")



if __name__ == "__main__":
    main()

