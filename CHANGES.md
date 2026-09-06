# Changes from Claude's review

This documents what was added, fixed, and removed in this pass, and
what's actually left to do. All claims below were verified by running
the project's own test suite, not just read by eye.

## Added: the entire `deepfake/data/` package

This didn't exist before, even though tests for it already did
(`tests/test_face_detector.py`, `test_video_loader.py`,
`test_preprocessing.py`, `test_audio_processor.py`,
`test_audio_preprocessing.py`, `test_dataset.py`) — meaning 7 test
files were failing to even collect. Every file below was built
directly against those existing test specs and passes them in full:

- `face_detector.py` — MediaPipe BlazeFace wrapper (`FaceDetector`, `FaceDetection`) — 5/5 tests pass
- `video_loader.py` — OpenCV-based frame reading (`VideoLoader`) — 4/4 tests pass
- `preprocessing.py` — frame resize/normalize for the ViT (`FramePreprocessor`) — 5/5 tests pass
- `audio_processor.py` — file-to-tensor audio pipeline (`AudioProcessor`) — 6/6 tests pass
- `audio_preprocessing.py` — in-memory waveform preprocessing (`AudioPreprocessor`), built as a thin wrapper around `AudioProcessor` so the two share one implementation — 7/7 tests pass
- `dataset.py` — `DeepfakeDataset`, wires video loading → face detection → preprocessing into `{"frames", "label"}` samples — 6/6 tests pass

## Fixed: two real bugs

1. **`trainer.py` / `deepfake_detector.py` parameter mismatch.**
   `Trainer._forward()` called `self.model(frames=frames, audio=audio)`,
   but `DeepfakeDetector.forward()` takes a parameter named `waveform`,
   not `audio`. This was invisible to `test_trainer.py` because it only
   exercises a mock model with a matching `audio` param name — it would
   have thrown `TypeError` the first time anyone trained the real model.
   Fixed by passing `waveform=audio` in `trainer.py`. Verified directly
   against a stand-in with the real model's exact signature.

2. **`visual_encoder.py` didn't support `pretrained=False`,** which is
   what broke `visual_feature_extractor.py` (and its 5 tests, all of
   which construct with `pretrained=False` to stay offline-testable).
   `VisualEncoder` now takes a `pretrained: bool = True` flag — `False`
   builds a randomly-initialized ViT with the same architecture, with
   zero network calls. Default behavior (`pretrained=True`) is
   unchanged, so `deepfake_detector.py` isn't affected.

## Removed: dead/duplicate code

- `src/models/audio_encoder.py`, `src/models/vision_encoder.py` — early
  draft duplicates, out of sync with the real versions in
  `src/deepfake/models/`.
- Empty leftover directories `src/explainability/`, `src/inference/`,
  `src/training/` — superseded by `src/deepfake/{explainability,inference,training}/`.

## Added: packaging

- `pyproject.toml` (src layout, editable install) + `requirements.txt`.
  Previously there was no dependency/packaging file at all. Now:
  `pip install -e .` once, then plain `pytest` works from the repo
  root — no manual `PYTHONPATH` needed.

## Test suite status (verified, not assumed)

```
69 passed, 4 skipped, 13 failed
```

All 13 failures are environment-only, not code bugs — confirmed by
their exact error messages:
- `test_audio_encoder.py`, `test_visual_encoder.py`,
  `test_deepfake_detector.py` (11 tests) fail with
  `OSError: couldn't connect to huggingface.co` — this sandbox has no
  internet access to Hugging Face Hub to download pretrained ViT/Wav2Vec2
  weights. Will pass on a machine with normal internet access.
- `test_gpu.py`, `test_neural_network.py` (2 tests) fail with
  `CUDA is not available` — this sandbox has no GPU. Will pass on your
  own GPU.

**Action item:** run `pytest` once on your actual machine (which has
both internet and a GPU) to get a true 100% baseline before building
further.

---

## Round 2: merged in `test_audio_window.py` + `test_video_audio_extractor.py`

A second upload added two new test files, written (again, nicely) in
the TDD style, but based on the *original* unfixed project rather than
the fixed version above — so this round starts by merging both
together, then builds out the two new modules those tests specify.

### Added

- **`deepfake/data/video_audio_extractor.py`** (`VideoAudioExtractor`)
  — shells out to `ffmpeg` to pull the audio track out of a video file
  into a mono, resampled waveform (or a `.wav` file, via
  `extract_audio`). **This adds a new system dependency: `ffmpeg` must
  be installed and on `PATH`** — it's not a Python package, so it
  won't come from `pip install -r requirements.txt`. Install it via
  your OS package manager (e.g. `apt install ffmpeg` on
  Ubuntu/WSL, or download a build for Windows and add it to `PATH`).
  6/6 tests pass, including two that exercise a real generated video
  end to end (not mocked).

- **`deepfake/data/audio_window.py`** (`AudioWindowExtractor`) — wraps
  `VideoAudioExtractor` to force the extracted audio to an exact
  number of samples (pads short clips, trims long ones), so every
  sample in a training batch lines up. 4/4 tests pass.

### Fixed: test portability bug

`test_video_audio_extractor.py` hardcoded an absolute Windows path
(`C:\temp\test_video.mp4`) with **no existence check or skip guard**
on 3 of its 5 tests — meaning they'd fail on any machine (including
CI, and your own teammates' machines) that didn't have that exact file
manually placed at that exact path first. `test_audio_window.py` had
the same hardcoded path but at least guarded the affected tests with
`@pytest.mark.skipif`.

Fixed by adding a shared `tests/conftest.py` fixture (`sample_video`)
that generates a tiny real synthetic video (test pattern + a 440Hz
tone) via `ffmpeg` on the fly, scoped once per test session. Both test
files now use this fixture instead of the hardcoded path — every test
runs for real, on any machine with `ffmpeg` installed, with nothing to
set up by hand.

### Test suite status after merge

```
78 passed, 4 skipped, 13 failed
```

Same 13 environment-only failures as before (no Hugging Face Hub
access, no GPU, both expected in this sandbox). All 9 tests for the
two new modules pass for real, against a genuinely generated video —
not mocked, not skipped.

## Where the project actually stands

This is further along than a typical Week 1 checkpoint. The full model
architecture — visual encoder, audio encoder, cross-attention fusion,
temporal transformer, classification head, and a working training loop
with checkpointing/metrics — was already built and is solid. Combined
with the data layer added here, you're close to being able to run a
real, if small, end-to-end training job.

**What's still genuinely missing:**
- `deepfake/explainability/` — empty stub. This is your XAI module
  (Grad-CAM/attention rollout for visual, attention weights for audio,
  the per-segment timeline for temporal) — the biggest remaining
  coding gap.
- `deepfake/inference/` — empty stub. This is the script that loads a
  trained checkpoint, runs a full video through the pipeline, and
  produces the human-readable report.
- No real training run has happened yet — worth doing a tiny smoke
  test (a handful of videos, 1 epoch) on your own GPU as the very next
  step, before scaling up data or building XAI, so you catch any
  further integration issues early while they're still cheap to fix.

---

## Round 3: `train.py`, `predict.py`, explainability, and inference

Built the two genuinely missing pieces from the original architecture
diagram -- explainability and inference -- plus the top-level scripts
that actually wire everything into something runnable.

### Added: `train.py`

Nothing previously tied `DeepfakeDataset` -> `DeepfakeDetector` ->
`Trainer` together into a runnable script. `train.py` does that:
`python train.py --data_dir data/raw`. Verified with a full dry run
using a lightweight stand-in model (same call signature as
`DeepfakeDetector`) against real generated video+audio data, proving
the dataset -> dataloader -> trainer -> checkpoint wiring is correct.

### Fixed: `DeepfakeDataset` didn't return audio

`DeepfakeDetector.forward()` requires both `frames` and `waveform`
with no default -- but `DeepfakeDataset` only returned `frames`. Now
extracts a fixed-length audio window per video (via
`AudioWindowExtractor`) and includes it as `"audio"` in each sample,
falling back to silence if a video has no audio track (verified this
doesn't break the existing silent-video test fixtures, and separately
verified a video *with* real audio produces a real, non-silent
waveform, not the fallback).

### Added: `deepfake/explainability/`

- `visual_explainer.py` (`VisualExplainer`) -- Attention Rollout
  (Abnar & Zuidema, 2020) on the ViT visual encoder, producing a
  saliency map per frame showing which regions the model attended to.
  5/5 tests pass.
- `audio_explainer.py` (`AudioExplainer`) -- same rollout technique
  adapted for Wav2Vec2's frame sequence (no CLS token, so importance
  is estimated from average received attention), producing an
  importance-over-time curve. 6/6 tests pass.
- `temporal_explainer.py` (`TemporalExplainer`) -- per-frame FAKE-
  probability timeline ("when is it fake"). Reuses the model's
  already-trained fusion + classification head, applied to each
  frame's temporal representation individually instead of only the
  pooled whole-video representation -- zero additional training
  required. Only the small fusion+classifier modules run once per
  frame; the heavy ViT/Wav2Vec2 encoders each still run exactly once.
  6/6 tests pass.
- `report.py` (`build_report` / `DeepfakeReport`) -- combines the
  classifier verdict, per-frame timeline, and both XAI outputs into a
  structured report with a human-readable `.summary()`. 5/5 tests pass.

### Fixed: attention extraction was silently broken

`output_attentions=True` returned an *empty* tuple on both ViT and
Wav2Vec2 -- caught by the explainer tests, not assumed. Root cause:
recent `transformers` versions default to SDPA attention, which
doesn't support returning attention weights at all, and fails
silently rather than erroring. Fixed by forcing
`attn_implementation="eager"` in both `VisualEncoder` and
`AudioEncoder`. This affects real pretrained weights too, not just
the offline test path.

### Added: `pretrained` flag on `DeepfakeDetector` and `AudioEncoder`

Same pattern as the earlier `VisualEncoder` fix -- both now support
`pretrained=False` for offline construction (random weights, same
architecture, no network access), which is what made it possible to
build and verify the explainability and inference modules fully
offline in this sandbox. Default behavior (`pretrained=True`) is
unchanged.

### Added: `TemporalTransformer` sequence output

Added an optional `return_sequence` flag to `TemporalTransformer.forward()`.
When `True`, also returns the full per-timestep encoded sequence
(before the final mean-pooling), which `TemporalExplainer` uses for
localization. Default behavior (`return_sequence=False`) is
unchanged, verified against the existing test suite.

### Added: `deepfake/inference/predictor.py` (`DeepfakePredictor`)

Ties everything together: loads a checkpoint (or runs with random
weights for smoke-testing), then `.predict(video_path)` runs face
detection, preprocessing, audio extraction, classification, temporal
localization, and both XAI explanations, returning a `DeepfakeReport`.
3/3 end-to-end tests pass against a real generated video with a real
audio track -- the full pipeline, not mocked.

### Added: `predict.py`

CLI entry point: `python predict.py --video path/to/video.mp4 --checkpoint models/checkpoints/best_model.pt`.
Verified by actually running it as a subprocess against a real video,
not just calling the underlying class from a test.

### Test suite status after this round

```
103 passed, 4 skipped, 13 failed
```

Same 13 environment-only failures throughout this entire project --
no Hugging Face Hub access and no GPU in this sandbox, both expected
and unrelated to code correctness. Every single test for newly added
code (25 new tests across explainer + predictor modules) passes for
real.

### What's genuinely left

1. **A real training run on real data**, on your own GPU -- this is
   the one thing that cannot be done in this sandbox. Everything is
   wired and verified; it just hasn't seen real weights or real data
   yet.
2. Once trained, run `predict.py` with the real checkpoint and sanity
   check the explanations look reasonable on actual videos.
3. Evaluation, hyperparameter tuning, and writeup -- all downstream of
   step 1.

---

## Round 4: real saved visualizations, not just text

`predict.py` previously only printed "explanation available" as text
without showing anything. Faculty (or anyone) can't evaluate an
explanation they can't see, so this round makes the explanations
actually visible.

### Added: `deepfake/explainability/visualize.py`

- `save_visual_saliency()` -- overlays the attention-rollout saliency
  map on its source frame (denormalized back to a viewable image) and
  saves as a PNG, using the most suspicious frame from the temporal
  timeline.
- `save_audio_importance_plot()` -- line plot of audio attention over
  time.
- `save_temporal_timeline_plot()` -- bar chart of per-frame fake
  probability, color-coded above/below the 0.5 threshold -- the "when
  is it fake" timeline as an actual chart.

All three verified by actually running `predict.py` end to end against
a real generated video and visually inspecting the output images, not
just checking file sizes.

### Changed: `predictor.py` / `predict.py` / `report.py`

- `DeepfakePredictor.predict()` takes a new `save_visualizations_dir`
  argument; when given, saves all three PNGs and attaches their paths
  to the report.
- `predict.py` CLI: new `--save_visualizations` flag (defaults to
  `outputs/`).
- `DeepfakeReport` has a new `visualization_paths` field, and
  `.summary()` lists the saved file paths when present.

### Added dependency: `matplotlib`

Added to both `requirements.txt` and `pyproject.toml`.

### Test suite status after this round

```
105 passed, 4 skipped, 13 failed
```

Same 13 environment-only failures throughout. 2 new tests added
specifically for the visualization-saving path (real files get
created, non-empty, paths appear in the summary; and confirmed no
files are saved when the option isn't used).

---

## Round 5: fixing the actual overfitting cause, not just symptoms

Diagnosed from a real training run: `Val Acc` stuck at exactly the
same value across all epochs, on a dataset later confirmed to be only
16.7% real videos (10 real, 50 fake, from `organize_dfdc.py`'s
default first-N-in-metadata-order sampling). With a val_split of
0.15 on 60 videos (9 val videos), a random split from that skew
statistically lands only 1-2 real videos in validation -- explaining
why validation accuracy was noisy and non-improving regardless of
training progress.

### Fixed: `organize_dfdc.py` now balances real/fake by default

Previously took the first N entries in `metadata.json` order, which
silently inherited DFDC's natural skew toward FAKE videos (each real
video has several fakes derived from it). Now separately shuffles and
samples from the REAL and FAKE lists, capping each at `max_videos //
2`, so a requested subset is genuinely balanced. Verified against a
synthetic 10-real/50-fake source (matching the user's actual reported
numbers): old behavior would reproduce that skew, new default
produces exactly 8/8 from a 16-video request. Also prints a clear
warning when a source folder doesn't have enough real videos to hit
the requested balance, with guidance to pull from additional DFDC
parts. `--no_balance` preserves the old behavior if ever needed.

### Added: backbone freezing flags in `train.py`

New `--freeze_visual_backbone` / `--freeze_audio_backbone` flags,
passed through to `DeepfakeDetector`'s existing (already-implemented)
freeze parameters. Verified this reduces trainable parameters by
89.3% (202M -> 21.6M). Fully fine-tuning two huge pretrained
transformers on a few dozen or few hundred videos is itself a major
overfitting risk; freezing them and training only the smaller
fusion/temporal/classifier layers on top is the standard fix for this
data regime, and can be relaxed later once the dataset grows.

### Test suite status after this round

```
105 passed, 4 skipped, 13 failed
```

Same 13 environment-only failures as every round. Verified on the
user's own machine separately: 120 passed, 0 failed (their local
suite has 2 additional GPU-variant tests not present in this
sandbox's copy).
