# Regression against the Python reference

The C++ library is a port of the Python pipeline in `reference/python/dashcam_vision.py`. This page describes how
the two are compared, in three levels from coarse to fine, and summarises the recorded results. The detailed tool
reference (every flag, the per-video tables, the `adas.jsonl` schema) is in
[tools/regression/README.md](../tools/regression/README.md). How the stages work is in [algorithm.md](algorithm.md).

What is compared:

- Level 1, events: the alerts CSV of the C++ `video_demo` against the CSV of the Python script, event by event.
  This is the pass/fail criterion.
- Level 2, frames: per-frame dumps of both pipelines (detections, tracks, lanes, segmentation, and with `--adas`
  the ADAS state), compared with gates. This is the tool to find out why a pair of events differs.
- Level 3, logic replay: C++ unit tests that feed recorded Python traces to the C++ ADAS classes and require the
  same outputs. They show that the decision logic is the same in both languages, so event differences are expected to come from
  the inference engines, not from the logic.

## Prerequisites

- A build of the C++ code (`build/native-release/examples/video_demo`, `build/native-release/adasvp_dump`) and the
  models in `models/`. See the README.
- For the Python side: the `.venv` of the README (`requirements.txt`) and the model weights set up with
  `tools/setup_models.py`. The Python reference runs on the same GPU, so it is slower and noisy to time.
- The videos. The reference CSVs in `tests/regression/reference/{fast,accurate}/` cover the 17 videos listed in
  `tools/regression/videos.txt`. Seven of them are published in the release and listed in
  `tools/regression/videos_public.txt`; the other ten are not distributed. Point `--videos-dir` at the directory
  that holds them, with `red_crop_*` clips in a `redlights/` subdirectory, which is the layout the download script
  creates.

The two paths below are independent: you can rerun only the C++ side against the committed reference CSVs, and
you only need the Python side to regenerate those CSVs or the traces.

## Procedure

### 1. Reference alerts from the Python script

```bash
tools/regression/make_reference.sh --videos-dir "$ADASVP_VIDEOS_DIR" --profile fast
tools/regression/make_reference.sh --videos-dir "$ADASVP_VIDEOS_DIR" --profile accurate
```

For every video in `videos.txt` it runs `dashcam_vision.py --profile P --no-seg` (detection, lanes and the ADAS
modules; segmentation does not influence any alert), copies the `<stem>_alerts.csv` files to
`tests/regression/reference/P/` and writes a `summary.txt` there with the wall time and the FCW / LDW / RLR counts
per video. The committed CSVs are the result of this step, so you only run it to regenerate them (it needs all 17
videos).

### 2. C++ alerts and the strict comparison

```bash
tools/regression/run_cpp_regression.sh --videos-dir "$ADASVP_VIDEOS_DIR" --profile fast --dry
tools/regression/run_cpp_regression.sh --videos-dir "$ADASVP_VIDEOS_DIR" --profile accurate --dry
```

For each video it runs `video_demo --no-seg --profile P` (the same flags as the reference run), then
`compare_alerts.py` against `tests/regression/reference/P/<stem>_alerts.csv`. It prints one line per video and exits
0 only if every pair is identical. `--dry` skips the output videos but still writes the alerts CSV, which makes the
run about as fast as the decode and inference allow. `--det-precision fp16|fp32` overrides the detector engine
precision of the profile.

If you only have the videos of the release, restrict the run to them:

```bash
scripts/download_videos.sh                      # puts the seven clips in videos/ and videos/redlights/
tools/regression/run_cpp_regression.sh --videos-dir videos --profile fast --dry \
    --list tools/regression/videos_public.txt
```

The list has seven videos, so the summary line reads `7/7 pairs identical` on success. The reference CSVs of those
seven are already in the repository.

`compare_alerts.py REF.csv CPP.csv [--tol 0.5]` is the comparison itself. Two events match when they have the same
`type` and the same `level_side` and their `time_s` differ by at most 0.5 s; each event is used once. It prints the
matched pairs, the events only in the reference (missing) and the events only in the C++ output (extra), and exits 0
only when nothing is missing or extra. The tolerance is not loosened to make a run pass.

### 3. Frame-by-frame diagnosis

When a pair fails, dump both pipelines on that video and compare the traces. With `--adas` the dumps include the
state of the ADAS modules frame by frame:

```bash
python tools/regression/dump_reference.py VIDEO --profile fast --adas --out outputs/dumps/fast/STEM
build/native-release/adasvp_dump VIDEO --profile fast --adas --out outputs/dumps_cpp/fast/STEM
python tools/regression/compare_dumps.py outputs/dumps/fast/STEM outputs/dumps_cpp/fast/STEM --adas
```

Without `--adas` the comparison covers the detector, tracker, lanes and segmentation, with these gates: detection
recall and precision at least 0.97 (IoU at least 0.9, same class), track id consistency at least 0.95, lane count
equal on at least 98% of the frames with a mean horizontal error of at most 2 px, and at least 99% equal segmentation
pixels. With `--adas` every alert event (type and side or level) must also have a partner in the other dump within
0.5 s. The agreement of the ego-lane state and of the lead id are printed as diagnostics.

### The borderline rule for FCW

A TTC threshold is a hard line on a continuous quantity. The two pipelines run different inference engines (PyTorch
for the reference, TensorRT for the C++), so boxes differ by float noise, and a smoothed TTC that touches the 2.7 s
threshold for a few more frames in one pipeline than in the other. Then one pipeline raises the warning and the other
does not, although both computed almost the same numbers. Per-frame equality across engines is not attainable at such a
crossing, and loosening the 0.5 s tolerance would not help, because the event is missing, not late.

`compare_dumps.py --adas` therefore classifies an FCW event that exists in only one dump as BORDERLINE (listed
separately, with both minimum TTCs, and not a failure) when the dump that did not fire it had its smoothed `fcw.ttc`
at or below `1.02 x` the threshold of that level within 0.5 s of the event. The thresholds are 2.7 s for a warning
and 1.5 s for critical, read from the reference `meta.json`. Anything else fails: any other FCW mismatch, and any
LDW or RLR mismatch. The rule is narrow on purpose, and the logic replay tests (level 3) are what show that the
decision logic itself is the same in both languages.

`compare_alerts.py` (level 1) does not know the rule: a borderline case shows up there as one missing event, and
you then run level 2 on that video to classify it.

## Results

Recorded on an RTX 4070 Laptop GPU, with the 17 videos of `videos.txt` and `--dry --no-seg`.

Events (level 1, strict, +-0.5 s):

| Profile | Result |
|---|---|
| accurate | 17 of 17 videos identical to the reference |
| fast | 16 of 17 identical, plus 1 borderline |

The borderline case is the FCW warning of `sp1_crop_30fps` at 127.96 s on the fast profile. The smoothed TTC minimum
around the event is 2.664 s in Python and 2.659 s in C++, against the 2.7 s threshold. Both dip below the threshold,
by a tiny margin; a warning needs the TTC to stay under it for a sustained number of frames
([algorithm.md](algorithm.md), section 6), and only the Python run did. Both minima are within `1.02 x` the
threshold, which is what the borderline rule above accepts. The accurate profile uses the fp16 detector by default,
which is the configuration behind the 17 of 17 result.

Speed (frames-weighted mean over the 17 videos, frames per second; C++ is `video_demo --dry --no-seg`, Python is the
wall time of the reference run, which includes model loading and shares the GPU, so it is noisy):

| Profile | C++ | Python |
|---|---|---|
| fast | 129.2 | 20.1 |
| accurate, fp16 detector (default) | 48.6 | 12.7 |
| accurate, fp32 detector | 20.3 | not measured |

The per-video tables are in [tools/regression/README.md](../tools/regression/README.md).

Tests: 72 of 72 passing in the last full run, with the models, videos and traces present.

## Replay tests and their traces

The unit tests of the ADAS classes (`EgoLaneTracker`, `LeadVehicleEstimator`, `FCWMonitor` with `LDWMonitor`,
`RLRMonitor`) and the `Renderer` test compare the C++ against recorded Python output. Those recordings are gitignored
(`outputs/`), so these tests skip themselves when they are missing. To regenerate them (they need the
videos, in `ADASVP_VIDEOS_DIR`):

```bash
# traces: adas.jsonl, frames.jsonl, seg/, meta.json, for the three videos the tests read
for v in sp1_crop_30fps sp1_crop_4_30fps redlights/red_crop_11_30fps; do
  python tools/regression/dump_reference.py "$ADASVP_VIDEOS_DIR/$v.mp4" --profile fast --adas \
      --out outputs/dumps/fast/$(basename "$v")
done
```

The `Renderer` test (`Renderer matches the Python drawing`) compares C++ drawings with PNGs that the Python
drawing code renders from those traces:

```bash
python tools/regression/render_reference.py --trace outputs/dumps/fast/STEM --frames A:B \
    --out outputs/render_ref/STEM_A_B
```

The windows (`STEM`, `A`, `B`) are listed in `tests/test_render.cpp`; the directory name must be `STEM_A_B` because
the test builds it from the window. `render_reference.py` reads the videos from `--videos-dir` (default
`ADASVP_VIDEOS_DIR`). The test requires at least 99.5% identical pixels and a mean absolute difference of at most
0.5 per channel; `ADASVP_RENDER_OUT=DIR` makes it keep its own renders for inspection.

Run all of it with:

```bash
ctest --test-dir build/native-release --output-on-failure
```

Tests that need models or videos are skipped without them, and the replay and `Renderer` tests are skipped without
the traces, so "all tests pass" is only meaningful when the output reports nothing skipped.
