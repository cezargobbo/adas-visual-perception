# Regression

The C++ library is the maintained implementation. It was ported line by line from the Python prototype in
`prototype/python/dashcam_vision.py`, and the regression has two parts:

- The day-to-day check compares the C++ against its own **golden** alert CSVs in
  `tests/regression/golden/{fast,accurate}/`. It catches any change of behaviour, intended or not.
- The parity record compares the C++ with the alert CSVs of the Python prototype in
  `tests/regression/python/{fast,accurate}/`. It was the evidence that the port preserved the behaviour the author
  validated on the prototype's output videos, and it can still be rerun as an optional parity check.

The detailed tool reference (every flag, the per-video tables, the `adas.jsonl` schema) is in
[tools/regression/README.md](../tools/regression/README.md). How the stages work is in [algorithm.md](algorithm.md).

What is compared:

- Level 1, events: the alerts CSV of `video_demo` against a baseline CSV, event by event: the golden CSVs (the
  default, pass/fail criterion) or the Python prototype's CSVs (`--against python`).
- Level 2, frames: per-frame dumps of the C++ and of the Python prototype (detections, tracks, lanes, segmentation,
  and with `--adas` the ADAS state), compared with gates. This is the tool to find out why a pair of events differs
  from the Python prototype; it needs the Python environment.
- Level 3, logic replay: C++ unit tests that feed recorded Python prototype traces to the C++ ADAS classes and
  require the same outputs. They show that the decision logic of the port is exact, so event differences against the
  Python prototype come from the inference engines, not from the logic. These tests stay, because they are what
  proves the port.

## Prerequisites

- Python 3 (the `.venv` if it exists, otherwise `python3`; `compare_alerts.py` only uses the standard library), for the
  comparison of the alert CSVs.
- A build of the C++ code (`build/native-release/examples/video_demo`, `build/native-release/adasvp_dump`) and the
  models in `models/`. See the README.
- For the Python side (parity check, traces): the `.venv` of the README (`requirements.txt`) and the model weights
  set up with `tools/setup_models.py`. The Python prototype runs on the same GPU, so it is slower and noisy to time.
- The videos. The golden and Python CSVs in `tests/regression/{golden,python}/{fast,accurate}/` cover the 17 videos
  listed in `tools/regression/videos.txt`. Seven of them are published in the release and listed in
  `tools/regression/videos_public.txt`; the other ten are not distributed. Point `--videos-dir` at the directory
  that holds them, with `red_crop_*` clips in a `redlights/` subdirectory, which is the layout the download script
  creates.

The paths below are independent: you can rerun only the C++ side against the committed golden CSVs, and you only
need the Python side to regenerate the Python CSVs or the traces.

## Procedure

### 1. C++ alerts against the golden baseline

```bash
tools/regression/run_cpp_regression.sh --videos-dir "$ADASVP_VIDEOS_DIR" --profile fast --dry
tools/regression/run_cpp_regression.sh --videos-dir "$ADASVP_VIDEOS_DIR" --profile accurate --dry
```

For each video it runs `video_demo --no-seg --profile P`, then `compare_alerts.py` against
`tests/regression/golden/P/<stem>_alerts.csv`. It prints one line per video and exits 0 only if every pair is
identical; the expected result is `17/17 pairs identical` for both profiles. `--dry` skips the output videos but
still writes the alerts CSV, which makes the run about as fast as the decode and inference allow.
`--det-precision fp16|fp32` overrides the detector engine precision of the profile (the golden files were made with
the profile defaults, so an fp32 run is a different configuration). `--against python` compares with the Python
prototype's CSVs instead (see step 2).

If you only have the videos of the release, restrict the run to them:

```bash
scripts/download_videos.sh                      # puts the seven clips in videos/ and videos/redlights/
tools/regression/run_cpp_regression.sh --videos-dir videos --profile accurate --dry \
    --list tools/regression/videos_public.txt
```

The list has seven videos, so the summary line reads `7/7 pairs identical` on success.

The golden files are a record of the C++ output with the current code, models, TensorRT and GPU. Another GPU or
another TensorRT version can move a threshold crossing, so a different machine may show a different event. Changing
the golden files is an intentional, reviewed change: regenerate them with

```bash
tools/regression/make_golden.sh --videos-dir "$ADASVP_VIDEOS_DIR" --profile fast
tools/regression/make_golden.sh --videos-dir "$ADASVP_VIDEOS_DIR" --profile accurate
```

(it copies only the CSVs into `tests/regression/golden/P/`), check that the differences are what you meant to
change, and commit them on their own with the reason in the message.

`compare_alerts.py BASELINE.csv CPP.csv [--tol 0.5]` is the comparison itself. Two events match when they have the
same `type` and the same `level_side` and their `time_s` differ by at most 0.5 s; each event is used once. It prints
the matched pairs, the events only in the baseline (missing) and the events only in the C++ output (extra), and exits
0 only when nothing is missing or extra. The tolerance is not loosened to make a run pass.

### 2. Parity with the Python prototype (optional)

```bash
tools/regression/run_cpp_regression.sh --videos-dir "$ADASVP_VIDEOS_DIR" --profile fast --dry --against python
```

This compares the C++ with the alerts of the Python prototype. The accurate profile matches on all 17 videos. With
`--profile fast` one pair fails: the FCW warning of `sp1_crop_30fps` at 127.96 s is the documented borderline case
(`16/17`, exit code 1; see Results). With the seven-video list the fast profile gives `6/7`. To confirm that a
failure is this kind of case, dump both pipelines on that video and run `compare_dumps.py --adas` (step 3 below),
which lists it as BORDERLINE instead of a failure.

The Python alerts are the result of

```bash
tools/regression/make_python_alerts.sh --videos-dir "$ADASVP_VIDEOS_DIR" --profile fast
tools/regression/make_python_alerts.sh --videos-dir "$ADASVP_VIDEOS_DIR" --profile accurate
```

For every video in `videos.txt` it runs `dashcam_vision.py --profile P --no-seg` (detection, lanes and the ADAS
modules; segmentation does not influence any alert), copies the `<stem>_alerts.csv` files to
`tests/regression/python/P/` and writes a `summary.txt` there with the wall time and the FCW / LDW / RLR counts per
video. The Python prototype is not maintained, so these CSVs are a record: you only run the script to regenerate
them (it needs all 17 videos and the Python environment).

### 3. Frame-by-frame diagnosis against the Python prototype

When a pair fails against the Python prototype, dump both pipelines on that video and compare the traces. With
`--adas` the dumps include the state of the ADAS modules frame by frame:

```bash
python tools/regression/dump_python.py VIDEO --profile fast --adas --out outputs/dumps/fast/STEM
build/native-release/adasvp_dump VIDEO --profile fast --adas --out outputs/dumps_cpp/fast/STEM
python tools/regression/compare_dumps.py outputs/dumps/fast/STEM outputs/dumps_cpp/fast/STEM --adas
```

Without `--adas` the comparison covers the detector, tracker, lanes and segmentation, with these main gates: detection
recall and precision at least 0.97 (IoU at least 0.9, same class), track id consistency at least 0.95, lane count
equal on at least 98% of the frames with a mean horizontal error of at most 2 px, and at least 99% equal segmentation
pixels. The lane gates also require that the point count of matched lanes differs by at most 1 on at least 99.9% of
the pairs, with a maximum difference of 2 (the full list is in the docstring of `compare_dumps.py`). With `--adas` every alert event (type and side or level)
must also have a partner in the other dump within 0.5 s. The agreement of the ego-lane state and of the lead id are
printed as diagnostics.

### The borderline rule for FCW

A TTC threshold is a hard line on a continuous quantity. The two pipelines run different inference engines (PyTorch
for the Python prototype, TensorRT for the C++), so boxes differ by float noise, and a smoothed TTC that touches the 2.7 s
threshold for a few more frames in one pipeline than in the other. Then one pipeline raises the warning and the other
does not, although both computed almost the same numbers. Per-frame equality across engines is not attainable at such a
crossing, and loosening the 0.5 s tolerance would not help, because the event is missing, not late.

`compare_dumps.py --adas` therefore classifies an FCW event that exists in only one dump as BORDERLINE (listed
separately, with both minimum TTCs, and not a failure) when the dump that did not fire it had its smoothed `fcw.ttc`
at or below `1.02 x` the threshold of that level within 0.5 s of the event. The thresholds are 2.7 s for a warning
and 1.5 s for critical, read from the `meta.json` of the Python prototype dump. Anything else fails: any other FCW mismatch, and any
LDW or RLR mismatch. The rule is narrow on purpose, and the logic replay tests (level 3) are what show that the
decision logic itself is the same in both languages.

`compare_alerts.py` (level 1) does not know the rule: against the Python prototype a borderline case shows up as one
missing event, and you then run level 2 on that video to classify it. Against the golden files the rule is not
needed, because both sides are the same engine.

## Results

Recorded on an RTX 4070 Laptop GPU, with the 17 videos of `videos.txt` and `--dry --no-seg`.

Parity with the Python prototype, recorded when the port was made (level 1, strict, +-0.5 s). It shows that the
port preserved the behaviour the author had validated on the prototype's output videos:

| Profile | Result |
|---|---|
| accurate | 17 of 17 videos identical to the Python prototype |
| fast | 16 of 17 identical, plus 1 borderline |

Golden baseline: generating the golden files and then running `run_cpp_regression.sh` against them again, as an
independent second run, gave 17 of 17 identical pairs on both profiles (fast about 130 fps, accurate about 50 fps,
frames-weighted). The golden files are the C++ output itself, so the fast `sp1_crop_30fps` has the 3 events the C++
fires, not the 4 of the Python prototype.
The borderline case is the FCW warning of `sp1_crop_30fps` at 127.96 s on the fast profile. The smoothed TTC minimum
around the event is 2.664 s in Python and 2.659 s in C++, against the 2.7 s threshold. Both dip below the threshold,
by a tiny margin. A warning needs the TTC to stay under it for a sustained number of frames
([algorithm.md](algorithm.md), section 6), and only the Python run raised it, presumably because the C++ TTC stayed
below 2.7 for fewer than the 9 consecutive frames the warning needs (the per-frame dumps are not reproduced here).
Both minima are within `1.02 x` the threshold, which is what the borderline rule above accepts. The accurate profile uses the fp16 detector by default,
which is the configuration behind the 17 of 17 result.

Speed (frames-weighted mean over the 17 videos, frames per second; C++ is `video_demo --dry --no-seg`, Python is the
wall time of the Python prototype run, which includes model loading and shares the GPU, so it is noisy):

| Profile | C++ | Python |
|---|---|---|
| fast | 129.2 | 20.1 |
| accurate, fp16 detector (default) | 48.6 | 12.7 |
| accurate, fp32 detector | 20.3 | not measured |

The per-video tables are in [tools/regression/README.md](../tools/regression/README.md).

Tests: 74 of 74 passing in the last full run (`ctest`, none skipped), with the models, videos and traces present.

## Replay tests and their traces

The unit tests of the ADAS classes (`EgoLaneTracker`, `LeadVehicleEstimator`, `FCWMonitor` with `LDWMonitor`,
`RLRMonitor`) and the `Renderer` test compare the C++ against recorded output of the Python prototype. Those recordings are gitignored
(`outputs/`), so these tests skip themselves when they are missing. To regenerate them (they need the
videos, in `ADASVP_VIDEOS_DIR`):

```bash
# traces: adas.jsonl, frames.jsonl, seg/, meta.json, for the three videos the tests read
for v in sp1_crop_30fps sp1_crop_4_30fps redlights/red_crop_11_30fps; do
  python tools/regression/dump_python.py "$ADASVP_VIDEOS_DIR/$v.mp4" --profile fast --adas \
      --out outputs/dumps/fast/$(basename "$v")
done
```

The `Renderer` test (`Renderer matches the Python drawing`) compares C++ drawings with PNGs that the Python
drawing code renders from those traces:

```bash
python tools/regression/render_python.py --trace outputs/dumps/fast/STEM --frames A:B \
    --out outputs/render_ref/STEM_A_B
```

The windows (`STEM`, `A`, `B`) are listed in `tests/test_render.cpp`; the directory name must be `STEM_A_B` because
the test builds it from the window. `render_python.py` reads the videos from `--videos-dir` (default
`ADASVP_VIDEOS_DIR`). The test requires at least 99.5% identical pixels and a mean absolute difference of at most
0.5 per channel; `ADASVP_RENDER_OUT=DIR` makes it keep its own renders for inspection.

Run all of it with:

```bash
ctest --test-dir build/native-release --output-on-failure
```

Tests that need models or videos are skipped without them, and the replay and `Renderer` tests are skipped without
the traces, so "all tests pass" is only meaningful when the output reports nothing skipped.
