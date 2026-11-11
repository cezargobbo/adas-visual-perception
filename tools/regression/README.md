# Regression tools

- `dump_reference.py`: dumps per-frame results (`frames.jsonl`, `seg/NNNNNN.png`, `meta.json`) of the Python
  reference pipeline.
- `dump_reference.py ... --adas` also writes `adas.jsonl` (schema below) and an `adas` object in `meta.json`;
  `frames.jsonl` and `seg/` are unchanged by the flag.
- `make_tl_fixtures.py`: cuts up to 60 traffic-light crops (class 9 boxes) out of the published test videos
  into `tests/fixtures/tl_state_NNN.png`, with the Python `traffic_light_state` result in
  `tests/fixtures/tl_state_expected.json`.
- `adasvp_dump` (built from `tools/cpp/adasvp_dump.cpp`): writes the same format from the C++ `Pipeline`.
  `--no-write` skips all output (used for the fps numbers below). `--adas` also writes `adas.jsonl` (same schema as
  below, outputs only: `i`, `t`, `ego`, `lead`, `fcw`, `ldw` (+`mon`), `rlr`).
- `compare_dumps.py REF_DIR CPP_DIR`: prints and gates one video (exit 0 only if every gate passes):
  - dets: recall and precision >= 0.97 (IoU >= 0.9, same class);
  - tracks: id consistency >= 0.95 (majority C++ id -> Python id mapping over IoU-matched tracks);
  - lanes: equal count on >= 98% of frames, mean |dx| <= 2 px; point count over the true pairs: >= 99.9% with
    difference <= 1 and max difference <= 2. Matched pairs with mean |dx| > 20 px are greedy cross-lane
    mismatches (the matcher paired two different lanes when the lane sets differ between Python and C++): they
    are listed separately (count, frame, lane) and excluded from the point-count gate; they do not fail the run.
    Reason for the 99.9% / max 2 tolerance: lanes right at the score threshold can differ by a point or two
    because the `endp*71` end-point rounding falls exactly on .5 (float32 noise between torch and TensorRT).
  - seg: >= 99% equal pixels in every PNG.
  - Gates cannot pass on an empty comparison: the reference must contain seg PNGs, a reference with lanes must
    produce at least one matched lane pair, and more than 5 mismatched pairs (mean |dx| > 20) per video fail.
- `compare_dumps.py REF_DIR CPP_DIR --adas` additionally compares `adas.jsonl`: every alert event (type + side/level)
  must have a partner in the other dump within +-0.5 s (gate); `ego.state` agreement (>= 97%) and lead-id agreement
  under the majority id mapping (>= 95%) are printed as diagnostics.
  Borderline rule: an FCW event present in only one dump is classified BORDERLINE (listed separately, with both
  minimum TTCs, not a failure) when the dump that did not fire it had its smoothed `fcw.ttc` at or below
  1.02 x the threshold of that level (warning `fcw_ttc` 2.7, critical `fcw_ttc_critical` 1.5, read from the reference
  `meta.json`, else the reference defaults) within +-0.5 s of the event. Any other mismatch, and any LDW/RLR
  mismatch, fails (non-zero exit). The rule exists because per-frame equality across inference engines (torch vs
  TensorRT) is unattainable at a threshold crossing, while the logic replay tests prove the logic itself is exact.
- `render_reference.py --trace DIR --frames A:B --out OUT`: renders the Detections (`det_*`), Segmentation
  (`seg_*`) and Combined (`comb_*`) PNGs of the Python drawing from a trace that has `adas.jsonl` (needs the
  video under `ADASVP_VIDEOS_DIR`) plus `manifest.json`. The C++ test `Renderer matches the Python drawing`
  compares the `Renderer` against `outputs/render_ref/STEM_A_B/` (gates: >= 99.5% identical pixels, mean abs
  diff <= 0.5 per channel); `ADASVP_RENDER_OUT=DIR` makes the test keep its own renders.
- `make_reference.sh`: regenerates the Python reference alerts.
- `compare_alerts.py REF.csv CPP.csv [--tol 0.5]`: strict event-level comparison of two alert CSVs. Events are matched
  by `type` + `level_side` within +-tol seconds on `time_s`; prints the pairs, the missing (reference only) and the
  extra (C++ only) events and exits 0 only when nothing is missing or extra. The tolerance is never loosened.
- `run_cpp_regression.sh --videos-dir DIR --profile fast|accurate [--dry] [--det-precision fp16|fp32]`: runs
  `video_demo --no-seg` (the flags of `make_reference.sh`) on the 17 videos of `videos.txt` and runs `compare_alerts.py`
  against `tests/regression/reference/<profile>/`; one line per video, exit 0 only if all 17 pairs are identical.
  `--dry` (hidden `video_demo` flag) skips the output videos but still writes `<prefix>_alerts.csv`.

```
python tools/regression/dump_reference.py VIDEO --profile fast --out outputs/dumps/fast/STEM
build/native-release/adasvp_dump VIDEO --profile fast --out outputs/dumps_cpp/fast/STEM
python tools/regression/compare_dumps.py outputs/dumps/fast/STEM outputs/dumps_cpp/fast/STEM
```

## End-to-end fps (RTX 4070 Laptop, no recording for C++)

Python = frames / wall time of `dump_reference.py` (includes model loading and PNG/JSON writing, `--seg-every 30`);
C++ = `adasvp_dump --no-write` (seg every 30 frames). Python timings are noisy (shared GPU).

| profile | video | frames | Python fps | C++ fps |
|---|---|---|---|---|
| fast | sp1_crop_30fps | 4786 | 29.2 | 107.5 |
| fast | sp1_crop_2_30fps | 1889 | 62.0 | 124.9 |
| fast | sp1_crop_3_30fps | 2510 | 66.5 | 124.3 |
| fast | sp1_crop_4_30fps | 4751 | 73.0 | 125.3 |
| fast | red_crop_11_30fps | 750 | 24.3 | 65.7 |
| fast | red_crop_12_30fps | 660 | 24.2 | 66.4 |
| fast | red_crop_13_30fps | 751 | 24.6 | 66.4 |
| accurate | sp1_crop_30fps (30 s) | 899 | 11.2 | 11.9 |
| accurate | red_crop_11_30fps | 750 | 10.9 | 11.6 |
| accurate | red_crop_12_30fps | 660 | 11.1 | 11.6 |
| accurate | red_crop_13_30fps | 751 | 11.1 | 12.1 |

(Measured with `adasvp_dump` in an earlier run; the dry-run `video_demo` numbers of the event-level regression below are the current ones.)

## adas.jsonl schema

One JSON object per frame (same order as `frames.jsonl`), produced by driving the reference classes with the
reference defaults exactly like the main loop of `dashcam_vision.py` (lane update, then detection/track, lead,
FCW, LDW, RLR). numpy arrays are lists, `None` is `null`, floats keep full precision (shortest round-trip
repr); non-finite floats are the strings `"inf"`, `"-inf"`, `"nan"`. Boxes are `[x1,y1,x2,y2]`.

| key | content |
|---|---|
| `i`, `t` | frame index relative to `--start`; `t = (start_frame + i) / fps` (the `t_s` of the loop) |
| `lanes_in` | list passed to `EgoLaneTracker.update`: polylines `[[x,y],...]` |
| `ego` | tracker state after the update: `xl`, `xr` (40 values per row of `ys`, or null), `state` (`none`/`detected`/`partial`/`estimated`), `available`, `fade`, `since`, `obs_l`, `obs_r`, `glitch`, `free` `{l,r}`, `raw_l`, `raw_r` (arrays or null), `wprof`, `pend` `{l,r}` = `[array-or-null, count]` |
| `lead_in` | list `[track_id, cls, box]` passed to `LeadVehicleEstimator.update` (tracks with an id only) |
| `lead` | `[track_id, cls, box, dist_m]` or null |
| `fcw_lead_in` | the `lead_f` passed to `FCWMonitor.update` (same layout as `lead`; null when dropped for being outside the ego lane) |
| `fcw` | returned dict: `ttc`, `active`, `level`, `event` |
| `ldw` | returned dict `armed`, `side`, `event`, `d`, `v`, `tlc`, plus `mon`: `fail` (disarm reason or null), `m` (`y0`,`wres`,`resl`,`resr`,`sll`,`slr` as set by the last full check), `stats` (cumulative frames per reason, key `armed` when armed), `last_v`, `n_cross` |
| `tl_in` | one entry per class-9 box passed to `RLRMonitor.update`: `box`, `score`, `state` (`traffic_light_state` of that box on this frame, computed for every box; RLR itself only uses it when `box[1] > 0.01*H`) |
| `tl_tracks` | `TrafficLightTracker.tracks` after the update: `id`, `lost`, `done`, `n_hist`, `last` = last history entry `{t, box, conf, state}` (the full history is not dumped) |
| `tl_closed` | ids of the tracks closed (lost too long) by this frame's update |
| `rlr` | returned dict: `event` (null or `{id, cx, red_for, growth, box}`) |

`meta.json` gains `adas`: `fps_used` (the fps from the video container, used to size every frame window),
the defaults used, the derived windows (`ego_hold_frames`, `ego_persist`, `fcw_on_n`, `fcw_off_n`, `ldw_on_n`,
`ldw_off_n`, `ldw_row`, `lead_hold_frames`, `tl_max_lost`) and the tracker geometry (`ego_ys`, `ego_t_row`,
`ego_wmin`, `ego_wmax`, `ego_gate`).

## Event-level regression: C++ `video_demo` vs Python reference (17 videos x 2 profiles)

`run_cpp_regression.sh --videos-dir DIR --profile P --dry` (RTX 4070 Laptop, dry run = no encoding, `--no-seg`,
fp32 detector, the default at the time of these runs) against `tests/regression/reference/P/` (strict, +-0.5 s, type + level/side).
The counts are all alert events (FCW + LDW + RLR). "Python fps" = frames / wall time of the reference run in
`summary.txt` (model loading and the detections video included, shared GPU: noisy); "C++ fps" = the `total` fps
printed by `video_demo` at the end of the run (decode + lanes + detection + tracking + ADAS, no encoding).

### Fast profile (default fp32 detector)

| video | frames | ref events | C++ events | matched | borderline | Python fps | C++ fps | result |
|---|---|---|---|---|---|---|---|---|
| sp1_crop_30fps | 4786 | 4 | 3 | 3 | 1 | 21.3 | 124.4 | PASS (borderline) |
| sp1_crop_2_30fps | 1889 | 1 | 1 | 1 | 0 | 20.5 | 124.5 | PASS |
| sp1_crop_3_30fps | 2510 | 0 | 0 | 0 | 0 | 20.6 | 124.5 | PASS |
| sp1_crop_4_30fps | 4751 | 7 | 7 | 7 | 0 | 21.4 | 126.5 | PASS |
| red_crop_1_30fps | 600 | 0 | 0 | 0 | 0 | 19.4 | 126.7 | PASS |
| red_crop_2_30fps | 451 | 1 | 1 | 1 | 0 | 17.3 | 123.3 | PASS |
| red_crop_3_30fps | 421 | 1 | 1 | 1 | 0 | 16.2 | 123.3 | PASS |
| red_crop_4_30fps | 480 | 1 | 1 | 1 | 0 | 17.1 | 142.6 | PASS |
| red_crop_5_30fps | 2880 | 2 | 2 | 2 | 0 | 22.2 | 147.3 | PASS |
| red_crop_6_30fps | 601 | 1 | 1 | 1 | 0 | 18.8 | 141.4 | PASS |
| red_crop_7_30fps | 450 | 1 | 1 | 1 | 0 | 16.1 | 120.3 | PASS |
| red_crop_8_30fps | 360 | 1 | 1 | 1 | 0 | 18.9 | 158.4 | PASS |
| red_crop_9_30fps | 479 | 1 | 1 | 1 | 0 | 17.1 | 159.3 | PASS |
| red_crop_10_30fps | 390 | 1 | 1 | 1 | 0 | 17.7 | 161.0 | PASS |
| red_crop_11_30fps | 750 | 1 | 1 | 1 | 0 | 17.9 | 118.3 | PASS |
| red_crop_12_30fps | 660 | 1 | 1 | 1 | 0 | 17.4 | 121.0 | PASS |
| red_crop_13_30fps | 751 | 1 | 1 | 1 | 0 | 17.9 | 122.7 | PASS |

`fps (all 17 videos, frames-weighted): Python 20.1, C++ 129.2`


### Accurate profile (fp32 detector, record of the original runs; fp16 experiment in the last two columns)

| video | frames | ref events | C++ events | matched | borderline | Python fps | C++ fps (fp32) | result | C++ fps (fp16) | fp16 matched/ref |
|---|---|---|---|---|---|---|---|---|---|---|
| sp1_crop_30fps | 4786 | 5 | 5 | 5 | 0 | 11.2 | 20.4 | PASS | 48.0 | 5/5 |
| sp1_crop_2_30fps | 1889 | 1 | 1 | 1 | 0 | 11.0 | 20.4 | PASS | 48.2 | 1/1 |
| sp1_crop_3_30fps | 2510 | 0 | 0 | 0 | 0 | 11.0 | 20.2 | PASS | 47.6 | 0/0 |
| sp1_crop_4_30fps | 4751 | 7 | 7 | 7 | 0 | 11.4 | 20.1 | PASS | 47.9 | 7/7 |
| red_crop_1_30fps | 600 | 0 | 0 | 0 | 0 | 10.9 | 19.9 | PASS | 48.0 | 0/0 |
| red_crop_2_30fps | 451 | 1 | 1 | 1 | 0 | 9.8 | 20.4 | PASS | 47.6 | 1/1 |
| red_crop_3_30fps | 421 | 1 | 1 | 1 | 0 | 10.0 | 20.2 | PASS | 47.5 | 1/1 |
| red_crop_4_30fps | 480 | 1 | 1 | 1 | 0 | 10.4 | 20.5 | PASS | 50.9 | 1/1 |
| red_crop_5_30fps | 2880 | 2 | 2 | 2 | 0 | 15.6 | 20.6 | PASS | 51.3 | 2/2 |
| red_crop_6_30fps | 601 | 1 | 1 | 1 | 0 | 22.3 | 20.5 | PASS | 50.9 | 1/1 |
| red_crop_7_30fps | 450 | 1 | 1 | 1 | 0 | 19.6 | 19.9 | PASS | 46.8 | 1/1 |
| red_crop_8_30fps | 360 | 1 | 1 | 1 | 0 | 21.2 | 20.6 | PASS | 51.2 | 1/1 |
| red_crop_9_30fps | 479 | 1 | 1 | 1 | 0 | 21.8 | 20.7 | PASS | 51.0 | 1/1 |
| red_crop_10_30fps | 390 | 1 | 1 | 1 | 0 | 21.7 | 20.6 | PASS | 51.0 | 1/1 |
| red_crop_11_30fps | 750 | 1 | 1 | 1 | 0 | 21.4 | 19.9 | PASS | 47.6 | 1/1 |
| red_crop_12_30fps | 660 | 1 | 1 | 1 | 0 | 20.6 | 19.9 | PASS | 49.0 | 1/1 |
| red_crop_13_30fps | 751 | 1 | 1 | 1 | 0 | 21.5 | 19.9 | PASS | 49.5 | 1/1 |

`fps (all 17 videos, frames-weighted): Python 12.7, C++ fp32 20.3, C++ fp16 48.6`

Results: 33 of 34 pairs are identical event by event (tolerance +-0.5 s, nothing loosened). The only strict failure
is the fast profile `sp1_crop_30fps` (FCW warning @ 127.96 s, frame 3835, missing in C++), classified BORDERLINE
with `compare_dumps.py --adas` on the per-frame traces (regenerated with the current C++ build): the smoothed TTC
minimum around the event is 2.664 s in Python and 2.659 s in C++ (not fired), against the 2.7 s threshold; the
rule is `ttc <= 1.02 x threshold` (2.754 s) within +-0.5 s. All other events of that video (FCW @ 4656, LDW left
@ 4593, LDW right @ 4749) match at dt 0.000 s. The accurate profile matches all 17 videos strictly, with no
borderline case. Per-video outputs of the runs: `outputs/cpp_regression/<profile>[_fp16]/` (gitignored).

### fp16 detector (accurate profile, `--det-precision fp16`)

All 17 pairs pass strictly (17/17 identical events, no borderline), with 48.6 fps against 20.3 fps for fp32 in the
dry run (2.4x; the fp16 TensorRT engine of yolo26x takes about 10 min to build the first time, then it is cached).
Decision: the accurate profile now defaults to the fp16 detector engine (`Config::accurate()`), based on this
experiment; fp32 stays available with the hidden `--det-precision fp32`. The fp32 table above is kept as the record
of the original runs, and the reference CSVs were not regenerated.
Recording the accurate-profile videos (`--no-seg`, with encoding) ran at 11.5-13 fps in fp32 because Python
dumps shared the GPU at the same time; the dry-run numbers above are the reference.
