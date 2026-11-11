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
- `make_reference.sh`: regenerates the Python reference alerts.

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
