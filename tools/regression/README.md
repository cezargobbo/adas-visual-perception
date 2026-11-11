# Regression tools

- `dump_reference.py`: dumps per-frame results (`frames.jsonl`, `seg/NNNNNN.png`, `meta.json`) of the Python
  reference pipeline.
- `adasvp_dump` (built from `tools/cpp/adasvp_dump.cpp`): writes the same format from the C++ `Pipeline`.
  `--no-write` skips all output (used for the fps numbers below).
- `compare_dumps.py REF_DIR CPP_DIR`: prints and gates one video (exit 0 only if every gate passes):
  - dets: recall and precision >= 0.97 (IoU >= 0.9, same class);
  - tracks: id consistency >= 0.95 (majority C++ id -> Python id mapping over IoU-matched tracks);
  - lanes: equal count on >= 98% of frames, mean |dx| <= 2 px, point-count difference <= 1;
  - seg: >= 99% equal pixels in every PNG.
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
