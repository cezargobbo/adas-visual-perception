# Golden alerts: notes

`fast/` and `accurate/` hold the alerts CSVs (`<video>_alerts.csv`) of the C++ `video_demo` on the 17 videos of
`tools/regression/videos.txt`. They are the regression baseline of the C++: `tools/regression/run_cpp_regression.sh`
(default `--against golden`) compares a fresh run with them, strictly (+-0.5 s, type and level or side), and expects
every pair identical.

## How they were made

`tools/regression/make_golden.sh --videos-dir DIR --profile fast|accurate`, which runs
`video_demo --dry --no-seg --profile P` with the profile defaults (the accurate profile uses the fp16 detector) and
copies only the CSVs. Machine: RTX 4070 Laptop GPU, TensorRT 10.16.1.11.

## Determinism

After generating them, `run_cpp_regression.sh` was run against them for both profiles. That second, independent run
gave 17 of 17 identical pairs in each profile (fast about 130 fps, accurate about 50 fps, frames-weighted).

## Relation to the Python prototype

Both profiles have the same event counts as `tests/regression/python/` (the alerts of the original Python
prototype), except the fast profile on `sp1_crop_30fps`: 3 events here and 4 there. The missing one is the borderline
FCW warning at 127.96 s documented in `tools/regression/README.md`.

## Changing them

A change of these files is a change of the behaviour that the project guards. Make it on purpose, review the diff of
the events, and commit it on its own with the reason (for example a new model, a changed threshold or a TensorRT
upgrade).
