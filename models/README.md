# Models

Model weights are not stored in this repository (see `.gitignore`).

## Get the models

    scripts/download_models.sh [--profile fast|accurate|all] [--dest DIR]

The script downloads the ONNX models from the GitHub release into `models/` (or `--dest`), checks each sha256
and only then gives the file its final name. `video_demo` reads them from `models/`; use `--models-dir DIR` for
another directory.

| File | Profile | Size |
|---|---|---|
| `yolo26s.onnx` | fast | 38.8 MB |
| `segformer-b1-cityscapes-1024x576.onnx` | fast | 54.9 MB |
| `yolo26x.onnx` | accurate | 224.2 MB |
| `segformer-b5-cityscapes-1280x720.onnx` | accurate | 339.3 MB |
| `polarrcnn_curvelanes_dla34.onnx` | fast and accurate | 64.4 MB |

`--profile fast` downloads about 158 MB, `accurate` about 628 MB and `all` (the default) about 722 MB.

To export the models yourself from the original weights, run `tools/setup_models.py` and the scripts in
`tools/export/` (see [docs/install.md](../docs/install.md) and [tools/export/README.md](../tools/export/README.md)).

## Licenses and credits

This table lists the origin, license and credit for each model and dataset used.

| Model / dataset | Origin | License | Notes |
|---|---|---|---|
| YOLO26 (detection/tracking) | [Ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 | |
| SegFormer B1/B5 Cityscapes (segmentation) | NVIDIA, [model page](https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024) | [NVIDIA Source Code License](https://github.com/NVlabs/SegFormer/blob/master/LICENSE), non-commercial use only | |
| Polar R-CNN DLA34 CurveLanes (lane detection) | [ShqWW/PolarRCNN](https://github.com/ShqWW/PolarRCNN); paper "Polar R-CNN: End-to-End Lane Detection with Fewer Anchors" (arXiv 2411.01499) | No license declared by the authors | Redistributed for non-commercial academic use with credit |
| CurveLanes (training dataset) | [SoulmateB/CurveLanes](https://github.com/SoulmateB/CurveLanes) | Non-commercial | Training data for Polar R-CNN |
| Cityscapes (training dataset) | [cityscapes-dataset.com](https://www.cityscapes-dataset.com) | Non-commercial | Training data for SegFormer |

Note: Polar R-CNN: no license declared by the authors; redistributed for non-commercial academic use with credit.
