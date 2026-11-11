# Models

Model weights are not stored in this repository (see `.gitignore`). This table lists the origin, license and credit for each model and dataset used.

| Model / dataset | Origin | License | Notes |
|---|---|---|---|
| YOLO26 (detection/tracking) | [Ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 | |
| SegFormer B1/B5 Cityscapes (segmentation) | NVIDIA, [model page](https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024) | [NVIDIA Source Code License](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) | Non-commercial use only |
| Polar R-CNN DLA34 CurveLanes (lane detection) | [ShqWW/PolarRCNN](https://github.com/ShqWW/PolarRCNN); paper "Polar R-CNN: End-to-End Lane Detection with Fewer Anchors" (arXiv 2411.01499) | No license declared by the authors | Redistributed for non-commercial academic use with credit |
| CurveLanes (training dataset) | [SoulmateB/CurveLanes](https://github.com/SoulmateB/CurveLanes) | Non-commercial | Training data for Polar R-CNN |
| Cityscapes (training dataset) | [cityscapes-dataset.com](https://www.cityscapes-dataset.com) | Non-commercial | Training data for SegFormer |

Note: Polar R-CNN: no license declared by the authors; redistributed for non-commercial academic use with credit.
