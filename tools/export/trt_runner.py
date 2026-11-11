"""TensorRT (pip wheel) runner used to validate ONNX models before the C++ port.

Engines built here come from the pip ``tensorrt`` package and are cached in
``cache/trt-py/`` (git-ignored). They are never mixed with the engines that the
C++ build produces with ``trtexec``.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache" / "trt-py"

YOLO_MIN = (1, 3, 320, 320)
YOLO_MAX = (1, 3, 1280, 1280)
YOLO_OPT = {"s": (1, 3, 544, 960), "x": (1, 3, 736, 1280)}

_LOGGER = trt.Logger(trt.Logger.WARNING)


def _sha12(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _torch_dtype(t: "trt.DataType") -> torch.dtype:
    return {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
        trt.int8: torch.int8,
    }[t]


class TrtModel:
    """Build (or load from cache) a TensorRT engine for an ONNX file and run it.

    ``profiles`` maps an input name to (min, opt, max) shapes; it is only needed for
    inputs with dynamic dimensions.
    """

    def __init__(self, onnx_path, fp16: bool, cache_dir=CACHE_DIR, profiles=None, workspace_gb: float = 4.0):
        self.onnx_path = Path(onnx_path)
        self.fp16 = fp16
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        key = f"{_sha12(self.onnx_path)}-{'fp16' if fp16 else 'fp32'}-trt{trt.__version__}"
        plan = cache / f"{self.onnx_path.stem}-{key}.engine"
        self.build_seconds = 0.0
        if not plan.exists():
            t0 = time.time()
            self._build(plan, profiles, workspace_gb)
            self.build_seconds = time.time() - t0
        runtime = trt.Runtime(_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(plan.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {plan}")
        self.context = self.engine.create_execution_context()
        self.inputs = [n for n in self._names() if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self._names() if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.stream = torch.cuda.Stream()

    def _names(self):
        return [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]

    def _build(self, plan: Path, profiles, workspace_gb: float) -> None:
        builder = trt.Builder(_LOGGER)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, _LOGGER)
        if not parser.parse_from_file(str(self.onnx_path)):
            errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT cannot parse {self.onnx_path}:\n{errs}")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
        if self.fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        dynamic = False
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            if any(d < 0 for d in inp.shape):
                dynamic = True
                if not profiles or inp.name not in profiles:
                    raise ValueError(f"dynamic input {inp.name} needs a profile")
                mn, opt, mx = profiles[inp.name]
                profile.set_shape(inp.name, mn, opt, mx)
        if dynamic:
            config.add_optimization_profile(profile)
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise RuntimeError(f"TensorRT failed to build an engine for {self.onnx_path}")
        plan.write_bytes(bytes(blob))

    def __call__(self, **inputs: np.ndarray) -> dict[str, np.ndarray]:
        keep = []
        for name in self.inputs:
            arr = np.ascontiguousarray(inputs[name], dtype=np.float32)
            t = torch.from_numpy(arr).to("cuda", dtype=_torch_dtype(self.engine.get_tensor_dtype(name)))
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())
            keep.append(t)
        outs = {}
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"output {name} has unresolved shape {shape}")
            o = torch.empty(shape, dtype=_torch_dtype(self.engine.get_tensor_dtype(name)), device="cuda")
            self.context.set_tensor_address(name, o.data_ptr())
            outs[name] = o
        self.stream.wait_stream(torch.cuda.current_stream())
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("execute_async_v3 failed")
        self.stream.synchronize()
        return {k: v.float().cpu().numpy() for k, v in outs.items()}


def load_runner(model: str, fp16: bool = True):
    """Return the callable the validate_*.py scripts expect.

    model: "s"/"x" (YOLO26), "b1"/"b5" (SegFormer) or "polar".
    YOLO/SegFormer runners return the first output array; polar returns [pts, endp, o2o].
    """
    models = ROOT / "models"
    if model in ("s", "x"):
        path = models / f"yolo26{model}.onnx"
        m = TrtModel(path, fp16, profiles={"images": (YOLO_MIN, YOLO_OPT[model], YOLO_MAX)})
        return lambda x: m(images=x)[m.outputs[0]]
    if model in ("b1", "b5"):
        from export_segformer import onnx_path

        m = TrtModel(onnx_path(model, models), fp16)
        return lambda x: m(pixel_values=x)[m.outputs[0]]
    if model == "polar":
        from export_polar import onnx_path

        m = TrtModel(onnx_path(models), fp16)
        return lambda x: [m(image=x)[n] for n in ("pts", "endp", "o2o")]
    raise ValueError(f"unknown model {model!r}")
