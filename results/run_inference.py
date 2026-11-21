"""run_inference.py — HeadVane head-pose prediction, 9th ABC Challenge.

Interface (as required by the challenge):
    load_model(model_dir: str) -> object
    predict(model, context: np.ndarray) -> np.ndarray
        context    : np.ndarray, float32, shape [400, 4] — unit quaternions, scalar-first (w, x, y, z),
                     oldest sample first, 100 Hz
        prediction : np.ndarray, float32, shape [20, 4] — the next 20 orientations (w, x, y, z), 10 ms apart

Standalone:
    python run_inference.py --model_dir . --input sample.npy --output pred.npy

The model is stateless: every call reads the last 100 samples of the context and keeps nothing between calls
or sessions, so no state needs resetting. It runs offline (no network access). onnxruntime uses the CPU by
default; set HEADPOSE_PROVIDER=cuda to use CUDAExecutionProvider when onnxruntime-gpu is installed.
model_fp32.onnx is loaded by default; set HEADPOSE_MODEL_FILE=model_int8.onnx to use the optional INT8 variant.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort


def load_model(model_dir: str) -> object:
    """Load model_fp32.onnx (default) from model_dir and return an onnxruntime InferenceSession."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    if os.environ.get("HEADPOSE_PROVIDER", "cpu").lower() == "cuda" and \
            "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    model_file = os.environ.get("HEADPOSE_MODEL_FILE", "model_fp32.onnx")      # model_int8.onnx for the INT8 variant
    return ort.InferenceSession(str(Path(model_dir) / model_file), opts, providers=providers)


def predict(model, context: np.ndarray) -> np.ndarray:
    """context float32 [400, 4] (w, x, y, z) -> prediction float32 [20, 4] (w, x, y, z)."""
    ctx = np.array(context, dtype=np.float32, copy=True)            # never modifies the caller's array
    if ctx.ndim == 3 and ctx.shape[0] == 1:                          # tolerate a leading batch axis
        ctx = ctx[0]
    if ctx.shape != (400, 4):
        raise ValueError(f"context must have shape [400, 4] (w, x, y, z); got {list(ctx.shape)}")
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ctx = np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(ctx, axis=1, keepdims=True)
    ctx = np.where(norm > 1e-6, ctx / np.maximum(norm, 1e-6), identity).astype(np.float32)
    out = model.run(None, {"context": ctx[None]})[0][0]              # [1, 20, 4] -> [20, 4]
    out = np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    out = np.where(norm > 1e-6, out / np.maximum(norm, 1e-6), identity)
    return np.ascontiguousarray(out, dtype=np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict the next 20 head orientations from 400 past ones.")
    ap.add_argument("--model_dir", default=".", help="folder containing model_fp32.onnx")
    ap.add_argument("--input", required=True, help=".npy file, float32 [400, 4] quaternions (w, x, y, z)")
    ap.add_argument("--output", required=True, help=".npy file to write, float32 [20, 4] quaternions (w, x, y, z)")
    a = ap.parse_args()
    np.save(a.output, predict(load_model(a.model_dir), np.load(a.input)))
