# Model card — HeadVane: Transformer head-pose predictor

**Task.** 400 past head orientations (unit quaternions, w,x,y,z, 100 Hz) → next 20 orientations (200 ms).

## Architecture
Inputs: the last 100 orientations; the encoder sees per-step features that do not depend on the player's heading: head-frame angular velocity (from consecutive quaternions) and the world 'up' direction seen from the head.
Encoder: transformer encoder (pre-norm blocks, multi-head attention) over 50 ms patch tokens of motion features (angular velocity, tilt, speed, angular acceleration) with learned recency embeddings and a readout token initialised from the newest sample.
Output: 20 rotation vectors relative to the last orientation = learned per-horizon multiple of the last
angular velocity + network correction, composed onto the last orientation and unit-normalised.

| item | value |
|---|---|
| trainable parameters | 121,892 |
| MACs per prediction (one call, 100-sample context) | 2,202,344 |
| context length used | 100 samples (the last 100 of the 400 provided) |
| state tensors | none (stateless: every call reads the context window) |
| quantisation / pruning / distillation | none (FP32) |

File: `model_fp32.onnx` — input `context` [1, 400, 4] → output `prediction` [1, 20, 4].
ONNX operators: Add, Concat, Constant, ConstantOfShape, Conv, Div, Equal, Erf, Expand, Gather, Gemm, LayerNormalization, MatMul, Mul, Pow, ReduceL2, ReduceSum, Reshape, ScatterND, Slice, Softmax, Split, Sqrt, Squeeze, Sub, Transpose, Unsqueeze, Where.

## Training data and preprocessing
BOXRR-23 (Nair et al., IEEE TVCG 2024), Beat Saber head tracking, filtered and cleaned by the team's
preprocessing (challenge filters; unit quaternions; hemisphere-aligned; 100 Hz grid; gaps split, not
interpolated). Users split 80/10/10 (train/val/test) by a hash of the user ID; no user appears in two splits.
10,677 training users (1079 h, random crops),
1,357 validation users (checkpoint selection), 1,330
test users (reported once). Loss: mean geodesic error at the 20th predicted sample
(+0.1 × mean over all horizons). AdamW, one-cycle LR 0.001, batch 256, 40000 steps.

## Test results (held-out test users, 46,474 windows)
| model | mean error @20th sample (°) | median (°) | p95 (°) |
|---|---|---|---|
| constant velocity | 7.479 | 5.709 | 19.193 |
| last observation | 6.618 | 5.094 | 16.815 |
| damped CV | 5.584 | 4.372 | 13.948 |
| **Transformer (this model)** | **4.517** | 3.607 | 11.093 |

CPU latency (onnxruntime, 1 thread): 0.297 ms per call.

## Known failure modes
* Largest errors at motion onsets and abrupt reversals (see by_regime_test.csv): nothing in the past
  1.0 s announces a movement that has not started.
* Trained and validated on Beat Saber only; other applications may move differently.
* Assumes clean, uniformly sampled 100 Hz input; gaps or duplicated samples appear as sudden velocity changes.

## Submission summary

| item | value |
|---|---|
| submitted model | `model_fp32.onnx` (input `context` float32 [1, 400, 4] → output `prediction` float32 [1, 20, 4]) |
| trainable parameters | 121,892 |
| MACs per streaming step | 2,202,344 (one stateless call per new sample) |
| context length | 100 samples |
| state tensors | none (stateless) |
| FP32 latency (desktop CPU, 1 thread) | 0.231 ms per step |
| optional INT8 (`model_int8.onnx`) | dynamic, per-channel INT8 quantisation of MatMul/Gemm/Conv; test error 4.518° (FP32 4.517°), 0.225 ms per step, 231 kB |
| pruning / distillation | none |
