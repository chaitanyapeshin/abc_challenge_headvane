# HeadVane — head-pose prediction (9th ABC Challenge)

Predicts the next 20 head orientations (200 ms at 100 Hz) from the last 400 (4 s), for any user, without fine-tuning.

## Install (standard Linux machine, CPU or CUDA)
```bash
pip install -r requirements.txt            # numpy + onnxruntime (CPU); onnxruntime-gpu also works
```

## Run
```bash
python run_inference.py --model_dir . --input sample.npy --output pred.npy
```
```python
from run_inference import load_model, predict
model = load_model(".")                      # loads model_fp32.onnx
prediction = predict(model, context)         # context: float32 [400, 4] (w, x, y, z), oldest first
                                             # prediction: float32 [20, 4] (w, x, y, z), next 20 samples
```
* Quaternion order is scalar-first (w, x, y, z) for input and output; outputs are unit quaternions.
* Stateless: each call uses only the context passed in (the last 100 samples); nothing is carried between calls,
  so no state needs to be reset between sessions or users.
* No network access is needed or used. Set `HEADPOSE_PROVIDER=cuda` to run on the GPU with onnxruntime-gpu.

## Files
* `model_card.md`
* `model_fp32.onnx`
* `model_int8.onnx`
* `requirements.txt`
* `run_inference.py`
* `weights/README.md`
* `weights/config.json`
* `weights/metrics.json`
* `weights/train_transformer.py`
* `weights/transformer_best.pt`

* `model_fp32.onnx` is the submitted model; `model_int8.onnx` is an optional INT8-quantised variant with the
  same interface (use it with `HEADPOSE_MODEL_FILE=model_int8.onnx`); `weights/` holds the PyTorch checkpoint.

Details, metrics and limitations: `model_card.md`.
