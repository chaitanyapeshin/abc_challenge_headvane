# PyTorch weights (optional, for transparency)

The submitted model is `../model_fp32.onnx`; these files are not needed for inference.

* `transformer_best.pt`
* `config.json`
* `metrics.json`
* `train_transformer.py`

Rebuild the network with the model class in the training script and `config.json` (`model_kwargs`, context 100), then `load_state_dict(torch.load(<checkpoint>))`.
