# SkipLayerNormFusion optimizer bug

## Repro:
If run `compare_torch_vs_ort_outputs.py` with `ort.GraphOptimizationLevel.ORT_ENABLE_ALL` then the error when running a model through ort is very large
```bash
python compare_torch_vs_ort_outputs.py
```
Outputs:
```
pytorch with ort max difference:  tensor(22.4375, device='cuda:0')
pytorch with ort mean difference:  tensor(2.0066, device='cuda:0')
```

## Ways to fix it:
1. Enable only basic graph optimizations
```python
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
```
2. Disable `SkipLayerNormFusion` by `disabled_optimizers`
```python
onnxruntime_session = ort.InferenceSession(
    onnx_model_path,
    providers=["CUDAExecutionProvider"],
    disabled_optimizers=['SkipLayerNormFusion']
)
```

3. Disable `SkipLayerNormFusion` by `provider_options`
```python
cuda_provider_options = {"enable_skip_layer_norm_strict_mode": True}
onnxruntime_session = ort.InferenceSession(
    onnx_model_path,
    provider_options={"CUDAExecutionProvider": cuda_provider_options},
    sess_options=sess_options,
```

Output:
```
pytorch with ort max difference:  tensor(4.0938, device='cuda:0')
pytorch with ort mean difference:  tensor(0.0589, device='cuda:0')
```
P.S. Max diff is still big because of some bug in my model code. It doesn't matter