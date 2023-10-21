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

4. Cast model to fp32 instead fp16
```python
# replace this
model = InputOutputTypeCast(model.to(torch.float16), dtype=torch.float16)
# on this
model = InputOutputTypeCast(model.to(torch.float32), dtype=torch.float32)
```

Output:
```
pytorch with ort max difference:  tensor(4.0938, device='cuda:0')
pytorch with ort mean difference:  tensor(0.0589, device='cuda:0')
```
P.S. Max diff is still big because of some bug in my model code. It doesn't matter

# [Issue](https://github.com/microsoft/onnxruntime/issues/17689)
# [Weigths](https://drive.google.com/drive/folders/1knactAG-JoTqSjwhXbCDnidrNnB58l0D?usp=share_link) 
Script `compare_torch_vs_ort_outputs.py` already has onnx export, so `my_model.onnx` is unnecessary

# Env
```
Image: nvcr.io/nvidia/pytorch:23.07-py3
GPU: A100
CUDA Version: 12.1

torch==2.0.1
onnx==1.14.0
onnxruntime-gpu==1.16.1
```