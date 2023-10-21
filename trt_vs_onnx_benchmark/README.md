# Issue with TensorRT dynamic shapes speed

## Issue:
When I run my model using TensorRT engine builded with dynamic shapes, it runs 5 times slower then using ORT

## Repro:
1. Take models from [there](https://drive.google.com/drive/folders/1XjLheCUHxmOf18lZ5tkDhB9uaPV2mOLe?usp=sharing)  
    Or export them by
    ```bash
    python export_model_to_onnx.py
    ```
    Comment or uncomment this line in script to export model to onnx with static or dynamic shapes
    ```python
    dynamic_axes={"x": {0: "B", 1: "T"}, "logits": {0: "B", 2: "t"}, "xlen": {0: "B"}},
    ```
2. Build TensorRT engines  
    For static shapes
    ```bash
    trtexec --nvtxMode=verbose --buildOnly --workspace=8192 --onnx=static_model.onnx --saveEngine=engine/static_model.onnx.engine --fp16
    ```
    For dynamic shapes
    ```bash
    trtexec --nvtxMode=verbose --buildOnly --workspace=8192 --onnx=dynamic_model.onnx --saveEngine=dynamic_model.onnx.engine --fp16 --minShapes=x:1x4000,xlen:1 --optShapes=x:64x40320,xlen:64 --maxShapes=x:64x300000,xlen:64
    ```

3. Run benchmark  
    Replace models paths in this lines on yours
    ```python
    trt_wrapper = TensorRTWrapper("dynamic_model.onnx.engine")

    ort_wrapper = ORTWrapper("dynamic_model.onnx")
    ```

    Run benchmark
    ```bash
    python trt_vs_onnx_benchmark.py
    ```

    ## My output:
    For static shapes:
    ```
    trt_mean_time=0.015416952169415219
    ort_mean_time=0.023787834812195367
    TRT runs 1.5429661161812935 times faster than ORT
    Mean difference: 0.0018365184077993035
    Max difference: 0.009765625
    ```
    For dynamic shapes:
    ```
    trt_mean_time=0.0932643054098166
    ort_mean_time=0.02416586849441561
    TRT runs 0.25911165464887503 times faster than ORT
    Mean difference: 0.0051328823901712894
    Max difference: 1.052978515625
    ```

# [Issue](https://github.com/NVIDIA/TensorRT/issues/3364)
# [Weigths](https://drive.google.com/drive/folders/1XjLheCUHxmOf18lZ5tkDhB9uaPV2mOLe?usp=sharing) 

# Env
```
Image: nvcr.io/nvidia/pytorch:23.07-py3
CUDA Version: 12.1
GPU: A100
TensorRT: 8.6.1.6

tensorrt==8.6.1
torch-tensorrt==1.5.0.dev0
torch==2.0.1
onnx==1.14.0
onnxruntime-gpu==1.16.1
```