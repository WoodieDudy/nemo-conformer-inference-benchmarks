# Issue with TensorRT dynamic shapes speed

## Issue:
When I run my model using TensorRT engine builded with dynamic shapes, it runs 5 times slower then with static shapes

## Repro:
Run trtexec benckmarks  
Building logs are [there](https://drive.google.com/drive/folders/1bueENdi3pv5eOgm2G27MQlIzHaNelFf7?usp=share_link)
  
For static shapes
```bash
trtexec --onnx=models/static_model.onnx --verbose --fp16
```
For dynamic shapes
```bash
trtexec --onnx=models/dynamic_model.onnx --minShapes=x:1x4000,xlen:1 --optShapes=x:64x40320,xlen:64 --maxShapes=x:64x300000,xlen:64 --shapes=x:16x40320,xlen:16 --verbose --fp16
```
[Weigths](https://drive.google.com/drive/folders/1XjLheCUHxmOf18lZ5tkDhB9uaPV2mOLe?usp=sharing) 

## My output:
For static shapes:
```
[I] === Performance summary ===
[I] Throughput: 65.7372 qps
[I] Latency: min = 15.1062 ms, max = 45.0152 ms, mean = 27.5067 ms, median = 29.2732 ms, percentile(90%) = 29.739 ms, percentile(95%) = 30.0433 ms, percentile(99%) = 30.3047 ms
[I] Enqueue Time: min = 7.62402 ms, max = 16.1389 ms, mean = 9.51405 ms, median = 9.28607 ms, percentile(90%) = 10.0674 ms, percentile(95%) = 10.4929 ms, percentile(99%) = 12.9241 ms
[I] H2D Latency: min = 0.119141 ms, max = 24.7606 ms, mean = 12.4057 ms, median = 14.269 ms, percentile(90%) = 14.4882 ms, percentile(95%) = 14.7051 ms, percentile(99%) = 14.7982 ms
[I] GPU Compute Time: min = 14.9122 ms, max = 20.2374 ms, mean = 15.0811 ms, median = 14.9981 ms, percentile(90%) = 15.3064 ms, percentile(95%) = 15.4503 ms, percentile(99%) = 15.5051 ms
[I] D2H Latency: min = 0.0163574 ms, max = 0.0310059 ms, mean = 0.0199066 ms, median = 0.0185547 ms, percentile(90%) = 0.026123 ms, percentile(95%) = 0.0284424 ms, percentile(99%) = 0.0297852 ms
[I] Total Host Walltime: 3.05763 s
[I] Total GPU Compute Time: 3.0313 s
```
For dynamic shapes:
```
[I] === Performance summary ===
[I] Throughput: 11.7297 qps
[I] Latency: min = 84.5625 ms, max = 95.9257 ms, mean = 85.2279 ms, median = 84.7446 ms, percentile(90%) = 86.5833 ms, percentile(95%) = 87.4399 ms, percentile(99%) = 95.9257 ms
[I] Enqueue Time: min = 84.3406 ms, max = 97.8796 ms, mean = 85.0945 ms, median = 84.5563 ms, percentile(90%) = 86.5012 ms, percentile(95%) = 87.3152 ms, percentile(99%) = 97.8796 ms
[I] H2D Latency: min = 0.111816 ms, max = 0.185974 ms, mean = 0.122898 ms, median = 0.112305 ms, percentile(90%) = 0.164673 ms, percentile(95%) = 0.168701 ms, percentile(99%) = 0.185974 ms
[I] GPU Compute Time: min = 84.4336 ms, max = 95.7728 ms, mean = 85.0862 ms, median = 84.6118 ms, percentile(90%) = 86.4541 ms, percentile(95%) = 87.3076 ms, percentile(99%) = 95.7728 ms
[I] D2H Latency: min = 0.0163574 ms, max = 0.0278931 ms, mean = 0.0188161 ms, median = 0.0177002 ms, percentile(90%) = 0.0250244 ms, percentile(95%) = 0.0269775 ms, percentile(99%) = 0.0278931 ms
[I] Total Host Walltime: 3.1544 s
[I] Total GPU Compute Time: 3.14819 s
```

# [Issue](https://github.com/NVIDIA/TensorRT/issues/3364)
# [Weigths](https://drive.google.com/drive/folders/1XjLheCUHxmOf18lZ5tkDhB9uaPV2mOLe?usp=sharing) 

# Env
```
Image: nvcr.io/nvidia/pytorch:23.07-py3
CUDA Version: 12.1
GPU: A100
TensorRT: 8.6.1.6
```