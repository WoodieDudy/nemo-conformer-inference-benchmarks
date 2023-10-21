import math

import torch
import torch.utils.benchmark as tbenchmark

import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import onnxruntime as ort

import pycuda.autoinit

ALPHABET_LEN = 35
DEVICE_ID = 0


def calc_time_dim(T: int):
    sample_rate = 8000
    window_size = 0.04
    window_stride = 0.01

    win_length = int(window_size * sample_rate)
    hop_length = int(window_stride * sample_rate)
    nfft = 2 ** math.ceil(math.log2(win_length))
    freq_cutoff = nfft // 2 + 1
    pad = freq_cutoff - 1
    T_prime = (T + 2 * pad - nfft + hop_length) // hop_length
    return (T_prime + 3) // 4


class TensorRTWrapper:
    def __init__(self, engine_path: str):
        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.trt_ctx = self.engine.create_execution_context()

    def prepocess_inputs(self, x: np.ndarray, xlen: np.ndarray, output_shape: tuple):
        self.x = x
        self.output_shape = output_shape

        self.trt_ctx.set_input_shape("x", x.shape)
        self.trt_ctx.set_input_shape("xlen", xlen.shape)

        self.input_bindings = [
            cuda.mem_alloc(int(np.prod(x.shape)) * x.dtype.itemsize),
            cuda.mem_alloc(int(np.prod(xlen.shape)) * xlen.dtype.itemsize)
        ]
        self.output_binding = cuda.mem_alloc((int(np.prod(output_shape)) * np.float32().itemsize))
        cuda.memcpy_htod(self.input_bindings[0], x.ravel())
        cuda.memcpy_htod(self.input_bindings[1], xlen.ravel())

    def run(self):
        self.trt_ctx.execute_v2(bindings=list(map(int, self.input_bindings + [self.output_binding])))
     
    def get_outputs(self):
        h_output = np.ones(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(h_output, self.output_binding)
        return h_output

class ORTWrapper:
    def __init__(self, onnx_path: str):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.ort_session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"], sess_options=sess_options)
        self.bindings = self.ort_session.io_binding()

    def prepocess_inputs(self, x: np.ndarray, xlen: np.ndarray, output_shape: tuple):
        device = "cuda"

        inputs = [
            ("x", x),
            ("xlen", xlen)
        ]

        for name, input_array in inputs:
            tensor = torch.from_numpy(input_array).to(f"{device}:{DEVICE_ID}")
            self.bindings.bind_input(
                name=name,
                device_type=device,
                device_id=DEVICE_ID,
                element_type=input_array.dtype,
                shape=input_array.shape,
                buffer_ptr=tensor.data_ptr(),
            )

        # for output in self.ort_session.get_outputs():
        #     print(output.name)
        #     print(output.type)
        self.output_tensor = torch.empty(output_shape, dtype=torch.float32, device=f"{device}:{DEVICE_ID}")
        self.bindings.bind_output(
            name="logits",
            device_type=device,
            device_id=DEVICE_ID,
            element_type=np.float32,
            shape=output_shape,
            buffer_ptr=self.output_tensor.data_ptr(),
        )

    def run(self):
        self.ort_session.run_with_iobinding(self.bindings)
    
    def get_outputs(self):
        return self.output_tensor.detach().cpu().numpy()


def main():
    x = np.random.rand(16, 40320).astype(np.float32)
    xlen = np.random.rand(16).astype(np.float32)
    batch, time = x.shape
    output_shape = (batch, ALPHABET_LEN, calc_time_dim(time))
    print()

    # trt_wrapper = TensorRTWrapper("model.onnx.engine")
    trt_wrapper = TensorRTWrapper("dynamic_model.onnx.engine")
    # trt_wrapper = TensorRTWrapper("engine/full_conformer_fp16_opset17_staticshapes_40320.onnx.engine")

    trt_wrapper.prepocess_inputs(x, xlen, output_shape)
    t = tbenchmark.Timer(
        stmt='model.run()',
        globals={"model": trt_wrapper}
    )
    m = t.blocked_autorange(min_run_time=2)
    trt_output = trt_wrapper.get_outputs()
    trt_mean_time = m.mean
    print(f"{trt_mean_time=}")

    # ort_wrapper = ORTWrapper("model.onnx")
    ort_wrapper = ORTWrapper("dynamic_model.onnx")
    ort_wrapper.prepocess_inputs(x, xlen, output_shape)
    t = tbenchmark.Timer(
        stmt='model.run()',
        globals={"model": ort_wrapper}
    )
    m = t.blocked_autorange(min_run_time=2)
    ort_output = ort_wrapper.get_outputs()
    ort_mean_time = m.mean
    print(f"{ort_mean_time=}")

    print(f"TRT runs {ort_mean_time/trt_mean_time} times faster than ORT")

    diff = np.abs(trt_output - ort_output)
    print(f"Mean difference: {diff.mean()}")
    print(f"Max difference: {diff.max()}")


if __name__ == '__main__':
    main()
