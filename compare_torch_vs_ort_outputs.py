import onnxruntime as ort
import torch
from torch import nn, Tensor

from conformer import SpeechRecognitionModel


torch.manual_seed(1337)


class InputOutputTypeCast(nn.Module):
    def __init__(self, model: nn.Module, dtype: torch.dtype) -> None:
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, x: Tensor, xlen: Tensor) -> dict[str, Tensor]:
        return self.model(x.to(self.dtype), xlen).to(x.dtype)


class OnnxWrapper(nn.Module):
    def __init__(
        self,
        onnx_model_path: str,
        providers: str = ["CUDAExecutionProvider", "CPUExecutionProvider"],
        severity=None,
    ) -> None:
        super().__init__()

        sess_options = ort.SessionOptions()

        # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_provider_options = {"enable_skip_layer_norm_strict_mode": True} 
        self.onnxruntime_session = ort.InferenceSession(
            onnx_model_path,
            providers=providers,
            # provider_options = {"CUDAExecutionProvider": cuda_provider_options},
            sess_options=sess_options,
            disabled_optimizers=[
            #     'QDQS8ToU8Transformer',
            #     'QDQSelectorActionTransformer',
            #     'ConvActivationFusion',
            #     'GeluFusion',
            #     'LayerNormFusion',
            #     'SimplifiedLayerNormFusion',
            #     'AttentionFusion',
            #     'EmbedLayerNormFusion',
            #     'GatherToSplitFusion',
            #     'GatherToSliceFusion',
            #     'MatmulTransposeFusion',
            #     'BiasGeluFusion',
            #     'FastGeluFusion',
            #     'GeluApproximation'
            #     'SkipLayerNormFusion',
            #     'QuickGeluFusion',
            ]
        )
        self.dict_adapter = dict

        if severity is not None:
            ort.set_default_logger_severity(severity)

    def forward(self, x, xlen):
        """Forward method for encoder.
        @param x: (B, C, T)
        @type x: FloatTensor
        @param xlen: (B)
        @type xlen: Tensor, range - [0, 1]
        @return: logits - (B, C, T), log_probs - (B, C, T), olen - (B), uncertainty - (B)
        @rtype: FloatTensor
        """
        logits = self.onnxruntime_session.run(None, {"x": x.squeeze(1).cpu().numpy(), "xlen": xlen.cpu().numpy()})[0]
        logits = torch.as_tensor(logits, device=x.device)
        return logits


def load_weights_from_ckpt(model, ckpt_path):
    state_dict = torch.load(ckpt_path)

    new_state_dict = {}
    new_state_dict["frontend.stft.weight"] = model.frontend.stft.weight # frontend has static weights
    for key, value in state_dict.items():
        if key.startswith('0.'):
            new_key = key[2:]
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)


def main():
    torch.set_grad_enabled(False)
    device = "cuda:0"
    batch_size = 16
    time_dim = 40320
    onnx_save_path = "model.onnx"
    checkpoint_path = "model.ckpt"

    model = SpeechRecognitionModel().eval().to(device)
    load_weights_from_ckpt(model, checkpoint_path)
    model.fuse_conv_bn_eval()

    print("Model in fp16 mode!")
    model = InputOutputTypeCast(model.to(torch.float16), dtype=torch.float16)

    waveform_input = torch.rand(
        batch_size,
        time_dim,
        device=device,
    )
    xlen = torch.rand(batch_size, device=device)
    torch_logits = model(waveform_input, xlen)

    torch.onnx.export(
        model,
        (
            waveform_input,
            xlen,
        ),
        onnx_save_path,
        verbose=False,
        opset_version=17,
        export_params=True,
        do_constant_folding=True,
        input_names=["x", "xlen"],
        output_names=["logits"],
    )

    ort_wrapper = OnnxWrapper(onnx_save_path)
    ort_logits = ort_wrapper(waveform_input, xlen)


    print(torch_logits)
    print(f"{torch.isnan(torch_logits).any()}")
    print(f"{torch.isnan(ort_logits).any()}")
    print(ort_logits)

    abs_diff = (torch_logits - ort_logits).abs()
    print("pytorch with ort max difference: ", abs_diff.max())
    print("pytorch with ort mean difference: ", abs_diff.mean())


if __name__ == "__main__":
    main()
