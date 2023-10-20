import math
from collections import OrderedDict

import librosa
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import LayerNorm
import torch.nn.functional as F
import torch.distributed
from speechbrain.lobes.augment import SpecAugment

from nemo.collections.asr.parts.submodules.multi_head_attention import PositionalEncoding, RelPositionalEncoding
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling, StackingSubsampling
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType
from nemo.collections.asr.parts.utils.activations import Swish
from nemo.core.classes.mixins import AccessMixin
from nemo.core.classes.mixins.adapter_mixins import AdapterModuleMixin


__all__ = ["SpeechRecognitionModel"]


NORM_CONST = 32


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer of Transformer.

    Args:
    ----
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate) -> None:
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transforms query, key and value.

        Args:
        ----
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value (torch.Tensor): (batch, time2, size)

        Returns:
        -------
            q (torch.Tensor): (batch, head, time1, size)
            k (torch.Tensor): (batch, head, time2, size)
            v (torch.Tensor): (batch, head, time2, size).
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
        ----
            value (torch.Tensor): (batch, time2, size)
            scores(torch.Tensor): (batch, time1, time2)
            mask(torch.Tensor): (batch, time1, time2)

        Returns:
        -------
            value (torch.Tensor): transformed `value` (batch, time2, d_model) weighted by the attention scores.
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, -10000.0)

            # CHANGED! overflow heuristic
            attn = torch.softmax((scores / NORM_CONST - (scores / NORM_CONST).max()) * NORM_CONST, dim=-1).masked_fill(
                mask,
                0.0,
            )  # (batch, head, time1, time2)
        else:
            # CHANGED! overflow heuristic
            attn = torch.softmax(
                (scores / NORM_CONST - (scores / NORM_CONST).max()) * NORM_CONST,
                dim=-1,
            )  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask, pos_emb=None):
        """Compute 'Scaled Dot Product Attention'.

        Args:
        ----
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)

        Returns:
        -------
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention.
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.s_d_k
        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadAttention(MultiHeadAttention):
    """Multi-Head Attention layer of Transformer-XL with support of relative positional encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate, pos_bias_u, pos_bias_v) -> None:
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
        ----
            x (torch.Tensor): (batch, nheads, time, 2*time-1).
        """
        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)
        return x

    def forward(self, query, key, value, mask, pos_emb):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
        ----
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            pos_emb (torch.Tensor) : (batch, time1, size)

        Returns:
        -------
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention.
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        # drops extra elements in the matrix_bd to match the matrix_ac's size
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]

        scores = (matrix_ac + matrix_bd) / self.s_d_k  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)

class ConformerLayer(torch.nn.Module, AdapterModuleMixin, AccessMixin):
    """A single block of the Conformer encoder.

    Args:
    ----
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        conv_kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions.
    """

    def __init__(
        self,
        d_model,
        d_ff,
        self_attention_model="rel_pos",
        n_heads=4,
        conv_kernel_size=31,
        conv_norm_type="batch_norm",
        dropout=0.1,
        dropout_att=0.1,
        pos_bias_u=None,
        pos_bias_v=None,
        adaptive_scale=True,
    ) -> None:
        super().__init__()

        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5

        # first feed forward module
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(d_model=d_model, kernel_size=conv_kernel_size, norm_type=conv_norm_type)

        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)
        if self_attention_model == "rel_pos":
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
            )
        elif self_attention_model == "abs_pos":
            self.self_attn = MultiHeadAttention(n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att)
        else:
            msg = (
                f"'{self_attention_model}' is not not a valid value for 'self_attention_model', valid values can be"
                " from ['rel_pos', 'abs_pos']"
            )
            raise ValueError(
                msg,
            )

        # second feed forward module
        self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)

        self.feed_forward1_scale = ScaleBiasLayer(d_model=d_model, adaptive_scale=adaptive_scale)
        self.conv_scale = ScaleBiasLayer(d_model=d_model, adaptive_scale=adaptive_scale)
        self.self_attn_scale = ScaleBiasLayer(d_model=d_model, adaptive_scale=adaptive_scale)
        self.feed_forward2_scale = ScaleBiasLayer(d_model=d_model, adaptive_scale=adaptive_scale)

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None):
        """Args:
        ----
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
        Returns:
            x (torch.Tensor): (B, T, d_model).
        """
        residual = x
        x = self.feed_forward1_scale(x)
        x = self.feed_forward1(x)

        residual = self.norm_self_att(residual + self.dropout(x) * self.fc_factor)
        x = self.self_attn_scale(residual)
        if self.self_attention_model == "rel_pos":
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb)
        elif self.self_attention_model == "abs_pos":
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask)
        else:
            x = None

        residual = self.norm_conv(residual + self.dropout(x))
        x = self.conv_scale(residual)
        x = self.conv(x, pad_mask)

        residual = self.norm_feed_forward2(residual + self.dropout(x))
        x = self.feed_forward2_scale(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_out(residual)

        if self.is_adapter_available():
            # Call the adapters
            x = self.forward_enabled_adapters(x)

        if self.is_access_enabled():
            self.register_accessible_tensor(tensor=x)

        return x


class ScaleBiasLayer(torch.nn.Module):
    """Computes an affine transformation y = x * scale + bias, either learned via adaptive weights, or fixed.
    Efficient alternative to LayerNorm where we can avoid computing the mean and variance of the input, and
    just rescale the output of the previous layer.

    Args:
    ----
        d_model (int): input dimension of layer.
        adaptive_scale (bool): whether to learn the affine transformation parameters or not. If set to False,
            the scale is fixed to 1 and bias to 0, effectively performing a No-Op on the input.
            This is done for export compatibility.
    """

    def __init__(self, d_model: int, adaptive_scale: bool) -> None:
        super().__init__()
        self.adaptive_scale = adaptive_scale
        if adaptive_scale:
            self.scale = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_buffer("scale", torch.ones(d_model), persistent=True)
            self.register_buffer("bias", torch.zeros(d_model), persistent=True)

    def forward(self, x):
        scale = self.scale.view(1, 1, -1)
        bias = self.bias.view(1, 1, -1)
        return x * scale + bias


class ConformerConvolution(nn.Module):
    """The convolution module for the Conformer model.

    Args:
    ----
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution.
    """

    def __init__(self, d_model, kernel_size, norm_type="batch_norm") -> None:
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
            bias=True,
        )
        if norm_type == "batch_norm":
            self.batch_norm = nn.BatchNorm1d(d_model)
        elif norm_type == "layer_norm":
            self.batch_norm = nn.LayerNorm(d_model)
        else:
            msg = f"conv_norm_type={norm_type} is not valid!"
            raise ValueError(msg)

        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x, pad_mask=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)

        if pad_mask is not None:
            x = x.float().masked_fill(pad_mask.unsqueeze(1), 0.0).to(x.dtype)

        x = self.depthwise_conv(x)

        if isinstance(self.batch_norm, nn.LayerNorm):
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        return x


class ConformerFeedForward(nn.Module):
    """feed-forward module of Conformer model."""

    def __init__(self, d_model, d_ff, dropout, activation=Swish()) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class RelPositionMultiHeadAttention(MultiHeadAttention):
    """Multi-Head Attention layer of Transformer-XL with support of relative positional encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate, pos_bias_u, pos_bias_v) -> None:
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
        ----
            x (torch.Tensor): (batch, nheads, time, 2*time-1).
        """
        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)
        return x

    def forward(self, query, key, value, mask, pos_emb):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
        ----
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            pos_emb (torch.Tensor) : (batch, time1, size)

        Returns:
        -------
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention.
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        # drops extra elements in the matrix_bd to match the matrix_ac's size
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]

        scores = (matrix_ac + matrix_bd) / self.s_d_k  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)


def temporal_mask(x, lengths):
    """Mask out the values of x that are beyond the length of the sequence.
    @param x: (B, C, T) / (B, T)
    @type x: Float Tensor
    @param lengths: (B)
    @type lengths: LongTensor
    @return: (B, C, T) / (B, T)
    @rtype: Float Tensor.
    """
    return (torch.arange(x.shape[-1], device=x.device, dtype=lengths.dtype).unsqueeze(0) < lengths.unsqueeze(1)).view(
        x.shape[:1] + (1,) * (len(x.shape) - 2) + x.shape[-1:],
    )


def compute_output_lengths(x: Tensor, lengths_fraction: Tensor = None):
    """Compute the output lengths of a batch of sequences.
    @param x: (B, C, T) / (B, T)
    @type x: Float Tensor
    @param lengths_fraction: Lengths of the input sequences, as a fraction of the total length. (B)
    @type lengths_fraction: Float Tensor
    @return: (B)
    @rtype: LongTensor.
    """
    if lengths_fraction is None:
        return torch.full(x.shape[:1], x.shape[-1], device=x.device, dtype=torch.long)
    return (lengths_fraction * x.shape[-1]).ceil().long()


class ConformerEncoder(NeuralModule, Exportable):
    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        dev = next(self.parameters()).device
        input_example = torch.randn(max_batch, self._feat_in, max_dim).to(dev)
        input_example_length = torch.randint(1, max_dim, (max_batch,)).to(dev)
        return (input_example, input_example_length)

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(("B", "D", "T"), SpectrogramType()),
                "length": NeuralType(tuple("B"), LengthsType()),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(("B", "D", "T"), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple("B"), LengthsType()),
            }
        )

    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        feat_out=-1,
        subsampling="striding",
        subsampling_factor=4,
        subsampling_conv_channels=-1,
        ff_expansion_factor=4,
        self_attention_model="rel_pos",
        n_heads=4,
        att_context_size=None,
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=5000,
        conv_kernel_size=31,
        conv_norm_type="batch_norm",
        dropout=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
    ):
        super().__init__()

        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self._feat_in = feat_in
        self.scale = math.sqrt(self.d_model)
        if att_context_size:
            self.att_context_size = att_context_size
        else:
            self.att_context_size = [-1, -1]

        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None

        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        if subsampling and subsampling_factor > 1:
            if subsampling == "stacking":
                self.pre_encode = StackingSubsampling(
                    subsampling_factor=subsampling_factor, feat_in=feat_in, feat_out=d_model
                )
            else:
                self.pre_encode = ConvSubsampling(
                    subsampling=subsampling,
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    conv_channels=subsampling_conv_channels,
                    activation=nn.ReLU(),
                )
        else:
            self.pre_encode = nn.Linear(feat_in, d_model)

        self._feat_out = d_model

        if not untie_biases and self_attention_model == "rel_pos":
            d_head = d_model // n_heads
            pos_bias_u = nn.Parameter(torch.Tensor(n_heads, d_head))
            pos_bias_v = nn.Parameter(torch.Tensor(n_heads, d_head))
            nn.init.zeros_(pos_bias_u)
            nn.init.zeros_(pos_bias_v)
        else:
            pos_bias_u = None
            pos_bias_v = None

        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == "abs_pos":
            pos_bias_u = None
            pos_bias_v = None
            self.pos_enc = PositionalEncoding(
                d_model=d_model, dropout_rate=dropout, max_len=pos_emb_max_len, xscale=self.xscale
            )
        else:
            raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")

        self.layers = nn.ModuleList()
        for _i in range(n_layers):
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type=conv_norm_type,
                dropout=dropout,
                dropout_att=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
            )
            self.layers.append(layer)

        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model
        self.set_max_audio_length(self.pos_emb_max_len)
        self.use_pad_mask = True

    def set_max_audio_length(self, max_audio_length):
        """
        Sets maximum input length.
        Pre-calculates internal seq_range mask.
        """
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        seq_range = torch.arange(0, self.max_audio_length, device=device)
        if hasattr(self, "seq_range"):
            self.seq_range = seq_range
        else:
            self.register_buffer("seq_range", seq_range, persistent=False)
        self.pos_enc.extend_pe(max_audio_length, device)

    def forward(self, audio_signal, length=None):
        self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        return self.forward_for_export(audio_signal=audio_signal, length=length)

    def forward_for_export(self, audio_signal, length):
        max_audio_length: int = audio_signal.size(-1)

        if max_audio_length > self.max_audio_length:
            self.set_max_audio_length(max_audio_length)

        if length is None:
            length = audio_signal.new_full(
                audio_signal.size(0), max_audio_length, dtype=torch.int32, device=self.seq_range.device
            )

        audio_signal = torch.transpose(audio_signal, 1, 2)

        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, length = self.pre_encode(audio_signal, length)

        audio_signal, pos_emb = self.pos_enc(audio_signal)
        # adjust size
        max_audio_length = audio_signal.size(1)
        # Create the self-attention and padding masks

        pad_mask = self.make_pad_mask(max_audio_length, length)
        att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
        att_mask = torch.logical_and(att_mask, att_mask.transpose(1, 2))
        if self.att_context_size[0] >= 0:
            att_mask = att_mask.triu(diagonal=-self.att_context_size[0])
        if self.att_context_size[1] >= 0:
            att_mask = att_mask.tril(diagonal=self.att_context_size[1])
        att_mask = ~att_mask

        if self.use_pad_mask:
            pad_mask = ~pad_mask
        else:
            pad_mask = None

        for _lth, layer in enumerate(self.layers):
            audio_signal = layer(x=audio_signal, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        return audio_signal

    def update_max_seq_length(self, seq_length: int, device):
        # Find global max audio length across all nodes
        if torch.distributed.is_initialized():
            global_max_len = torch.tensor([seq_length], dtype=torch.float32, device=device)

            # Update across all ranks in the distributed system
            torch.distributed.all_reduce(global_max_len, op=torch.distributed.ReduceOp.MAX)

            seq_length = global_max_len.int().item()

        if seq_length > self.max_audio_length:
            self.set_max_audio_length(seq_length)

    def make_pad_mask(self, max_audio_length, seq_lens):
        """Make masking for padding."""
        mask = self.seq_range[:max_audio_length].expand(seq_lens.size(0), -1) < seq_lens.unsqueeze(-1)
        return mask

    def enable_pad_mask(self, on=True):
        # On inference, user may chose to disable pad mask
        mask = self.use_pad_mask
        self.use_pad_mask = on
        return mask


def normalize_signal(signal, dim=-1, eps=1e-5, denom_multiplier=1.0):
    signal_max = signal.abs().max(dim=dim, keepdim=True).values + eps
    return signal / (signal_max * denom_multiplier) if signal.numel() > 0 else signal


class LogFilterBankFrontend(nn.Module):
    def __init__(
        self,
        out_channels,
        sample_rate,
        window_stride,
        window,
        window_size,
        preemphasis=0.97,
        eps=torch.finfo(torch.float16).tiny,
        normalize_signal=True,
        stft_mode=None,
        window_periodic=True,
        data_augmentation=None,
        spec_augmentation=None,
    ) -> None:
        super().__init__()
        self.stft_mode = stft_mode
        self.preemphasis = preemphasis
        self.normalize_signal = normalize_signal
        self.sample_rate = sample_rate
        self.window_stride = window_stride

        self.win_length = int(window_size * sample_rate)
        self.hop_length = int(window_stride * sample_rate)
        self.nfft = 2 ** math.ceil(math.log2(self.win_length))
        self.freq_cutoff = self.nfft // 2 + 1

        self.register_buffer("window", getattr(torch, window)(self.win_length, periodic=window_periodic).float())
        mel_basis = torch.as_tensor(
            librosa.filters.mel(
                sr=sample_rate, n_fft=self.nfft, n_mels=out_channels, fmin=0, fmax=int(sample_rate / 2)
            ),
        )
        self.mel = nn.Conv1d(mel_basis.shape[1], mel_basis.shape[0], 1).requires_grad_(False)
        self.mel.weight.copy_(mel_basis.unsqueeze(-1))
        self.mel.bias.fill_(eps)

        self.data_augmentation = data_augmentation
        self.spec_augmentation = spec_augmentation


        # implementation torch.fft.fft is unstable in pytorch 1.10.1 and sometimes put nan into result
        fourier_basis = torch.tensor(np.fft.fft(np.eye(self.nfft), axis=1).view("(2,)float"), dtype=torch.float32)
        forward_basis = fourier_basis[: self.freq_cutoff].permute(2, 0, 1).reshape(-1, 1, fourier_basis.shape[1])
        forward_basis = forward_basis * torch.as_tensor(
            librosa.util.pad_center(self.window, size=self.nfft),
            dtype=forward_basis.dtype,
        )
        self.stft = nn.Conv1d(
            forward_basis.shape[1],
            forward_basis.shape[0],
            forward_basis.shape[2],
            bias=False,
            stride=self.hop_length,
        ).requires_grad_(False)
        self.stft.weight.copy_(forward_basis)

    def forward(self, signal: Tensor, training: bool = False, mask: Tensor = None):
        """Forward method for encoder
        @param signal: (B, T)
        @type signal: Tensor
        @param training: is training (for augs)
        @type training: bool
        @param mask: (B)
        @type mask: LongTensor
        @return: (B, C, T)
        @rtype: Tensor.
        """
        assert signal.ndim == 2

        if self.data_augmentation is not None and training:
            signal = self.data_augmentation(signal.unsqueeze(1), sample_rate=self.sample_rate).squeeze(1)

        signal = signal if signal.is_floating_point() else signal.to(torch.float32)

        signal = normalize_signal(signal) if self.normalize_signal else signal
        signal = (
            torch.cat([signal[..., :1], signal[..., 1:] - self.preemphasis * signal[..., :-1]], dim=-1)
            if self.preemphasis > 0
            else signal
        )
        signal = signal * mask if mask is not None else signal

        pad = self.freq_cutoff - 1
        padded_signal = F.pad(
            signal,
            (pad, 0),
            mode="constant",
        )

        padded_signal = F.pad(
            padded_signal,
            (0, pad),
            mode="constant",
            value=0,
        ) 

        stft_res = self.stft(padded_signal.unsqueeze(dim=1))
        real_squared, imag_squared = (stft_res * stft_res).split(self.freq_cutoff, dim=1)

        power_spectrum = real_squared + imag_squared
        log_mel_features = self.mel(power_spectrum).log()

        if self.spec_augmentation and training:
            log_mel_features = self.spec_augmentation(log_mel_features)
        return log_mel_features



class MaskedInstanceNorm1d(nn.InstanceNorm1d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x, mask=None):
        if mask is not None:
            assert self.track_running_stats is False
            xlen = mask.int().sum(dim=-1, keepdim=True)
            mean = (x * mask).sum(dim=-1, keepdim=True) / xlen
            zero_mean_masked = mask * (x - mean)
            std = ((zero_mean_masked * zero_mean_masked).sum(dim=-1, keepdim=True) / xlen).sqrt()
            return zero_mean_masked / (std + self.eps)
        else:
            return super().forward(x)
        

class Decoder(nn.Sequential):
    def __init__(self, input_size, num_classes) -> None:
        super().__init__(nn.Conv1d(input_size, num_classes, kernel_size=1))

    def forward(self, x):
        return self[0](x)


class Conformer(nn.Module):
    def __init__(self) -> None:
        """Nemo Conformer initialization from nemo.collections.asr.modules.conformer_encoder.ConformerEncoder
        @param backbone: Conformer Encoder
        @type backbone: nemo.collections.asr.modules.conformer_encoder.ConformerEncoder.
        """
        super().__init__()

        num_input_features = 64
        backbone = ConformerEncoder(
            feat_in=num_input_features,
            feat_out=-1,
            n_layers=18,
            d_model=512,

            subsampling="striding",
            subsampling_factor=4,
            subsampling_conv_channels=-1,

            ff_expansion_factor=4,

            self_attention_model="rel_pos",
            n_heads=8,
            att_context_size=[ -1, -1 ], 
            xscaling=True,
            untie_biases=True, 
            pos_emb_max_len=50000,

            conv_kernel_size=31,
            conv_norm_type='batch_norm',

            dropout=0.1,
            dropout_emb=0.0,
            dropout_att=0.1,
        )
        self.backbone = backbone

    def forward(self, x: Tensor, xlen: Tensor) -> tuple[Tensor, Tensor]:
        """Forward method for encoder.
        @param x: (B, C, T)
        @type x: FloatTensor
        @param xlen: (B)
        @type xlen: Tensor, range - [0, 1]
        @return: x - (B, C, T) tensor of features, olen - (B) tensor of output long lengths
        @rtype: Tuple[Tensor, Tensor].
        """
        long_len = compute_output_lengths(x, xlen)
        logits = self.backbone(audio_signal=x, length=long_len)

        return logits


class SpeechRecognitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        num_classes = 35
        num_input_features = 64

        self.frontend = LogFilterBankFrontend(
            out_channels=num_input_features,
            sample_rate=8000,
            window_size=0.04,
            window_stride=0.01,
            window="hann_window",
            preemphasis=0.97,
            normalize_signal=True,
            window_periodic=True,
            stft_mode="conv",
        )

        self.backbone = Conformer()

        self.normalize_features = MaskedInstanceNorm1d(
            num_input_features,
            affine=False,
            eps=torch.finfo(torch.float16).tiny,
            track_running_stats=False,
        )


        output_features = int(512)
        self.decoder = Decoder(output_features, num_classes)

    def fuse_conv_bn_eval(self):
        if hasattr(self.backbone, "fuse_conv_bn_eval"):
            self.backbone.fuse_conv_bn_eval()

    def forward(self, x: Tensor, xlen: Tensor) -> Tensor:
        """Forward method for encoder.
        @param x: (B, C, T)
        @type x: FloatTensor
        @param xlen: (B)
        @type xlen: Tensor, range - [0, 1]
        @return: logits - (B, C, T), log_probs - (B, C, T), olen - (B), uncertainty - (B)
        @rtype: Dict[str, Tensor].
        """
        # self.frontend is not None because we want to read old checkpoints
        mask = temporal_mask(x, compute_output_lengths(x, xlen)) if xlen is not None else None
        x = self.frontend(x, mask=mask)

        mask = temporal_mask(x, compute_output_lengths(x, xlen)) if xlen is not None else None
        # NOTE: x.to(dtype=torch.float32) need in case of long audio with fp16 inference.
        # Long xlen cause float16 overflow to -inf.
        x = self.normalize_features(x.to(dtype=torch.float32), mask=mask).to(dtype=x.dtype)

        x = self.backbone(x, xlen)
        logits = self.decoder(x)

        return logits
