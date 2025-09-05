Description = "NN model collection"

import math
from copy import deepcopy
import pytorch_lightning as pl
import torch
from einops.layers.torch import Rearrange
from torch import nn

class squeeze(nn.Module):
    def forward(self, x):
        return torch.squeeze(x)

def default(val, d):
    return val if val is not None else d


def exponential_linspace_int(start, end, num, divisible_by=1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]


def DenseBlock(num_f, activation=nn.LeakyReLU, dropout=0.0):
    return nn.Sequential(
        nn.Linear(num_f, num_f),
        activation(),
        nn.BatchNorm1d(num_f),
        nn.Dropout(dropout),
    )


def ConvBlock(dim, dim_out=None, kernel_size=1, stride=1, dilation=1):
    "Standard convolutional block"
    return nn.Sequential(
        nn.Conv1d(
            dim,
            default(dim_out, dim),
            kernel_size,
            padding=int(dilation * (kernel_size - 1) / 2),
            stride=stride,
            dilation=dilation,
        ),
        nn.GELU(),
        nn.BatchNorm1d(default(dim_out, dim)),
    )


def RConvBlock(dim, kernel_size=1, num_conv=1, skip=True):
    "A convolutional block stack with skip connnection"
    layer_list = []
    for i in range(num_conv):
        layer_list.extend(
            [
                nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.BatchNorm1d(dim),
            ]
        )
    if skip:
        return Residual(nn.Sequential(*layer_list))
    else:
        return nn.Sequential(*layer_list)


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim is None:
            return torch.squeeze(x)
        else:
            return torch.squeeze(x, dim=self.dim)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class ConvTowerDomain_v6(pl.LightningModule):
    """
    Convolutional tower model v6
    Separate sequence and chromatin convolutional tower for only applying domain adaptation to one of them
    Final Attention layers merge seq and chrom
    """

    def __init__(
        self,
        chroms_channel=1,
        input_len=512,
        conv1d_filter=64,
        conv_tower_depth=4,
        conv_tower_kernel=5,
        conv_tower_dilation=1,
        attn_num_layers=2,
        attn_num_heads=8,
        attn_dim_feedforward=2048,
        activation="LeakyReLU",
        dropout=0.0,
        gamma=0.99995,
        lr=1e-4,
        lambd=0,
        classification=False,
        seqonly=True,
    ):
        super().__init__()
        self.chroms_channel = chroms_channel
        self.input_len = input_len
        self.conv1d_filter = conv1d_filter
        self.conv_tower_depth = conv_tower_depth
        self.conv_tower_kernel = conv_tower_kernel
        self.conv_tower_dilation = conv_tower_dilation
        self.attn_num_layers = attn_num_layers
        self.attn_num_heads = attn_num_heads
        self.attn_dim_feedforward = attn_dim_feedforward
        self.activation = getattr(nn, activation)
        self.dropout = dropout
        self.gamma = gamma
        self.lr = lr
        self.lambd = lambd
        self.classification = classification
        self.seqonly = seqonly
        self.save_hyperparameters()

        self.example_input_array = (
            torch.zeros(512, 4, self.input_len).index_fill_(1, torch.tensor(2), 1),
            torch.ones(512, self.chroms_channel, self.input_len),
        )

        # stem
        self.seq_stem = nn.Sequential(
            nn.Conv1d(4, self.conv1d_filter, 25, padding=12, bias=False),
            self.activation(),
            nn.BatchNorm1d(self.conv1d_filter),
        )
        if not self.seqonly:
            self.chrom_stem = nn.Sequential(
                nn.Conv1d(
                    self.chroms_channel, self.conv1d_filter, 25, padding=12, bias=False
                ),
                self.activation(),
                nn.BatchNorm1d(self.conv1d_filter),
            )

        # convolutional tower
        ## compute the depth because the exponential base is fixed to 2 here
        self.conv_tower_outdim = int(
            self.conv1d_filter * (2 ** int(self.conv_tower_depth - 1))
        )
        self.conv_tower_outlen = int(
            self.input_len / (2 ** int(self.conv_tower_depth - 1))
        )
        dim_lists = exponential_linspace_int(
            self.conv1d_filter,
            self.conv_tower_outdim,
            num=self.conv_tower_depth,
            divisible_by=2,
        )
        conv_tower = []
        for dim_in, dim_out in zip(dim_lists[:-1], dim_lists[1:]):
            conv_tower.append(
                nn.Sequential(
                    RConvBlock(dim_in, kernel_size=self.conv_tower_kernel, skip=True),
                    ConvBlock(dim_in, dim_out=dim_out, kernel_size=1),
                    nn.MaxPool1d(kernel_size=2),
                )
            )
        self.seq_conv_tower = nn.Sequential(*conv_tower)
        if not self.seqonly:
            self.chrom_conv_tower = deepcopy(self.seq_conv_tower)
        merge_dim = (
            self.conv_tower_outdim if self.seqonly else self.conv_tower_outdim * 2
        )

        # self attention
        self.pos_encoder = PositionalEncoding(
            merge_dim, self.dropout, self.conv_tower_outlen
        )  # assume we merged sequence and chromatin inputs here
        transformer_encoder = []
        for i in range(self.attn_num_layers):
            transformer_encoder.append(
                nn.TransformerEncoderLayer(
                    d_model=merge_dim,
                    nhead=self.attn_num_heads,
                    dim_feedforward=self.attn_dim_feedforward,
                    activation="relu",
                )
            )
        self.transformer_encoder = nn.Sequential(*transformer_encoder)
        self.pre_attn = nn.Sequential(Rearrange("b c l -> l b c"), self.pos_encoder)
        self.post_attn = nn.Sequential(
            Rearrange("l b c -> b c l"),
            nn.BatchNorm1d(merge_dim),
            self.activation(),
            nn.Dropout(self.dropout),
            nn.Conv1d(merge_dim, 1, 1),
            self.activation(),
            nn.BatchNorm1d(1),
            Squeeze(dim=1),
        )

        # final output
        self.main_pred = nn.Sequential(nn.Linear(self.conv_tower_outlen, 1))

        # tpcav parameters
        # self.register_buffer("zscore_mean", None)
        # self.register_buffer("zscore_std", None)
        # self.register_buffer("pca_inv", None)
        # self.orig_shape = None

    def forward(self, seq):
        seq = self.seq_stem(seq)
        seq = self.seq_conv_tower(seq)
        y_hat = seq
        y_hat = self.pre_attn(y_hat)
        y_hat = self.transformer_encoder(y_hat)
        y_hat = self.post_attn(y_hat)
        y_pred = self.main_pred(y_hat)
        return y_pred

    def merge_projected_and_residual(self, y_projected, y_residual):
        "Project PCA activations back to original space and add with residual, then multiply with zscore_std"
        "NOTE: zscore_mean is not added here , this is to avoid adding it twice if you disentange y_projected and y_residual"
        y_hat = torch.matmul(y_projected, self.pca_inv) + y_residual  # PCA inverse
        y_hat = y_hat * self.zscore_std  # reverse normalize

        return y_hat

    def forward_until_select_layer(self, seq):
        seq = self.seq_stem(seq)
        seq = self.seq_conv_tower(seq)
        y_hat = seq
        y_hat = self.pre_attn(y_hat)
        y_hat = self.transformer_encoder(y_hat)
        y_hat = self.post_attn(y_hat)

        return y_hat

    def resume_forward_from_select_layer(self, y_hat):
        return self.main_pred(y_hat)

    def forward_from_start(
        self,
        seq,
        cavs_list=None,
        mute_x_avs=False,
        mute_remainder=False,
        project_to_pca=True,
    ):
        y_hat = self.forward_until_select_layer(seq)

        if project_to_pca:
            avs_residual, avs_projected = self.project_post_attn_to_pca(
                y_hat.flatten(start_dim=1)
            )

            return self.forward_from_projected_and_residual(
                avs_residual,
                avs_projected,
                cavs_list,
                mute_x_avs,
                mute_remainder,
            )
        else:
            return self.resume_forward_from_select_layer(y_hat)

    def forward_from_projected_and_residual(
        self,
        y_residual,
        y_projected,
        cavs_list=None,
        mute_x_avs=False,
        mute_remainder=False,
    ):
        if cavs_list is not None:
            y_hat_x_avs, y_hat_remainder = self.disentangle_avs_x_cavs(
                y_projected, y_residual, cavs_list
            )

            if mute_x_avs:  # which part of the activations to use
                y_hat_x_avs.register_hook(lambda grad: torch.zeros_like(grad))
            if mute_remainder:
                y_hat_remainder.register_hook(lambda grad: torch.zeros_like(grad))
            y_hat = y_hat_x_avs + y_hat_remainder + self.zscore_mean
        else:
            y_hat = (
                self.merge_projected_and_residual(y_projected, y_residual)
                + self.zscore_mean
            )

        if self.orig_shape is not None:
            y_hat = y_hat.reshape((-1, *self.orig_shape[1:]))

        # resume back to normal forward process
        y_hat = self.resume_forward_from_select_layer(y_hat)
        return y_hat

    def project_avs_to_pca(self, y):
        return self.project_post_attn_to_pca(y)

    def project_post_attn_to_pca(self, y):
        "Project activations of post attn layer to PCA space"
        if self.pca_inv is not None:
            V = self.pca_inv.T
            y_standardized = (y - self.zscore_mean) / self.zscore_std
            y_projected = torch.matmul(y_standardized, V)

            y_residual = y_standardized - torch.matmul(y_projected, self.pca_inv)

            return y_residual, y_projected
        else:
            return y, None
