#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math


class SdfDecoder(nn.Module):
    def __init__(
        self,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        xyz_in_all=None,
        use_tanh=False,
        weight_norm=False,
    ):
        super(SdfDecoder, self).__init__()
        dims = [3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            out_dim = dims[layer + 1]
            if self.xyz_in_all and layer != self.num_layers - 2:
                out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]
        x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x


def init_recurrent_weights(self):
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)


def lstm_forget_gate_init(lstm_layer):
    for name, parameter in lstm_layer.named_parameters():
        if not "bias" in name: continue
        n = parameter.size(0)
        start, end = n // 4, n // 2
        parameter.data[start:end].fill_(1.)


def clip_grad_norm_hook(x, max_norm=10):
    total_norm = x.norm()
    total_norm = total_norm ** (1 / 2.)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        return x * clip_coef


def init_out_weights(self):
    for m in self.modules():
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -1e-5, 1e-5)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)


class Warper(nn.Module):
    def __init__(
            self,
            latent_size,
            hidden_size,
            steps,
    ):
        super(Warper, self).__init__()
        self.n_feature_channels = latent_size + 3
        self.steps = steps
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size=self.n_feature_channels,
                                hidden_size=hidden_size)
        self.lstm.apply(init_recurrent_weights)
        lstm_forget_gate_init(self.lstm)

        self.out_layer_coord_affine = nn.Linear(hidden_size, 6)
        self.out_layer_coord_affine.apply(init_out_weights)

    def forward(self, input, step=1.0):
        if step < 1.0:
            input_bk = input.clone().detach()

        xyz = input[:, -3:]
        code = input[:, :-3]
        states = [None]
        warping_param = []

        warped_xyzs = []
        for s in range(self.steps):
            state = self.lstm(torch.cat([code, xyz], dim=1), states[-1])
            if state[0].requires_grad:
                state[0].register_hook(lambda x: x.clamp(min=-10, max=10))
            a = self.out_layer_coord_affine(state[0])
            tmp_xyz = torch.addcmul(a[:, 3:], (1 + a[:, :3]), xyz)

            warping_param.append(a)
            states.append(state)
            if (s+1) % (self.steps // 4) == 0:
                warped_xyzs.append(tmp_xyz)
            xyz = tmp_xyz

        if step < 1.0:
            xyz_ = input_bk[:, -3:]
            xyz = xyz * step + xyz_ * (1 - step)

        return xyz, warping_param, warped_xyzs


class Decoder(nn.Module):
    def __init__(self, latent_size, warper_kargs, decoder_kargs):
        super(Decoder, self).__init__()
        self.warper = Warper(latent_size, **warper_kargs)
        self.sdf_decoder = SdfDecoder(**decoder_kargs)

    def forward(self, input, output_warped_points=False, output_warping_param=False,
                step=1.0):
        p_final, warping_param, warped_xyzs = self.warper(input, step=step)

        if not self.training:
            x = self.sdf_decoder(p_final)
            if output_warped_points:
                if output_warping_param:
                    return p_final, x, warping_param
                else:
                    return p_final, x
            else:
                if output_warping_param:
                    return x, warping_param
                else:
                    return x
        else:   # training mode, output intermediate positions and their corresponding sdf prediction
            xs = []
            for p in warped_xyzs:
                xs.append(self.sdf_decoder(p))
            if output_warped_points:
                if output_warping_param:
                    return warped_xyzs, xs, warping_param
                else:
                    return warped_xyzs, xs
            else:
                if output_warping_param:
                    return xs, warping_param
                else:
                    return xs

    def forward_template(self, input):
        return self.sdf_decoder(input)
