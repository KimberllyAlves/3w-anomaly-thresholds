"""1-D convolutional VAE for multivariate windows.

Input and output: (batch, window, sensors). Each encoder block is Conv1d (stride 2) + BatchNorm +
LeakyReLU(0.1) + spatial dropout; the decoder mirrors it with transposed convolutions. The lengths
of every encoder layer are computed from the window size, and the decoder uses them to recover the
exact input length for any window size. With predict_variance the decoder ends in two heads, the
reconstruction mean and its log-variance (heteroscedastic Gaussian likelihood, Kendall and Gal, 2017).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _conv_out(length, kernel, stride, padding):
    return (length + 2 * padding - kernel) // stride + 1


def _output_padding(in_len, out_len, kernel, stride, padding):
    pad = out_len - ((in_len - 1) * stride - 2 * padding + kernel)
    if not 0 <= pad < stride:
        raise ValueError(f'window size incompatible with kernel {kernel} and stride {stride}')
    return pad


@dataclass
class ConvVAEConfig:
    in_channels: int
    window_size: int
    latent_dim: int = 32
    channels: tuple[int, ...] = (16, 32, 64, 128)
    kernel_size: int = 5
    stride: int = 2
    dropout: float = 0.2
    predict_variance: bool = False


class Conv1DVAE(nn.Module):
    def __init__(self, config: ConvVAEConfig):
        super().__init__()
        self.config = c = config
        pad = c.kernel_size // 2
        lengths = [c.window_size]
        for _ in c.channels:
            lengths.append(_conv_out(lengths[-1], c.kernel_size, c.stride, pad))
        if lengths[-1] < 1:
            raise ValueError(f'window size {c.window_size} too small for {len(c.channels)} layers')
        self._lengths = lengths

        layers, in_ch = [], c.in_channels
        for out_ch in c.channels:
            layers += [nn.Conv1d(in_ch, out_ch, c.kernel_size, c.stride, padding=pad),
                       nn.BatchNorm1d(out_ch), nn.LeakyReLU(0.1, inplace=True), nn.Dropout1d(c.dropout)]
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)

        flat = c.channels[-1] * lengths[-1]
        self.fc_mu = nn.Linear(flat, c.latent_dim)
        self.fc_logvar = nn.Linear(flat, c.latent_dim)
        self.fc_decode = nn.Linear(c.latent_dim, flat)
        self._unflatten = (c.channels[-1], lengths[-1])

        out_channels = list(reversed(c.channels[:-1])) + [c.in_channels]
        layers, in_ch, n = [], c.channels[-1], len(c.channels)
        for i, out_ch in enumerate(out_channels):
            in_len, out_len = lengths[n - i], lengths[n - i - 1]
            last = i == len(out_channels) - 1
            if last and c.predict_variance:
                head_in, head_len = in_ch, (in_len, out_len)
                break
            op = _output_padding(in_len, out_len, c.kernel_size, c.stride, pad)
            layers.append(nn.ConvTranspose1d(in_ch, out_ch, c.kernel_size, c.stride, padding=pad, output_padding=op))
            if not last:
                layers += [nn.BatchNorm1d(out_ch), nn.LeakyReLU(0.1, inplace=True), nn.Dropout1d(c.dropout)]
            in_ch = out_ch
        self.decoder = nn.Sequential(*layers)

        if c.predict_variance:
            op = _output_padding(*head_len, c.kernel_size, c.stride, pad)
            head = lambda: nn.ConvTranspose1d(head_in, c.in_channels, c.kernel_size, c.stride, padding=pad, output_padding=op)
            self.decoder_mean_head = head()
            self.decoder_logvar_head = head()

    def encode(self, x):
        """x: (batch, channels, window) -> mean and log-variance of the latent distribution."""
        h = self.encoder(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        h = self.decoder(self.fc_decode(z).view(-1, *self._unflatten))
        if not self.config.predict_variance:
            return h
        return self.decoder_mean_head(h), self.decoder_logvar_head(h)

    def forward(self, x):
        mu, logvar = self.encode(x.permute(0, 2, 1))
        z = self.reparameterize(mu, logvar)
        out = {'mu': mu, 'logvar': logvar, 'z': z}
        decoded = self.decode(z)
        if self.config.predict_variance:
            out['reconstruction'] = decoded[0].permute(0, 2, 1)
            out['recon_logvar'] = decoded[1].permute(0, 2, 1)
        else:
            out['reconstruction'] = decoded.permute(0, 2, 1)
        return out

    def enable_mc_dropout(self):
        """Dropout active at inference while batch normalization stays in eval mode."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout1d)):
                m.train()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
