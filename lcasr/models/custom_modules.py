import math
import random

import torch
import torch.nn as nn
import torchaudio
from madmom.audio.stft import fft_frequencies
from madmom.audio.spectrogram import LogarithmicFilterbank

torchaudio.set_audio_backend("sox_io")


class TemporalBatchNorm(nn.Module):
    """
    Batch normalization of a (batch, channels, bands, time) tensor over all but
    the previous to last dimension (the frequency bands).
    credits:
    https://github.com/CPJKU/cyolo_score_following/blob/eusipco-2021/cyolo_score_following/models/custom_modules.py
    """
    def __init__(self, num_bands, affine=True):
        super(TemporalBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_bands, affine=affine)

    def forward(self, x):
        shape = x.shape
        # squash channels into the batch dimension
        x = x.reshape((-1,) + x.shape[-2:])

        # B x T x F -> B x F x T
        x = x.permute(0, 2, 1)
        # pass through 1D batch normalization
        x = self.bn(x)

        # B x F x T -> B x T x F
        x = x.permute(0, 2, 1)

        # restore squashed dimensions
        return x.reshape(shape)


class LogSpectrogramModule(nn.Module):
    """
    credits:
    https://github.com/CPJKU/cyolo_score_following/blob/eusipco-2021/cyolo_score_following/models/custom_modules.py
    """

    def __init__(self, args):
        super(LogSpectrogramModule, self).__init__()

        self.sr = args.sr
        self.fps = args.fps
        self.n_fft = args.fft_frame_size
        self.hop_length = int(self.sr / self.fps) + 1
        self.min_rate = args.tstretch_range[0]
        self.max_rate = args.tstretch_range[1]
        self.p_tstretch = args.p_tstretch
        self.p_fmask = args.p_fmask
        self.tc = int(args.fps * args.snippet_len[args.audio_context])  # temporal context

        fbank = LogarithmicFilterbank(fft_frequencies(self.n_fft // 2 + 1, self.sr),
                                      num_bands=16, fmin=30, fmax=6000, norm_filters=True, unique_filters=True)
        fbank = torch.from_numpy(fbank)
        phase_advance = torch.linspace(0, math.pi * self.hop_length, self.n_fft // 2 + 1)[..., None]

        self.register_buffer('window', torch.hann_window(self.n_fft))
        self.register_buffer('fbank', fbank.unsqueeze(0))
        self.register_buffer('phase_advance', phase_advance)

        max_freq_mask = int(fbank.shape[1] * args.fmask_max)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=max_freq_mask, iid_masks=True)

        # self.apply_impulse_response = Compose(transforms=[
        #     ApplyImpulseResponse(ir_paths=args.rir_root, p=args.p_rir_gpu, sample_rate=self.sr, target_rate=self.sr)
        # ])

    def compute_batch_spectrograms(self, x):

        # x = self.apply_impulse_response(x.unsqueeze(dim=1)).squeeze()

        x_stft_batch = torch.stft(x.squeeze(), n_fft=self.n_fft, hop_length=self.hop_length, window=self.window,
                                  center=True, return_complex=True)

        if self.p_tstretch:
            specs = []
            for spec in x_stft_batch:
                if self.p_tstretch > random.random():
                    rate = random.uniform(self.min_rate, self.max_rate)
                    spec = torchaudio.functional.phase_vocoder(spec, rate, self.phase_advance)
                spec = spec[:, (spec.shape[1] - self.tc) // 2:(spec.shape[1] - self.tc) // 2 + self.tc]
                specs.append(spec)
            specs = torch.stack(specs, dim=0)
        else:
            specs = x_stft_batch

        assert self.tc == specs.shape[-1]

        specs = torch.view_as_real(specs).pow(2).sum(-1).sqrt().permute(0, 2, 1)
        specs = torch.log10(torch.matmul(specs, self.fbank) + 1).permute(0, 2, 1).unsqueeze(dim=1)

        # apply freq masking
        if random.random() < self.p_fmask:
            specs = self.freq_masking(specs)

        return specs

    def forward(self, x):

        return self.compute_batch_spectrograms(x)


if __name__ == '__main__':
    pass
