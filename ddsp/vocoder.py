import math
import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
from torch.nn import functional as F

from .mel2control import Mel2Control
from .modules import SawtoothGenerator, HarmonicOscillator, WavetableSynthesizer, WaveGeneratorOscillator
from .core import scale_function, unit_to_hz2, frequency_filter, upsample


# HarmonicOscillator type:
#   - WavetableSynthesizer    : DWS
#   - HarmonicOscillator      : Full, Sins
#   - WaveGeneratorOscillator : SawSinSub
#   - SawtoothGenerator       : SawSub


class Full(nn.Module):
    def __init__(self, sampling_rate, block_size, n_mag_harmonic, n_mag_noise, n_harmonics, n_mels=80):
        """
        Args:
            sampling_rate  :: - waveform sampling rate [Hz]
            block_size     :: -
            n_mag_harmonic :: -
            n_mag_noise    :: -
            n_harmonics    :: -
            n_mels         :: -
        """
        super().__init__()
        print(' [Model] Sinusoids Synthesiser, gt fo')

        self.register_buffer("block_size", torch.tensor(block_size))
        self.mel2ctrl = Mel2Control(n_mels, {'f0': 1, 'A': 1, 'amplitudes': n_harmonics, 'harmonic_magnitude': n_mag_harmonic, 'noise_magnitude': n_mag_noise})
        self.harmonic_synthsizer = HarmonicOscillator(sampling_rate)

    def forward(self, mel, initial_phase=None):
        '''
        Args:
            mel :: (B, Frame, Freq) - Mel-spectrogram
            initial_phase :: ()     -
        '''

        # aNN
        ctrls = self.mel2ctrl(mel)
        f0_unit, A, amp, harmo_mag, noise_mag = ctrls['f0'], ctrls['A'], ctrls['amplitudes'], ctrls['harmonic_magnitude'], ctrls['noise_magnitude']

        # Feature transform
        ## fo - scaling + upsampling
        f0_unit = torch.sigmoid(f0_unit)
        f0 = unit_to_hz2(f0_unit, hz_min = 80.0, hz_max=1000.0)
        f0[f0<80] = 0
        pitch = upsample(f0, self.block_size)
        ## harmonic magnitude - scaling
        src_param   = scale_function(harmo_mag)
        ## noise magnitude - scaling
        noise_param = scale_function(noise_mag)
        ## amplitudes - scaling + ? + upsampling
        A    = scale_function(A)
        amp  = scale_function(amp)
        amp /= amp.sum(-1, keepdim=True) # to distribution
        amp *= A
        amplitudes = upsample(amp, self.block_size)

        # sDSP
        ## harmonic
        base_harmo, final_phase = self.harmonic_synthsizer(pitch, amplitudes, initial_phase)
        colored_harmo = frequency_filter(base_harmo, src_param)
        ## SubNoise
        base_noise = torch.rand_like(colored_harmo).to(noise_param) * 2 - 1
        colored_noise = frequency_filter(base_noise, noise_param)
        ## Harmo+Noise
        signal = colored_harmo + colored_noise

        return signal, f0, final_phase, (colored_harmo, colored_noise)


class SawSinSub(nn.Module):
    def __init__(self, sampling_rate, block_size, n_mag_harmonic, n_mag_noise, n_harmonics, n_mels=80):
        super().__init__()
        print(' [Model] Sawtooth (with sinusoids) Subtractive Synthesiser')

        self.register_buffer("block_size", torch.tensor(block_size))
        self.mel2ctrl = Mel2Control(n_mels, {'f0': 1, 'harmonic_magnitude': n_mag_harmonic, 'noise_magnitude': n_mag_noise})
        # Harmonic Synthsizer
        self.harmonic_amplitudes = nn.Parameter(1. / torch.arange(1, n_harmonics + 1).float(), requires_grad=False)
        self.ratio = nn.Parameter(torch.tensor([0.4]).float(), requires_grad=False)
        self.harmonic_synthsizer = WaveGeneratorOscillator(sampling_rate, amplitudes=self.harmonic_amplitudes, ratio=self.ratio)

    def forward(self, mel, initial_phase=None):
        '''
            mel :: (B, Frame, Freq)
        '''

        # aNN
        ctrls = self.mel2ctrl(mel)
        f0_unit, harmo_mag, noise_mag = ctrls['f0'], ctrls['harmonic_magnitude'], ctrls['noise_magnitude']

        # Feature transform
        ## fo - scaling + upsampling
        f0_unit = torch.sigmoid(f0_unit)
        f0 = unit_to_hz2(f0_unit, hz_min = 80.0, hz_max=1000.0)
        f0[f0<80] = 0
        pitch = upsample(f0, self.block_size)
        ## harmonic magnitude - scaling
        src_param = scale_function(harmo_mag)
        ## noise magnitude - scaling
        noise_param = scale_function(noise_mag)

        # sDSP
        ## SubHarmo
        base_harmo, final_phase = self.harmonic_synthsizer(pitch, initial_phase)
        colored_harmo = frequency_filter(base_harmo, src_param)
        ## SubNoise
        base_noise = torch.rand_like(colored_harmo).to(noise_param) * 2 - 1
        colored_noise = frequency_filter(base_noise, noise_param)
        ## Harmo+Noise
        signal = colored_harmo + colored_noise

        return signal, f0, final_phase, (colored_harmo, colored_noise)


class Sins(nn.Module):
    def __init__(self, sampling_rate, block_size, n_harmonics, n_mag_noise, n_mels=80):
        super().__init__()
        print(' [Model] Sinusoids Synthesiser')

        self.register_buffer("block_size", torch.tensor(block_size))
        self.mel2ctrl = Mel2Control(n_mels, {'f0': 1, 'A': 1, 'amplitudes': n_harmonics, 'noise_magnitude': n_mag_noise})
        self.harmonic_synthsizer = HarmonicOscillator(sampling_rate)

    def forward(self, mel, initial_phase=None):
        '''
            mel :: (B, Frame, Freq)
        '''

        # aNN
        ctrls = self.mel2ctrl(mel)
        f0_unit, A, amp, noise_mag = ctrls['f0'], ctrls['A'], ctrls['amplitudes'], ctrls['noise_magnitude']

        # Feature transform
        ## fo - scaling + upsampling
        f0_unit = torch.sigmoid(f0_unit)
        f0 = unit_to_hz2(f0_unit, hz_min = 80.0, hz_max=1000.0)
        f0[f0<80] = 0
        pitch = upsample(f0, self.block_size)
        ## noise_magnitude - scaling
        noise_param = scale_function(noise_mag)
        ## amplitudes - scaling + ? + upsampling
        A    = scale_function(A)
        amp  = scale_function(amp)
        amp /= amp.sum(-1, keepdim=True) # to distribution
        amp *= A
        amplitudes = upsample(amp, self.block_size)

        # sDSP
        ## AddHarmo
        colored_harmo, final_phase = self.harmonic_synthsizer(pitch, amplitudes, initial_phase)
        ## SubNoise
        base_noise = torch.rand_like(colored_harmo).to(noise_param) * 2 - 1
        colored_noise = frequency_filter(base_noise, noise_param)
        ## Harmo+Noise
        signal = colored_harmo + colored_noise

        return signal, f0, final_phase, (colored_harmo, colored_noise)


class DWS(nn.Module):
    def __init__( self, sampling_rate, block_size, num_wavetables, len_wavetables, is_lpf=False):
        super().__init__()
        print(' [Model] Wavetables Synthesiser, is_lpf:', is_lpf)

        self.register_buffer("block_size", torch.tensor(block_size))
        self.mel2ctrl = Mel2Control(80, {'f0': 1, 'A': 1, 'amplitudes': num_wavetables, 'noise_magnitude': 80})
        # Harmonic Synthsizer
        self.wavetables = nn.Parameter(torch.randn(num_wavetables, len_wavetables))
        self.harmonic_synthsizer = WavetableSynthesizer(sampling_rate, self.wavetables, block_size, is_lpf=is_lpf)

    def forward(self, mel, initial_phase=None):
        '''
            mel :: (B, Frame, Freq)
        '''

        # aNN
        ctrls = self.mel2ctrl(mel)
        f0_unit, A, amp, noise_mag = ctrls['f0'], ctrls['A'], ctrls['amplitudes'], ctrls['noise_magnitude']

        # Feature transform
        ## fo - scaling
        f0_unit = torch.sigmoid(f0_unit)
        f0 = unit_to_hz2(f0_unit, hz_min = 80.0, hz_max=1000.0)
        f0[f0<80] = 0
        pitch = f0
        ## noise_magnitude - scaling
        noise_param = scale_function(noise_mag)
        ## amplitudes - scaling + ?
        A    = scale_function(A)
        amp  = scale_function(amp)
        amp /= amp.sum(-1, keepdim=True) # to distribution
        amplitudes *= A

        # sDSP
        ## harmonic
        colored_harmo, final_phase = self.harmonic_synthsizer(pitch, amplitudes, initial_phase)
        ## SubNoise
        base_noise = torch.rand_like(colored_harmo).to(noise_param) * 2 - 1
        colored_noise = frequency_filter(base_noise, noise_param)
        ## Harmo+Noise
        signal = colored_harmo + colored_noise

        return signal, f0, final_phase, (colored_harmo, colored_noise)



class SawSub(nn.Module):
    def __init__(self, sampling_rate, block_size,):
        super().__init__()
        print(' [Model] Sawtooth Subtractive Synthesiser')

        self.register_buffer("block_size", torch.tensor(block_size))
        self.mel2ctrl = Mel2Control(80, {'f0': 1, 'harmonic_magnitude': 512, # 1024 for 48k, 512 for 24k 
            'noise_magnitude': 80})
        self.harmonic_synthsizer = SawtoothGenerator(sampling_rate)

    def forward(self, mel, initial_phase=None):
        """
        Args:
            mel :: (B, Frame, Freq) -
            initial_phase           - 
        Returns:
            signal                  - estimated waveform
            f0                      - estimated fo contour
            final_phase             - 
            Tuple
                colored_harmo       - Harmonic components
                colored_noise       - Noise    components
        """

        # aNN
        ctrls = self.mel2ctrl(mel)
        f0_unit, harmo_mag, noise_mag = ctrls['f0'], ctrls['harmonic_magnitude'], ctrls['noise_magnitude']

        # Feature transform
        ## fo - scaling + upsampling
        f0_unit = torch.sigmoid(f0_unit)
        f0 = unit_to_hz2(f0_unit, hz_min = 80.0, hz_max=1000.0)
        f0[f0<80] = 0
        pitch = upsample(f0, self.block_size)
        ## harmonic magnitude - scaling
        src_param   = scale_function(harmo_mag)
        ## noise magnitude - scaling
        noise_param = scale_function(noise_mag)

        # sDSP
        ## SubHarmo
        base_harmo, final_phase = self.harmonic_synthsizer(pitch, initial_phase)
        colored_harmo = frequency_filter(base_harmo, src_param)
        ## SubNoise
        base_noise = torch.rand_like(colored_harmo).to(noise_param) * 2 - 1
        colored_noise = frequency_filter(base_noise, noise_param)
        ## Harmo+Noise
        signal = colored_harmo + colored_noise

        return signal, f0, final_phase, (colored_harmo, colored_noise)
