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

# TODO: [f0 bug]
# Theoretically, `f0<80` never happen because f0 is in (80, 1000).
# But when `sigmoid(f0_unscaled) < 4.4e-8`, it goes to `79.99999237060547...` and it is cut to 0 (maybe) because of numerical issue.
# This enables 'proper' f0=zero learning practically,
# but aNN is required to output super big negative number for f0=0[Hz] (this could be the reason of drastic fo loss behavior).
# If we fix numerical issue, the bug will disappear but learning will fail.
# So, we should totally replace this mechanism.


class Full(nn.Module):
    """Feature analyzer + Subtracted added sinusoids + Subtracted noise."""
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
        ## Mixed NN
        ctrls = self.mel2ctrl(mel)
        f0_unscaled, noise_mag, harmo_mag, A, amp = ctrls['f0'], ctrls['noise_magnitude'], ctrls['harmonic_magnitude'], ctrls['A'], ctrls['amplitudes']
        ## f0 :: (B, Frame, 1) - Fundamental tone's frequency contour [Hz] (0 or within (80, 1000))
        f0 = unit_to_hz2(torch.sigmoid(f0_unscaled), hz_min = 80.0, hz_max=1000.0)
        f0[f0<80] = 0
        ## TODO: Fix bug. Check [f0 bug] in top of this file.

        # Feature transform
        ## fo - upsampling
        pitch = upsample(f0, self.block_size)
        ## noise magnitude :: (B, Frame, n_mag_noise) - scaling to (0, 2)
        noise_param = scale_function(noise_mag)
        ## harmonic magnitude :: (B, Frame, n_mag_harmonic) - scaling to (0, 2)
        src_param   = scale_function(harmo_mag)
        ## amplitudes - normalized_partial_wise * global + upsampling
        amp  = scale_function(amp)
        amp /= amp.sum(-1, keepdim=True) # - Normalized partial amplitudes
        amp *= scale_function(A) # :: * (B, Frame, 1) - * Global amplitude
        amplitudes = upsample(amp, self.block_size)

        # sDSP
        ## SubAddHarmo
        base_harmo,    final_phase = self.harmonic_synthsizer(pitch, amplitudes, initial_phase)
        colored_harmo = frequency_filter(base_harmo, src_param)
        ## SubNoise
        base_noise = torch.rand_like(colored_harmo).to(noise_param) * 2 - 1
        colored_noise = frequency_filter(base_noise, noise_param)
        ## Harmo+Noise
        signal = colored_harmo + colored_noise

        return signal, f0, final_phase, (colored_harmo, colored_noise)


class SawSinSub(nn.Module):
    """Feature analyzer + Subtracted sawtooth (sinusoid approx.) + Subtracted noise."""
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
        ## Mixed NN
        ctrls = self.mel2ctrl(mel)
        f0_unscaled, noise_mag, harmo_mag         = ctrls['f0'], ctrls['noise_magnitude'], ctrls['harmonic_magnitude']
        ## f0 :: (B, Frame, 1) - Fundamental tone's frequency contour [Hz] (0 or within (80, 1000))
        f0 = unit_to_hz2(torch.sigmoid(f0_unscaled), hz_min = 80.0, hz_max=1000.0)
        f0[f0<80] = 0
        ## TODO: Fix bug. Check [f0 bug] in top of this file.

        # Feature transform
        ## fo - upsampling
        pitch = upsample(f0, self.block_size)
        ## noise magnitude :: (B, Frame, n_mag_noise) - scaling to (0, 2)
        noise_param = scale_function(noise_mag)
        ## harmonic magnitude :: (B, Frame, n_mag_harmonic) - scaling to (0, 2)
        src_param   = scale_function(harmo_mag)

        # sDSP
        ## SubHarmo
        base_harmo,    final_phase = self.harmonic_synthsizer(pitch,             initial_phase)
        colored_harmo = frequency_filter(base_harmo, src_param)
        ## SubNoise
        base_noise = torch.rand_like(colored_harmo).to(noise_param) * 2 - 1
        colored_noise = frequency_filter(base_noise, noise_param)
        ## Harmo+Noise
        signal = colored_harmo + colored_noise

        return signal, f0, final_phase, (colored_harmo, colored_noise)


class Sins(nn.Module):
    """Feature analyzer + Added sinusoids + Subtracted noise."""
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
        ## Mixed NN
        ctrls = self.mel2ctrl(mel)
        f0_unscaled, noise_mag,            A, amp = ctrls['f0'], ctrls['noise_magnitude'],                              ctrls['A'], ctrls['amplitudes']
        ## f0 :: (B, Frame, 1) - Fundamental tone's frequency contour [Hz] (0 or within (80, 1000))
        f0 = unit_to_hz2(torch.sigmoid(f0_unscaled), hz_min = 80.0, hz_max=1000.0)
        f0[f0<80] = 0
        ## TODO: Fix bug. Check [f0 bug] in top of this file.

        # Feature transform
        ## fo - upsampling
        pitch = upsample(f0, self.block_size)
        ## noise magnitude :: (B, Frame, n_mag_noise) - scaling to (0, 2)
        noise_param = scale_function(noise_mag)
        ## amplitudes - normalized_partial_wise * global + upsampling
        amp  = scale_function(amp)
        amp /= amp.sum(-1, keepdim=True) # - Normalized partial amplitudes
        amp *= scale_function(A) # :: * (B, Frame, 1) - * Global amplitude
        amplitudes = upsample(amp, self.block_size)

        # sDSP
        ## AddHarmo
        colored_harmo, final_phase = self.harmonic_synthsizer(pitch, amplitudes, initial_phase)
        colored_harmo = colored_harmo
        ## SubNoise
        base_noise = torch.rand_like(colored_harmo).to(noise_param) * 2 - 1
        colored_noise = frequency_filter(base_noise, noise_param)
        ## Harmo+Noise
        signal = colored_harmo + colored_noise

        return signal, f0, final_phase, (colored_harmo, colored_noise)


class DWS(nn.Module):
    """Feature analyzer + Wavetable + Subtracted noise."""
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
        ## Mixed NN
        ctrls = self.mel2ctrl(mel)
        f0_unscaled, noise_mag,            A, amp = ctrls['f0'], ctrls['noise_magnitude'],                              ctrls['A'], ctrls['amplitudes']
        ## f0 :: (B, Frame, 1) - Fundamental tone's frequency contour [Hz] (0 or within (80, 1000))
        f0 = unit_to_hz2(torch.sigmoid(f0_unscaled), hz_min = 80.0, hz_max=1000.0)
        f0[f0<80] = 0
        ## TODO: Fix bug. Check [f0 bug] in top of this file.

        # Feature transform
        ## fo - no modification
        pitch = f0
        ## noise magnitude :: (B, Frame, n_mag_noise) - scaling to (0, 2)
        noise_param = scale_function(noise_mag)
        ## amplitudes - normalized_partial_wise * global
        amp  = scale_function(amp)
        amp /= amp.sum(-1, keepdim=True) # - Normalized partial amplitudes
        amplitudes *= scale_function(A) # :: * (B, Frame, 1) - * Global amplitude

        # sDSP
        ## Wavetable
        colored_harmo, final_phase = self.harmonic_synthsizer(pitch, amplitudes, initial_phase)
        colored_harmo = colored_harmo
        ## SubNoise
        base_noise = torch.rand_like(colored_harmo).to(noise_param) * 2 - 1
        colored_noise = frequency_filter(base_noise, noise_param)
        ## Harmo+Noise
        signal = colored_harmo + colored_noise

        return signal, f0, final_phase, (colored_harmo, colored_noise)


class SawSub(nn.Module):
    """Feature analyzer + Subtracted sawtooth (exact, w/ aliasing) + Subtracted noise."""
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
        ## Mixed NN
        ctrls = self.mel2ctrl(mel)
        f0_unscaled, noise_mag, harmo_mag         = ctrls['f0'], ctrls['noise_magnitude'], ctrls['harmonic_magnitude']
        ## f0 :: (B, Frame, 1) - Fundamental tone's frequency contour [Hz] (0 or within (80, 1000))
        f0 = unit_to_hz2(torch.sigmoid(f0_unscaled), hz_min = 80.0, hz_max=1000.0)
        f0[f0<80] = 0
        ## TODO: Fix bug. Check [f0 bug] in top of this file.

        # Feature transform
        ## fo - upsampling
        pitch = upsample(f0, self.block_size)
        ## noise magnitude :: (B, Frame, n_mag_noise) - scaling to (0, 2)
        noise_param = scale_function(noise_mag)
        ## harmonic magnitude :: (B, Frame, n_mag_harmonic) - scaling to (0, 2)
        src_param   = scale_function(harmo_mag)

        # sDSP
        ## SubHarmo
        base_harmo,    final_phase = self.harmonic_synthsizer(pitch,             initial_phase)
        colored_harmo = frequency_filter(base_harmo, src_param)
        ## SubNoise
        base_noise = torch.rand_like(colored_harmo).to(noise_param) * 2 - 1
        colored_noise = frequency_filter(base_noise, noise_param)
        ## Harmo+Noise
        signal = colored_harmo + colored_noise

        return signal, f0, final_phase, (colored_harmo, colored_noise)
