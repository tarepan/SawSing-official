import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import numpy as np


#### unit_to_hz2 ######################################################################################
def _hz_to_midi(frequencies: float):
    """Scale linear frequency scale [Hz] to logarithmic MIDI scale [NoteNumber] (TF-compatible).
    
    Returns:
        :: Tensor
    """
    freq, ref = torch.tensor(frequencies, dtype=torch.float), torch.tensor(440., dtype=torch.float)
    # MIDI scale with non-negative rounding
    return F.relu(12. * (torch.log2(freq) - torch.log2(ref)) + 69.)

def _midi_to_hz(notes):
    """Scale logarithmic MIDI scale [NoteNumber] to linear frequency scale [Hz] (TF-compatible).
  
    Args:
        notes :: Tensor
    """
  return 440.0 * (2.0**((notes - 69.0) / 12.0))


def unit_to_hz2(unit, min_hz: float, max_hz: float) :
    """Map `unit` ∈ (0, 1) to 'frequency' ∈ (min_hz, max_hz) logarithmically (linear interpolation in MIDI-scale).

    Args:
        unit :: Tensor - Value to be scaled, within (0, 1)
    Returns:
             :: Tensor - Value scaled, within (min_hz, max_hz)
    """
    min_midi, max_midi = _hz_to_midi(min_hz), _hz_to_midi(max_hz)
    return _midi_to_hz(min_midi + (max_midi - min_midi) * unit)
#### /unit_to_hz2 #####################################################################################


#### Common functions #################################################################################
def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
    return signal.permute(0, 2, 1)


def remove_above_nyquist(amplitudes, pitch, sampling_rate):
    """Remove partials above nyquist frequency.

    Args:
        amplitudes    :: (B, T, Partial) - Partial's amplitude contour
        pitch         :: (B, T, Partial) - f0 contour [Hz]
        sampling_rate :: int             - Sampling rate (not nyquist frequency)
    Returns:
        amplitudes    :: (B, T, Partial) - Masked amplitude
    """
    n_harm = amplitudes.shape[-1]
    # Partial's frequency :: (Partial,) - k*fo [hz]
    pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
    # zero-mask above nyqust frequency
    aa = (pitches < sampling_rate / 2).float() + 1e-7
    return amplitudes * aa


def scale_function(x):
    """
                 σ          **log         *2
    (-inf, +inf) -> (0, 1) ------> (0, 1) --> (0, 2)
    """
    return 2 * torch.sigmoid(x)**(np.log(10)) + 1e-7
#### /Common functions ################################################################################


#### for WavetableSynthesizer #########################################################################
def linear_lookup(index, wavetable):
    '''
    Args:
        index: B x T, value: [0, 1]
        wavetables: B x num_wavetables x len_wavetable
    Returns:
        singal: B x num_wavetables x T
    '''
    wavetable = torch.cat([wavetable, wavetable[..., 0:1]], dim=-1)
    wavetable = wavetable.unsqueeze(-1)        # B x C x L x 1
    
    index = index.unsqueeze(-1) * 2 - 1        # [0, 1] -> [-1, 1]
    index_ = torch.zeros_like(index) 
    index = torch.cat([index_, index], dim=-1) # B x L x 2
    index = index.unsqueeze(2)                 # B x L x 1 x 2

    signal = F.grid_sample(wavetable, grid=index, mode='bilinear', align_corners=True).squeeze(-1)
    return signal
#### /for WavetableSynthesizer ########################################################################


#### frequency_filter #################################################################################
def _get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
  """Calculate final size for efficient FFT.
  Args:
    frame_size: Size of the audio frame.
    ir_size: Size of the convolving impulse response.
    power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
      numbers. TPU requires power of 2, while GPU is more flexible.
  Returns:
    fft_size: Size for efficient FFT.
  """
  convolved_frame_size = ir_size + frame_size - 1
  if power_of_2:
    # Next power of 2.
    fft_size = int(2**np.ceil(np.log2(convolved_frame_size)))
  else:
    fft_size = int(fftpack.helper.next_fast_len(convolved_frame_size))
  return fft_size


def _fft_convolve(audio,
                 impulse_response,
                 padding = 'same',
                 delay_compensation = -1,
                 mel_scale_noise = False):
    """Filter audio with frames of time-varying impulse responses.
    Given audio (B, n_samples), and a series of impulse responses (B, n_frames, n_impulse_response]), execute:
        1. splits the audio into frames
        2. applies filters
        3. overlap-and-adds audio back together
    Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute convolution for large impulse response sizes.
    Args:
        audio            :: (B, T)                                 - Input audio
        impulse_response :: (B, ir_size) | (B, ir_frames, ir_size) - The FIR, 2D for LTI, 3D for LTV
            Automatically chops the audio into equally shaped blocks to match ir_frames.
        padding          :: 'valid' | 'same'                       -
            For 'same' the final output to be the same size as the input audio (T)
            For 'valid' the audio is extended to include the tail of the impulse response (T + ir_timesteps - 1).
        delay_compensation                                         - Samples to crop from start of output audio to compensate for group delay of the impulse response.
            If delay_compensation is less than 0 it defaults to automatically calculating a constant group delay of the windowed linear phase filter from frequency_impulse_response().
    Returns:
        :: (B, T + ir_timesteps - 1) | (B, T) - Filtered audio by 'valid' padding | 'same' padding
    Raises:
        ValueError: If audio and impulse response have different batch size.
        ValueError: If audio cannot be split into evenly spaced frames. (i.e. the
        number of impulse response frames is on the order of the audio size and
        not a multiple of the audio size.)
    """
    # LTI-to-psuedoLTV :: (B, ir_size) | (B, ir_frames, ir_size) -> (B, ir_frames=1|f, ir_size)
    ir_shape = impulse_response.size()
    if len(ir_shape) == 2:
        impulse_response = impulse_response.unsqueeze(1)
        ir_shape = impulse_response.size()

    # Get shapes of audio and impulse response.
    batch_size_ir, n_ir_frames, ir_size = ir_shape
    batch_size, audio_size = audio.size()#shape.as_list()

    # Validate that batch sizes match.
    if batch_size != batch_size_ir:
        raise ValueError('Batch size of audio ({}) and impulse response ({}) must '
                        'be the same.'.format(batch_size, batch_size_ir))

    # Cut audio into frames.
    frame_size = int(np.ceil(audio_size / n_ir_frames))
    hop_size = frame_size
    
    # audio --> batch, time_step
    audio = audio.unsqueeze(1)
    if frame_size!=audio_size:
        filters = torch.eye(frame_size).unsqueeze(1).cuda()
        audio_frames = F.conv1d(audio, filters, stride= hop_size).transpose(1,2)
        n_audio_frams = audio_frames.size(1)
    else:
        audio_frames = audio#.transpose(1,2)
    # Check that number of frames match.
    #print ("audio_frames here:", audio_frames.size())
    n_audio_frames = int(audio_frames.shape[1])

    if n_audio_frames != n_ir_frames:
        raise ValueError(
            'Number of Audio frames ({}) and impulse response frames ({}) do not '
            'match. For small hop size = ceil(audio_size / n_ir_frames), '
            'number of impulse response frames must be a multiple of the audio '
            'size.'.format(n_audio_frames, n_ir_frames))
    
    # Pad and FFT the audio and impulse responses.
    fft_size = _get_fft_size(frame_size, ir_size, power_of_2=True)

    # Convolution through fourier-space    
    audio_fft = torch.fft.rfft(audio_frames,     fft_size)
    ir_fft    = torch.fft.rfft(impulse_response, fft_size)
    audio_ir_fft = torch.multiply(audio_fft, ir_fft)
    audio_frames_out = torch.fft.irfft(audio_ir_fft)

    #audio_out = torch.signal.overlap_and_add(audio_frames_out, hop_size)
    if frame_size!=audio_size:
        overlap_add_filter = torch.eye(audio_frames_out.size(-1), requires_grad = False).unsqueeze(1).cuda()
        #print (overlap_add_filter.size())
        #print (audio_frames_out.size(), overlap_add_filter.size())
        output_signal = F.conv_transpose1d(audio_frames_out.transpose(1, 2), 
                                                        overlap_add_filter, 
                                                        stride = frame_size, 
                                                        padding = 0).squeeze(1)
    else:
        output_signal = crop_and_compensate_delay(audio_frames_out.squeeze(1), audio_size, ir_size, padding,
                                    delay_compensation)
    return output_signal[...,:frame_size*n_ir_frames]
    # Crop and shift the output audio.


def _apply_window_to_impulse_response(impulse_response,
                                     window_size: int = 0,
                                     causal: bool = False):
    """Apply a window to an impulse response and put in causal form.

    Args:
        impulse_response: A series of impulse responses frames to window, of shape [batch, n_frames, ir_size]. ---------> ir_size means size of filter_bank ??????
        window_size: Size of the window to apply in the time domain.
            If window_size is less than 1, it defaults to the impulse_response size.
        causal: Impulse response input is in causal form (peak in the middle).
    Returns:
        impulse_response: Windowed impulse response in causal form, with last dimension cropped to window_size if window_size is greater than 0 and less than ir_size.
    """
    
    
    # If IR is in causal form, put it in zero-phase form.
    if causal:
        impulse_response = torch.fftshift(impulse_response, axes=-1)
    
    # Get a window for better time/frequency resolution than rectangular.
    # Window defaults to IR size, cannot be bigger.
    ir_size = int(impulse_response.size(-1))
    if (window_size <= 0) or (window_size > ir_size):
        window_size = ir_size
    window = nn.Parameter(torch.hann_window(window_size), requires_grad = False).cuda()
    # Zero pad the window and put in in zero-phase form.
    
    padding = ir_size - window_size
    if padding > 0:
        half_idx = (window_size + 1) // 2
        window = torch.cat([window[half_idx:],
                            torch.zeros([padding]),
                            window[:half_idx]], axis=0)
    else:
        window = window.roll((window.size(-1)+1)//2, -1)
        
    # Apply the window, to get new IR (both in zero-phase form).
    window = window.unsqueeze(0)
    impulse_response = impulse_response*window
    
    # Put IR in causal form and trim zero padding.
    if padding > 0:
        first_half_start = (ir_size - (half_idx - 1)) + 1
        second_half_end = half_idx + 1
        impulse_response = torch.cat([
                impulse_response[..., first_half_start:               ],
                impulse_response[...,                 :second_half_end],
            ], dim=-1)
    else:
        impulse_response = impulse_response.roll((impulse_response.size(-1)+1)//2, -1)

    return impulse_response


def _frequency_impulse_response(magnitudes, window_size: int = 0):
    """Get windowed impulse responses using the frequency sampling method.
    Follows the approach in: https://ccrma.stanford.edu/~jos/sasp/Windowing_Desired_Impulse_Response.html

    Args:
        magnitudes :: (B, n_frames, n_frequencies) | (B, n_frequencies) - Frequency transfer curve. 
            The frequencies of the last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ..., f_nyquist] (`f_nyquist=sr/2`).
            Automatically splits the audio into equally sized frames to match frames in magnitudes.
        window_size                                                     - Size of the window to apply in the time domain.
            If window_size is less than 1, it defaults to the impulse_response size.
    Returns:
        impulse_response :: (B, frames, window_size) | (B, window_size) - Time-domain FIR filter
        
    Raises:
        ValueError: If window size is larger than fft size.
    """
    # Get the IR (zero-phase form).
    
    magnitudes = torch.complex(magnitudes, torch.zeros_like(magnitudes))
    impulse_response = torch.fft.irfft(magnitudes)
    
    """ This means this?
    First: Initilize at fourier space, where real part equal magnitueds, complex part equal 0
    Second: Convert back to time domain
    """ 
    
    # Window and put in causal form.
    impulse_response = _apply_window_to_impulse_response(impulse_response, impulse_response.size(-1))
    return impulse_response


def frequency_filter(audio,
                     magnitudes,
                     window_size = 0,
                     padding = 'same',
                     mel_scale_noise = False,
                     mel_basis = None):
    """Filter audio with a finite impulse response filter.
    Args:
        audio: Input audio. Tensor of shape [batch, audio_timesteps].
        magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
        n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
        last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
        f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
        audio into equally sized frames to match frames in magnitudes.
        window_size: Size of the window to apply in the time domain. If window_size
        is less than 1, it is set as the default (n_frequencies).
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
        same size as the input audio (audio_timesteps). For 'valid' the audio is
        extended to include the tail of the impulse response (audio_timesteps +
        window_size - 1).
    Returns:
        Filtered audio. Tensor of shape
            [batch, audio_timesteps + window_size - 1] ('valid' padding) or shape
            [batch, audio_timesteps] ('same' padding).
    """

    impulse_response = _frequency_impulse_response(magnitudes, window_size=window_size)
    return _fft_convolve(audio, impulse_response, padding=padding, mel_scale_noise=mel_scale_noise)
#### /frequency_filter ################################################################################
