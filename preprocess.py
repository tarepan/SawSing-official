import os
import numpy as np
import librosa as li
from librosa.filters import mel as librosa_mel_fn
import soundfile as sf
# import crepe

import torch
import shutil



def traverse_dir(root_dir, extension, amount=None, str_include=None, str_exclude=None, is_pure=False, is_sort=False, is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


class Audio2Mel(torch.nn.Module):
    def __init__(self, hop_length, sampling_rate, n_mel_channels, win_length=1024, n_fft=None, mel_fmin=0.0, mel_fmax=None):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.n_fft, self.hop_length, self.win_length = n_fft, hop_length, win_length
        self.n_mel_channels = n_mel_channels
        self.register_buffer("window", torch.hann_window(win_length).float())
        self.register_buffer("mel_basis", torch.from_numpy(librosa_mel_fn(sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax)).float())

    def forward(self, audio):
        '''
        Args:
            audio :: (B, Channel, T)
        Returns:
                  :: (B, Frame, Freq) - Log-amplitude Mel-frequency spectrogram
        '''
        B, C, T = audio.shape
        audio = audio.reshape(B * C, T)

        # Linear-amplitude Linear-frequency spectrogram
        ## TODO: `return_complex` is deprecated
        fft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, center=False, return_complex=False)
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)

        # Linear-amplitude Mel-frequency spectrogram
        mel_output = torch.matmul(self.mel_basis, magnitude)

        # Log-amplitude Mel-frequency spectrogram
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))

        #                        reshape                     permute                      squeeze 
        # (B*Channel, Freq, Frame) -> (B, Channel, Freq, Frame) -> (B, Frame, Channel, Freq) -> (B, Frame, Freq)
        log_mel_spec = log_mel_spec.reshape(B, C, self.n_mel_channels ,log_mel_spec.shape[-1]).permute(0, 3, 1, 2).squeeze(2)

        return log_mel_spec


def process_mel(path_srcdir, path_dstdir, device, sampling_rate, hop_length, win_length, n_mel_channels, src_ext, dst_ext):

    # list files
    filelist =  traverse_dir(path_srcdir, extension=(src_ext), is_pure=True, is_sort=True, is_ext=False)

    # initilize extractor
    mel_extractor = Audio2Mel(hop_length=hop_length, sampling_rate=sampling_rate, n_mel_channels=n_mel_channels, win_length=win_length).to(device)

    # run
    n_file = len(filelist)
    print(' > path_srcdir:', path_srcdir)
    print(' > num files:', n_file)
    for idx, file in enumerate(filelist):
        print('\n--- {idx}/{n_file} ----')

        path_srcfile = os.path.join(path_srcdir, file+'.'+src_ext) # f"{path_rootdir}/{spk}/{scenario}/audio/{file}.{src_ext}"
        path_dstfile = os.path.join(path_dstdir, file+'.'+dst_ext) # f"{path_rootdir}/{spk}/{scenario}/mel/{file}.{dst_ext}"
        print(' >  path src wav:', path_srcfile)
        print(' >  path dst mel:', path_dstfile)
        
        # Load
        x, sr = sf.read(path_srcfile)
        assert sr == sampling_rate
        x_t = torch.from_numpy(x).float().to(device)
        x_t = x_t.unsqueeze(0).unsqueeze(0) # :: (T,) -> (1, 1, T)

        # Preprocessing
        m_t = mel_extractor(x_t)

        # Save
        os.makedirs(os.path.dirname(path_dstfile), exist_ok=True)
        mel = m_t.squeeze().to('cpu').numpy()
        np.save(path_dstfile, mel)
        print(' > mel:', mel.shape)


if __name__ == '__main__':
    path_rootdir = './data'
    device = 'cuda'
    sampling_rate  = 24000
    hop_length     =   240
    win_length     =  1024
    n_mel_channels = 80
    src_ext, dst_ext = 'wav', 'npy'
    spks = ['solo',]
    scenarios = ['train-full', 'val', 'test']

    for spk in spks:
        for scenario in scenarios:
            print(f'=== {spk} - {scenario} =============')
            path_srcdir  = os.path.join(path_rootdir, spk, scenario, 'audio') # f"{path_rootdir}/{spk}/{scenario}/audio"
            path_dstdir  = os.path.join(path_rootdir, spk, scenario, 'mel')   # f"{path_rootdir}/{spk}/{scenario}/mel"
            process_mel(path_srcdir, path_dstdir, device, sampling_rate, hop_length, win_length, n_mel_channels, src_ext, dst_ext)
    