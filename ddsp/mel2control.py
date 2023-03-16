import gin


import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .pcmer import PCmer


def _split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors.
    
    Args:
        tensor
        tensor_splits
    Returns:
        Dict
            k: split(tensor)[i]
                k                - Key specified in `tensor_splits`
                split(tensor)[i] - Split of `tensor`, size `v` specified in `tensor_splits`)            
    """
    labels = []
    sizes = []

    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)

    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))


class Mel2Control(nn.Module):
    """
    Convert mel-spectrogram to DSP-control parameters (DDSP-derived naming).
    Model: Conv1d-GN-LReLU-Conv1d-Conformer-LN-Linear
    """
    def __init__(self, input_channel, output_splits):
        """
        Args:
            output_splits :: list[] - Tensor split specifiers
        """
        super().__init__()
        self.output_splits = output_splits
        n_out = sum([v for k, v in output_splits.items()])

        # PreNet/Conformer/PostNet
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 64, 3, padding="same"),
                nn.GroupNorm(4, 64),
                nn.LeakyReLU(),
                nn.Conv1d(           64, 64, 3, padding="same")) 
        self.decoder = PCmer(num_layers=3, num_heads=8, dim_model=64, dim_keys=64, dim_values=64, residual_dropout =0.1, attention_dropout=0.1)
        self.norm = nn.LayerNorm(64)
        self.dense_out = weight_norm(nn.Linear(64, n_out))

    def forward(self, x):
        '''
        Args:
            x :: (B, Frame, Freq)
        returns:
            dict of (B, Frame, Feat)
        '''
        x = self.stack(x.transpose(1,2)).transpose(1,2)
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)

        return _split_to_dict(e, self.output_splits)
