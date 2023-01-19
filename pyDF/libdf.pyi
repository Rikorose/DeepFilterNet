from typing import List, Optional, Union

from numpy import ndarray

class DF:
    def __init__(
        self,
        sr: int,
        fft_size: int,
        hop_size: int,
        nb_bands: int,
        min_nb_erb_freqs: Optional[int] = 1,
    ):
        """DeepFilter state used for analysis and synthesis.

        Args:
            sr (int): Sampling rate.
            fft_size (int): Window length used for the Fast Fourier transform.
            hop_size (int): Hop size between two analysis windows. Also called frame size.
            nb_bands (int): Number of ERB bands.
            min_nb_erb_freqs (int): Minimum number of frequency bands per ERB band. Defaults to 1.
        """
        ...
    def analysis(self, input: ndarray) -> ndarray:
        """Analysis of a time-domain signal.

        Args:
            input (ndarray): 2D real-valued array of shape [C, T].
            reset (bool): Reset STFT buffers before processing. Defaults to `true`.
        Output:
            output (ndarray): 3D complex-valued array of shape [C, T', F], where F is the `fft_size`,
                and T' the original time T divided by `hop_size`.
        """
        ...
    def synthesis(self, input: ndarray) -> ndarray:
        """Synthesis of a frequency-domain signal.

        Args:
            input (ndarray): 3D complex-valued array of shape [C, T, F].
            reset (bool): Reset STFT buffers before processing. Defaults to `true`.
        Output:
            output (ndarray): 2D real-valued array of shape [C, T].
        """
        ...
    def erb_widths(self) -> ndarray: ...
    def fft_window(self) -> ndarray: ...
    def sr(self) -> int: ...
    def fft_size(self) -> int: ...
    def hop_size(self) -> int: ...
    def nb_erb(self) -> int: ...
    def reset(self) -> None: ...

def erb(input: ndarray, erb_fb: Union[ndarray, List[int]], db: bool = True) -> ndarray:
    """ERB filterbank and transform to decibel scale.

    Args:
        input (array): Spectrum of shape [B, C, T, F]
        erb_fb (array): ERB filterbank array of shape [B] containing the ERB widths,
            where B are the number of ERB bins
        db (bool): Whether to transform the output into decibel scale. Defaults to `True`.
    """
    ...

def erb_inv(input: ndarray, erb_fb: Union[ndarray, List[int]]) -> ndarray: ...
def erb_norm(erb: ndarray, alpha: float, state: Optional[ndarray] = None) -> ndarray: ...
def unit_norm(spec: ndarray, alpha: float, state: Optional[ndarray] = None) -> ndarray: ...
def unit_norm_init(num_freq_bins: int) -> ndarray: ...
