import torch
import argparse

from torch import Tensor, nn
from torch.nn import functional as F
from typing import Tuple


class TorchDF(nn.Module):
    def __init__(self, model, sr=48000, fft_size=960, hop_size=480, 
                 nb_bands=32, min_nb_freqs=2, nb_df=96, alpha=0.99):
        """
        Torch Only Version, Can't be exported to ORT:
        - Complex type
        - rfft 
        - fors
        """
        super().__init__()

        assert hop_size * 2 == fft_size
        self.fft_size = fft_size
        self.frame_size = hop_size # f
        self.window_size = fft_size 
        self.window_size_h = fft_size // 2
        self.freq_size = fft_size // 2 + 1 # F
        self.wnorm = 1. / (self.window_size ** 2 / (2 * self.frame_size))

        # Initialize the vorbis window: sin(pi/2*sin^2(pi*n/N))
        self.register_buffer('window', torch.zeros(self.fft_size))
        self.window = torch.sin(
            0.5 * torch.pi * (torch.arange(self.fft_size) + 0.5) / self.window_size_h
        )
        self.window = torch.sin(0.5 * torch.pi * self.window ** 2)
        
        self.sr = sr
        self.min_nb_freqs = min_nb_freqs
        self.nb_df = nb_df

        # Initializing erb features
        self.erb_indices = torch.tensor([
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8, 
            10, 12, 13, 15, 18, 20, 24, 28, 31, 37, 42, 50, 56, 67
        ])
        self.n_erb_features = nb_bands

        self.erb_conv = nn.Conv1d(self.freq_size, self.n_erb_features, 1, bias=False).requires_grad_(False)
        out_weight = torch.zeros(self.n_erb_features, self.freq_size, 1)

        start_index = 0
        for i, num in enumerate(self.erb_indices):
            out_weight[i, start_index: start_index + num, 0] = 1 / num
            start_index += num

        self.erb_conv.weight.copy_(out_weight)

        # Normalization
        self.mean_norm_init = [-60., -90.]
        self.unit_norm_init = [0.001, 0.0001]

        # Model
        self.model = model

        # Alpha
        self.alpha = alpha

        # Init Buffers
        self.register_buffer('analysis_mem', torch.zeros(self.fft_size - self.frame_size))
        self.register_buffer('synthesis_mem', torch.zeros(self.fft_size - self.frame_size))
        self.register_buffer('band_unit_norm_state', torch.linspace(
            self.unit_norm_init[0], self.unit_norm_init[1], self.nb_df
        ))
        self.register_buffer('erb_norm_state', torch.linspace(
            self.mean_norm_init[0], self.mean_norm_init[1], self.n_erb_features
        ))
        
    
    def analysis_time(self, input_data: Tensor) -> Tensor:
        """
        Original code - pyDF/src/lib.rs - analysis()
        Calculating spectrogram for each frame on T. Every frame is concated with previous frame. 
        So rfft takes on input frame on window_size.

        Parameters:
            input_data: Float[T] - Input raw audio

        Returns:
            output:     Complex[T, F] - Spectrogram for every frame
        """
        in_chunks = torch.split(input_data, self.frame_size)

        # time chunks iteration
        output = []
        for ichunk in in_chunks:
            output.append(self.frame_analysis(ichunk))

        return torch.stack(output, dim=0)
        
    def frame_analysis(self, input_frame: Tensor) -> Tensor:
        """
        Original code - libDF/src/lib.rs - frame_analysis()
        Calculating spectrograme for one frame. Every frame is concated with buffer from previous frame.

        Parameters:
            input_frame:    Float[f] - Input raw audio frame
        
        Returns:
            output:         Complex[F] - Spectrogram
        """
        # First part of the window on the previous frame
        # Second part of the window on the new input frame
        buf = torch.cat([self.analysis_mem, input_frame]) * self.window
        buf_fft = torch.fft.rfft(buf, norm='backward') * self.wnorm

        # Copy input to analysis_mem for next iteration
        self.analysis_mem = input_frame
        
        return buf_fft
    
    def erb(self, input_data: Tensor, db=True) -> Tensor:
        """
        Original code - pyDF/src/lib.rs - erb()
        Calculating ERB features for each frame.

        Parameters:
            input_data:     Float[T, F] or Float[F] - audio spectrogram 

        Returns:
            erb_features:   Float[T, ERB] or Float[ERB] - erb features for given spectrogram
        """

        magnitude_squared = input_data.real ** 2 + input_data.imag ** 2
        erb_features = self.erb_conv(magnitude_squared.unsqueeze(-1)).squeeze(-1)

        # Convergins given features into DB scale
        if db:
            erb_features = 10.0 * torch.log10(erb_features + 1e-10)

        return erb_features

    def erb_norm_time(self, input_data: Tensor, alpha: float = 0.9) -> Tensor:
        """
        Original code - libDF/src/transforms.rs - erb_norm()
        Normalizing ERB features. And updates the normalization state on every step.

        Parameters:
            input_data:     Float[T, ERB] - erb features
            alpha:          float - alpha value

        Returns:
            output:         Float[T, ERB] - normalized erb features
        """
        
        output = []
        for in_step in input_data:
            output.append(self.band_mean_norm_erb(in_step, alpha))

        return torch.stack(output, dim=0)

    def band_mean_norm_erb(self, xs: Tensor, alpha: float, denominator: float = 40.0):
        """
        Original code - libDF/src/lib.rs - band_mean_norm()
        Normalizing ERB features. And updates the normalization state.

        Parameters:
            xs:             Float[ERB] - erb features
            alpha:          float - alpha value which is needed for adaptation of the normalization state for given scale.
            denominator:    float - denominator for normalization

        Returns:
            output:         Float[ERB] - normalized erb features
        """

        self.erb_norm_state = torch.lerp(xs, self.erb_norm_state, alpha)
        output = (xs - self.erb_norm_state) / denominator
        
        return output

    def unit_norm_time(self, input_data: Tensor, alpha: float = 0.9) -> Tensor:
        """
        Original code - libDF/src/transforms.rs - unit_norm()
        Normalizing Deep Filtering features. And updates the normalization state for every step.

        Parameters:
            input_data:     Complex[T, DF] - deep filtering features
            alpha:          float - alpha value

        Returns:
            output:         Complex[T, DF] - normalized deep filtering features
        """
        output = []

        for in_step in input_data:
            output.append(self.band_unit_norm(in_step, alpha))

        return torch.stack(output, dim=0)

    def band_unit_norm(self, xs: Tensor, alpha: float) -> Tensor:
        """
        Original code - libDF/src/lib.rs - band_unit_norm()
        Normalizing Deep Filtering features. And updates the normalization state.

        Parameters:
            xs:             Complex[DF] - deep filtering features
            alpha:          float - alpha value which is needed for adaptation of the normalization state for given scale.

        Returns:
            output:         Complex[DF] - normalized deep filtering features
        """
        self.band_unit_norm_state = torch.lerp(xs.abs(), self.band_unit_norm_state, alpha)
        output = xs / self.band_unit_norm_state.sqrt()
        
        return output

    def synthesis_time(self, input_data: Tensor) -> Tensor:
        """
        Original code - pyDF/src/lib.rs - synthesis()
        Inverse rfft for each frame. Every frame is summarized with buffer from previous frame.

        Parameters:
            input_data: Complex[T, F] - Enhanced audio spectrogram

        Returns:
            output:     Float[Tr] - Enhanced audio
        """
        out_chunks = []

        # time iteration
        for ichunk in input_data:
            output_frame = self.frame_synthesis(ichunk)
            out_chunks.append(output_frame)

        return torch.cat(out_chunks)

    def frame_synthesis(self, input_data):
        """
        Original code - libDF/src/lib.rs - frame_synthesis()
        Inverse rfft for one frame. Every frame is summarized with buffer from previous frame.
        And saving buffer for next frame.

        Parameters:
            input_data: Complex[F] - Enhanced audio spectrogram

        Returns:
            output:     Float[f] - Enhanced audio
        """
        x = torch.fft.irfft(input_data, norm='forward') * self.window

        x_first, x_second = torch.split(x, [self.frame_size, x.shape[0] - self.frame_size])
        output = x_first + self.synthesis_mem 

        self.synthesis_mem = x_second

        return output

    def df_features(self, audio: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Original code - DeepFilterNet/df/enhance - df_features()
        Generating the features for audio.
        B = 1, CH = 1 implementation only.

        Parameters:
            audio:      Float[Tf] - Input raw audio 

        Returns:
            spec:       Float[B, CH, T, F, 2] - Audio spectrogram for each frame, concated with previous frame. Complex as real.
            erb_feat:   Float[B, CH, T, ERB] - Normalized ERB features.
            spec_feat:  Float[B, CH, T, DF, 2] - Normalized deep filtering features. Complex as real.
        """

        spec = self.analysis_time(audio)
        erb_feat = self.erb_norm_time(self.erb(spec), alpha=self.alpha)
        spec_feat = torch.view_as_real((self.unit_norm_time(spec[..., :self.nb_df], alpha=self.alpha)))
        spec = torch.view_as_real((spec))

        return spec.view(1, 1, *spec.shape), \
                erb_feat.view(1, 1, *erb_feat.shape), \
                spec_feat.view(1, 1, *spec_feat.shape)
    
    @torch.no_grad()
    def forward(self, audio: Tensor, atten_lim_db: float = None, pad: bool = True, normalize_atten_lim=20) -> Tensor:
        """
        Original code - DeepFilterNet/df/enhance.py - enhance()
        Enhancing the audio using offline processing
        
        Parameters:
            audio:          Float[Tf]
            atten_lim_db:   float - attenuation limit in dB. How much noise should we mix in the enhanced audio
            pad:            bool - if True, pad audio to compensate for the delay due to the real-time STFT implementation

        Returns:
            enhanced_audio: Float[Tf]
        """
        orig_len = audio.shape[-1]

        if pad:
            # Pad audio to compensate for the delay due to the real-time STFT implementation
            hop_size_divisible_padding_size = (self.fft_size - orig_len % self.fft_size) % self.fft_size
            orig_len += hop_size_divisible_padding_size
            audio = F.pad(audio, (0, self.fft_size + hop_size_divisible_padding_size))
            assert audio.shape[-1] % self.fft_size == 0

        spec, erb_feat, spec_feat = self.df_features(audio)
        enhanced = self.model(spec, erb_feat, spec_feat)[0] # [B=1, CH=1, T, F, 2]

        if atten_lim_db is not None and abs(atten_lim_db) > 0:
            lim = 10 ** (-abs(atten_lim_db) / normalize_atten_lim)
            enhanced = torch.lerp(enhanced, spec, lim)

        enhanced = torch.view_as_complex(enhanced.view(*enhanced.shape[2:]))
        audio = self.synthesis_time(enhanced).unsqueeze(0)
        if pad:
            # The frame size is equal to p.hop_size. Given a new frame, the STFT loop requires e.g.
            # ceil((n_fft-hop)/hop). I.e. for 50% overlap, then hop=n_fft//2
            # requires 1 additional frame lookahead; 75% requires 3 additional frames lookahead.
            # Thus, the STFT/ISTFT loop introduces an algorithmic delay of n_fft - hop.
            assert self.fft_size % self.frame_size == 0  # This is only tested for 50% and 75% overlap
            d = self.fft_size - self.frame_size
            audio = audio[:, d : orig_len + d]

        return audio
    

def main(args):
    import os
    import glob
    import torchaudio

    from tqdm import tqdm
    from df import init_df

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    model, _, _ = init_df(config_allow_defaults=True, model_base_dir='DeepFilterNet3')
    model.to('cpu')
    model.eval()

    torch_offline = TorchDF(sr=48000, nb_bands=32, min_nb_freqs=2, hop_size=480, fft_size=960, model=model)
    torch_offline = torch_offline.to('cpu')

    print(f'Reading audio from folder - {args.input_folder}')
    clips = glob.glob(os.path.join(args.input_folder, "*.wav")) + glob.glob(os.path.join(args.input_folder, "*.flac"))
    assert len(clips) > 0, f"Not wound wav or flac in folder {args.input_folder}"
    print(f'Found {len(clips)} audio in {args.input_folder}')

    print(f'Inferencing model to folder - {args.output_folder}...')
    os.makedirs(args.output_folder, exist_ok=False)

    for clip in tqdm(clips):
        noisy_audio, _ = torchaudio.load(clip, channels_first=True)
        enhanced_audio = torch_offline(noisy_audio.mean(dim=0))
        save_path = os.path.join(args.output_folder, 'denoised_' + os.path.basename(clip))
        torchaudio.save(
            save_path, 
            enhanced_audio.data.cpu(),
            48000, 
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-folder', help='path with folder for inference', required=True
    )
    parser.add_argument(
        '--output-folder', help='path to save enhanced audio'
    )
    main(parser.parse_args())
