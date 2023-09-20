"""
ONNX exportable classes
"""
import math
import torch
import argparse
import torchaudio

from torch.nn import functional as F

from torch import nn
from torch import Tensor
from typing import Tuple

from df import init_df


class ExportableStreamingTorchDF(nn.Module):
    def __init__(self, fft_size, hop_size, nb_bands,
                 enc, df_dec, erb_dec, df_order=5, lookahead=2,
                 conv_lookahead=2, nb_df=96, alpha=0.99, 
                 min_db_thresh=-10.0,
                 max_db_erb_thresh=30.0,
                 max_db_df_thresh=20.0,
                 normalize_atten_lim=20.0,
                 silence_thresh=1e-7,
                 sr=48000,
                 always_apply_all_stages=False,
                 ):
        # All complex numbers are stored as floats for ONNX compatibility
        super().__init__()
        
        self.fft_size = fft_size
        self.frame_size = hop_size # dimension "f" in Float[f]
        self.window_size = fft_size 
        self.window_size_h = fft_size // 2
        self.freq_size = fft_size // 2 + 1 # dimension "F" in Float[F]
        self.wnorm = 1. / (self.window_size ** 2 / (2 * self.frame_size))
        self.df_order = df_order
        self.lookahead = lookahead
        self.always_apply_all_stages = torch.tensor(always_apply_all_stages)
        self.sr = sr

        # Initialize the vorbis window: sin(pi/2*sin^2(pi*n/N))
        window = torch.sin(
            0.5 * torch.pi * (torch.arange(self.fft_size) + 0.5) / self.window_size_h
        )
        window = torch.sin(0.5 * torch.pi * window ** 2)
        self.register_buffer('window', window)
        
        self.nb_df = nb_df

        # Initializing erb features
        self.erb_indices = torch.tensor([
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8, 
            10, 12, 13, 15, 18, 20, 24, 28, 31, 37, 42, 50, 56, 67
        ])
        self.nb_bands = nb_bands

        self.register_buffer('forward_erb_matrix', self.erb_fb(self.erb_indices, normalized=True, inverse=False))
        self.register_buffer('inverse_erb_matrix', self.erb_fb(self.erb_indices, normalized=True, inverse=True))

        # Model
        self.enc = enc
        # Instead of padding we put tensor with buffers into encoder
        # I didn't checked receptived fields of convolution, but equallity tests are working
        self.enc.erb_conv0 = self.remove_conv_block_padding(self.enc.erb_conv0)
        self.enc.df_conv0 = self.remove_conv_block_padding(self.enc.df_conv0)

        # Instead of padding we put tensor with buffers into df_decoder
        self.df_dec = df_dec
        self.df_dec.df_convp = self.remove_conv_block_padding(self.df_dec.df_convp)

        self.erb_dec = erb_dec
        self.alpha = alpha

        # RFFT
        # FFT operations are performed as matmuls for ONNX compatability
        self.register_buffer('rfft_matrix', torch.view_as_real(torch.fft.rfft(torch.eye(self.window_size))).transpose(0, 1))
        self.register_buffer('irfft_matrix', torch.linalg.pinv(self.rfft_matrix))

        # Thresholds
        self.register_buffer('min_db_thresh', torch.tensor([min_db_thresh]))
        self.register_buffer('max_db_erb_thresh', torch.tensor([max_db_erb_thresh]))
        self.register_buffer('max_db_df_thresh', torch.tensor([max_db_df_thresh]))
        self.normalize_atten_lim = torch.tensor(normalize_atten_lim)
        self.silence_thresh = torch.tensor(silence_thresh)
        self.linspace_erb = [-60., -90.]
        self.linspace_df = [0.001, 0.0001]

        self.erb_norm_state_shape = (self.nb_bands, )
        self.band_unit_norm_state_shape = (1, self.nb_df, 1) # [bs=1, nb_df, mean of complex value = 1]
        self.analysis_mem_shape = (self.frame_size, )
        self.synthesis_mem_shape = (self.frame_size, )
        self.rolling_erb_buf_shape = (1, 1, conv_lookahead + 1, self.nb_bands) # [B, 1, conv kernel size, nb_bands]
        self.rolling_feat_spec_buf_shape = (1, 2, conv_lookahead + 1, self.nb_df) # [B, 2 - complex, conv kernel size, nb_df]
        self.rolling_c0_buf_shape = (1, self.enc.df_conv0_ch, self.df_order, self.nb_df) # [B, conv hidden, df_order, nb_df]
        self.rolling_spec_buf_x_shape = (max(self.df_order, conv_lookahead), self.freq_size, 2) # [number of specs to save, ...]
        self.rolling_spec_buf_y_shape = (self.df_order + conv_lookahead, self.freq_size, 2) # [number of specs to save, ...]
        self.enc_hidden_shape = (1, 1, self.enc.emb_dim) # [n_layers=1, batch_size=1, emb_dim]
        self.erb_dec_hidden_shape = (2, 1, self.erb_dec.emb_dim) # [n_layers=2, batch_size=1, emb_dim]
        self.df_dec_hidden_shape = (2, 1, self.df_dec.emb_dim) # [n_layers=2, batch_size=1, emb_dim]

        # States
        state_shapes = [
            self.erb_norm_state_shape,
            self.band_unit_norm_state_shape,
            self.analysis_mem_shape,
            self.synthesis_mem_shape,
            self.rolling_erb_buf_shape,
            self.rolling_feat_spec_buf_shape,
            self.rolling_c0_buf_shape,
            self.rolling_spec_buf_x_shape,
            self.rolling_spec_buf_y_shape,
            self.enc_hidden_shape,
            self.erb_dec_hidden_shape,
            self.df_dec_hidden_shape
        ]
        self.state_lens = [
            math.prod(x) for x in state_shapes
        ]
        self.states_full_len = sum(self.state_lens)

        # Zero buffers
        self.register_buffer('zero_gains', torch.zeros(self.nb_bands))
        self.register_buffer('zero_coefs', torch.zeros(self.rolling_c0_buf_shape[2], self.nb_df, 2))

    @staticmethod
    def remove_conv_block_padding(original_conv: nn.Module) -> nn.Module:
        """
        Remove paddings for convolutions in the original model

        Parameters:
            original_conv:  nn.Module - original convolution module
        
        Returns:
            output:         nn.Module - new convolution module without paddings
        """
        new_modules = []

        for module in original_conv:
            if not isinstance(module, nn.ConstantPad2d):
                new_modules.append(module)
                
        return nn.Sequential(*new_modules)
    
    def erb_fb(self, widths: Tensor, normalized: bool = True, inverse: bool = False) -> Tensor:
        """
        Generate the erb filterbank
        Taken from https://github.com/Rikorose/DeepFilterNet/blob/fa926662facea33657c255fd1f3a083ddc696220/DeepFilterNet/df/modules.py#L206
        Numpy removed from original code

        Parameters:
            widths:     Tensor - widths of the erb bands
            normalized: bool - normalize to constant energy per band
            inverse:    bool - inverse erb filterbank

        Returns:
            fb:         Tensor - erb filterbank
        """
        n_freqs = int(torch.sum(widths))
        all_freqs = torch.linspace(0, self.sr // 2, n_freqs + 1)[:-1]

        b_pts = torch.cumsum(torch.cat([torch.tensor([0]), widths]), dtype=torch.int32, dim=0)[:-1]

        fb = torch.zeros((all_freqs.shape[0], b_pts.shape[0]))
        for i, (b, w) in enumerate(zip(b_pts.tolist(), widths.tolist())):
            fb[b : b + w, i] = 1

        # Normalize to constant energy per resulting band
        if inverse:
            fb = fb.t()
            if not normalized:
                fb /= fb.sum(dim=1, keepdim=True)
        else:
            if normalized:
                fb /= fb.sum(dim=0)

        return fb

    @staticmethod
    def mul_complex(t1, t2):
        """
        Compute multiplication of two complex numbers in view_as_real format.

        Parameters:
            t1:         Float[F, 2] - First number
            t2:         Float[F, 2] - Second number
        
        Returns:
            output:     Float[F, 2] - final multiplication of two complex numbers
        """
        t1_real = t1[..., 0] 
        t1_imag = t1[..., 1]
        t2_real = t2[..., 0]
        t2_imag = t2[..., 1]
        return torch.stack((t1_real * t2_real - t1_imag * t2_imag, t1_real * t2_imag + t1_imag * t2_real), dim=-1)
    
    def erb(self, input_data: Tensor, erb_eps: float = 1e-10) -> Tensor:
        """
        Original code - pyDF/src/lib.rs - erb()
        Calculating ERB features for each frame.

        Parameters:
            input_data:     Float[T, F] or Float[F] - audio spectrogram 

        Returns:
            erb_features:   Float[T, ERB] or Float[ERB] - erb features for given spectrogram
        """

        magnitude_squared = torch.sum(input_data ** 2, dim=-1)
        erb_features = magnitude_squared.matmul(self.forward_erb_matrix)
        erb_features_db = 10.0 * torch.log10(erb_features + erb_eps)

        return erb_features_db
    
    @staticmethod
    def band_mean_norm_erb(xs: Tensor, erb_norm_state: Tensor, alpha: float, denominator: float = 40.0) -> Tuple[Tensor, Tensor]:
        """
        Original code - libDF/src/lib.rs - band_mean_norm()
        Normalizing ERB features. And updates the normalization state.

        Parameters:
            xs:             Float[ERB] - erb features
            erb_norm_state: Float[ERB] - normalization state from previous step
            alpha:          float - alpha value which is needed for adaptation of the normalization state for given scale.
            denominator:    float - denominator for normalization

        Returns:
            output:         Float[ERB] - normalized erb features
            erb_norm_state: Float[ERB] - updated normalization state
        """
        new_erb_norm_state = torch.lerp(xs, erb_norm_state, alpha)
        output = (xs - new_erb_norm_state) / denominator
        
        return output, new_erb_norm_state

    @staticmethod    
    def band_unit_norm(xs: Tensor, band_unit_norm_state, alpha: float) -> Tuple[Tensor, Tensor]:
        """
        Original code - libDF/src/lib.rs - band_unit_norm()
        Normalizing Deep Filtering features. And updates the normalization state.

        Parameters:
            xs:                     Float[1, DF, 2] - deep filtering features
            band_unit_norm_state:   Float[1, DF, 1] - normalization state from previous step
            alpha:                  float - alpha value which is needed for adaptation of the normalization state for given scale.

        Returns:
            output:                 Float[1, DF] - normalized deep filtering features
            band_unit_norm_state:   Float[1, DF, 1] - updated normalization state
        """
        xs_abs = torch.linalg.norm(xs, dim=-1, keepdim=True) # xs.abs() from complex
        new_band_unit_norm_state = torch.lerp(xs_abs, band_unit_norm_state, alpha)
        output = xs / new_band_unit_norm_state.sqrt()
        
        return output, new_band_unit_norm_state

    def frame_analysis(self, input_frame: Tensor, analysis_mem: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Original code - libDF/src/lib.rs - frame_analysis()
        Calculating spectrograme for one frame. Every frame is concated with buffer from previous frame.

        Parameters:
            input_frame:    Float[f] - Input raw audio frame
            analysis_mem:   Float[f] - Previous frame
        
        Returns:
            output:         Float[F, 2] - Spectrogram
            analysis_mem:   Float[f] - Saving current frame for next iteration
        """
        # First part of the window on the previous frame
        # Second part of the window on the new input frame
        buf = torch.cat([analysis_mem, input_frame]) * self.window
        rfft_buf = torch.matmul(buf, self.rfft_matrix) * self.wnorm

        # Copy input to analysis_mem for next iteration        
        return rfft_buf, input_frame
    
    def frame_synthesis(self, x: Tensor, synthesis_mem: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Original code - libDF/src/lib.rs - frame_synthesis()
        Inverse rfft for one frame. Every frame is summarized with buffer from previous frame.
        And saving buffer for next frame.

        Parameters:
            x:     Float[F, 2] - Enhanced audio spectrogram
            synthesis_mem:  Float[f] - Previous synthesis frame

        Returns:
            output:         Float[f] - Enhanced audio
            synthesis_mem:  Float[f] - Saving current frame
        """
        # x - [F=481, 2]
        # self.irfft_matrix - [fft_size=481, 2, f=960]
        # [f=960]
        x = torch.einsum('fi,fij->j', x, self.irfft_matrix) * self.fft_size * self.window

        x_first, x_second = torch.split(x, [self.frame_size, self.window_size - self.frame_size])
        output = x_first + synthesis_mem 

        return output, x_second

    def is_apply_gains(self, lsnr: Tensor) -> Tensor:
        """
        Original code - libDF/src/tract.rs - is_apply_stages()
        This code decomposed for better graph capturing

        Parameters:
            lsnr:   Tensor[Float] - predicted lsnr value

        Returns:
            output: Tensor[Bool] - whether to apply gains or not
        """
        if self.always_apply_all_stages:
            return torch.ones_like(lsnr, dtype=torch.bool)
        
        return torch.le(lsnr, self.max_db_erb_thresh) * torch.ge(lsnr, self.min_db_thresh)
    
    def is_apply_gain_zeros(self, lsnr: Tensor) -> Tensor:
        """
        Original code - libDF/src/tract.rs - is_apply_stages()
        This code decomposed for better graph capturing

        Parameters:
            lsnr:   Tensor[Float] - predicted lsnr value

        Returns:
            output: Tensor[Bool] - whether to apply gain_zeros or not
        """
        if self.always_apply_all_stages:
            return torch.zeros_like(lsnr, dtype=torch.bool)
        
        # Only noise detected, just apply a zero mask
        return torch.ge(self.min_db_thresh, lsnr)
        
    def is_apply_df(self, lsnr: Tensor) -> Tensor:
        """
        Original code - libDF/src/tract.rs - is_apply_stages()
        This code decomposed for better graph capturing

        Parameters:
            lsnr:   Tensor[Float] - predicted lsnr value

        Returns:
            output: Tensor[Bool] - whether to apply deep filtering or not
        """
        if self.always_apply_all_stages:
            return torch.ones_like(lsnr, dtype=torch.bool)
        
        return torch.le(lsnr, self.max_db_df_thresh) * torch.ge(lsnr, self.min_db_thresh)

    def apply_mask(self, spec: Tensor, gains: Tensor) -> Tensor:
        """
        Original code - libDF/src/lib.rs - apply_interp_band_gain()

        Applying ERB Gains for input spectrogram

        Parameters:
            spec:   Float[F, 2] - Input frame spectrogram
            gains:  Float[ERB] - ERB gains from erb decoder

        Returns:
            spec:   Float[F] - Spectrogram with applyed ERB gains
        """
        gains = gains.matmul(self.inverse_erb_matrix)
        spec = spec * gains.unsqueeze(-1)
        
        return spec
    
    def deep_filter(self, gain_spec: Tensor, coefs: Tensor, rolling_spec_buf_x: Tensor) -> Tensor:
        """
        Original code - libDF/src/tract.rs - df()

        Applying Deep Filtering to gained spectrogram by multiplying coefs to rolling_buffer_x (spectrograms from past / future).
        Deep Filtering replacing lower self.nb_df spec bands.

        Parameters:
            gain_spec:              Float[F, 2] - spectrogram after ERB gains applied
            coefs:                  Float[DF, BUF, 2] - coefficients for deep filtering from df decoder
            rolling_spec_buf_x:     Float[buffer_size, F, 2] - spectrograms from past / future
        
        Returns:
            gain_spec:              Float[F, 2] - spectrogram after deep filtering
        """
        stacked_input_specs = rolling_spec_buf_x[:, :self.nb_df]
        mult = self.mul_complex(stacked_input_specs, coefs)
        gain_spec[:self.nb_df] = torch.sum(mult, dim=0)
        return gain_spec
    
    def unpack_states(self, states: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        splitted_states = torch.split(states, self.state_lens)

        erb_norm_state = splitted_states[0].view(self.erb_norm_state_shape)
        band_unit_norm_state = splitted_states[1].view(self.band_unit_norm_state_shape)
        analysis_mem = splitted_states[2].view(self.analysis_mem_shape)
        synthesis_mem = splitted_states[3].view(self.synthesis_mem_shape)
        rolling_erb_buf = splitted_states[4].view(self.rolling_erb_buf_shape)
        rolling_feat_spec_buf = splitted_states[5].view(self.rolling_feat_spec_buf_shape)
        rolling_c0_buf = splitted_states[6].view(self.rolling_c0_buf_shape)
        rolling_spec_buf_x = splitted_states[7].view(self.rolling_spec_buf_x_shape)
        rolling_spec_buf_y = splitted_states[8].view(self.rolling_spec_buf_y_shape)
        enc_hidden = splitted_states[9].view(self.enc_hidden_shape)
        erb_dec_hidden = splitted_states[10].view(self.erb_dec_hidden_shape)
        df_dec_hidden = splitted_states[11].view(self.df_dec_hidden_shape)

        new_erb_norm_state = torch.linspace(
            self.linspace_erb[0], self.linspace_erb[1], self.nb_bands, device=erb_norm_state.device
        ).view(self.erb_norm_state_shape).to(torch.float32) # float() to fix export issue
        new_band_unit_norm_state = torch.linspace(
            self.linspace_df[0], self.linspace_df[1], self.nb_df, device=band_unit_norm_state.device
        ).view(self.band_unit_norm_state_shape).to(torch.float32) # float() to fix export issue

        erb_norm_state = torch.where(
            torch.tensor(torch.nonzero(erb_norm_state).shape[0] == 0),
            new_erb_norm_state,
            erb_norm_state
        )
    
        band_unit_norm_state = torch.where(
            torch.tensor(torch.nonzero(band_unit_norm_state).shape[0] == 0),
            new_band_unit_norm_state,
            band_unit_norm_state
        )

        return (
            erb_norm_state, band_unit_norm_state,
            analysis_mem, synthesis_mem,
            rolling_erb_buf, rolling_feat_spec_buf, rolling_c0_buf,
            rolling_spec_buf_x, rolling_spec_buf_y,
            enc_hidden, erb_dec_hidden, df_dec_hidden
        )
    
    def forward(self, 
                input_frame: Tensor, 
                states: Tensor,
                atten_lim_db: Tensor
                ) -> Tuple[
                    Tensor, Tensor, Tensor
                ]:
        """
        Enhancing input audio frame

        Parameters:
            input_frame:        Float[t] - Input raw audio frame
            states:             Float[state_len] - Flattened and concated states
            atten_lim_db:       Float[1] - Attenuation lim

        Returns:
            enhanced_frame:     Float[t] - Enhanced audio frame
            new_states:         Float[state_len] - Flattened and concated updated states
            lsnr:               Float[1] - Estimated lsnr of input frame

        """
        assert input_frame.ndim == 1, 'only bs=1 and t=frame_size supported'
        assert input_frame.shape[0] == self.frame_size, 'input_frame must be bs=1 and t=frame_size'

        (
            erb_norm_state, band_unit_norm_state,
            analysis_mem, synthesis_mem,
            rolling_erb_buf, rolling_feat_spec_buf, rolling_c0_buf,
            rolling_spec_buf_x, rolling_spec_buf_y,
            enc_hidden, erb_dec_hidden, df_dec_hidden
        ) = self.unpack_states(states)

        # If input_frame is silent, then do nothing and return zeros
        rms_non_silence_condition = (input_frame ** 2).sum() / self.frame_size >= self.silence_thresh
        rms_non_silence_condition = torch.logical_or(rms_non_silence_condition, self.always_apply_all_stages)

        spectrogram, new_analysis_mem = self.frame_analysis(input_frame, analysis_mem)
        spectrogram = spectrogram.unsqueeze(0) # [1, freq_size, 2] reshape needed for easier stacking buffers
        new_rolling_spec_buf_x = torch.cat([rolling_spec_buf_x[1:, ...], spectrogram]) # [n_frames=5, 481, 2]
        # rolling_spec_buf_y - [n_frames=7, 481, 2] n_frames=7 for compatability with original code, but in code we use only one frame
        new_rolling_spec_buf_y = torch.cat([rolling_spec_buf_y[1:, ...], spectrogram])

        erb_feat, new_erb_norm_state = self.band_mean_norm_erb(
            self.erb(spectrogram).squeeze(0), erb_norm_state, alpha=self.alpha
        ) # [ERB]
        spec_feat, new_band_unit_norm_state = self.band_unit_norm(
            spectrogram[:, :self.nb_df], band_unit_norm_state, alpha=self.alpha
        ) # [1, DF, 2]

        erb_feat = erb_feat[None, None, None, ...] # [b=1, conv_input_dim=1, t=1, n_erb=32]
        spec_feat = spec_feat[None, ...].permute(0, 3, 1, 2) # [bs=1, conv_input_dim=2, t=1, df_order=96]

        # (1, 1, T, self.nb_bands)
        new_rolling_erb_buf = torch.cat([rolling_erb_buf[:, :, 1:, :], erb_feat], dim=2)

        #  (1, 2, T, self.nb_df)
        new_rolling_feat_spec_buf = torch.cat([rolling_feat_spec_buf[:, :, 1:, :], spec_feat], dim=2)

        e0, e1, e2, e3, emb, c0, lsnr, new_enc_hidden = self.enc(
            new_rolling_erb_buf, 
            new_rolling_feat_spec_buf, 
            enc_hidden
        )
        lsnr = lsnr.flatten() # [b=1, t=1, 1] -> 1

        apply_gains = self.is_apply_gains(lsnr)
        apply_df = self.is_apply_df(lsnr)
        apply_gain_zeros = self.is_apply_gain_zeros(lsnr)

        # erb_dec
        # [BS=1, 1, T=1, ERB]
        new_gains, new_erb_dec_hidden = self.erb_dec(emb, e3, e2, e1, e0, erb_dec_hidden)
        gains = torch.where(apply_gains, new_gains.view(self.nb_bands), self.zero_gains)
        new_erb_dec_hidden = torch.where(apply_gains, new_erb_dec_hidden, erb_dec_hidden)

        # df_dec
        new_rolling_c0_buf = torch.cat([rolling_c0_buf[:, :, 1:, :], c0], dim=2)
        # new_coefs - [BS=1, T=1, F, O*2]
        new_coefs, new_df_dec_hidden = self.df_dec(
            emb, 
            new_rolling_c0_buf, 
            df_dec_hidden
        )
        new_rolling_c0_buf = torch.where(apply_df, new_rolling_c0_buf, rolling_c0_buf)
        new_df_dec_hidden = torch.where(apply_df, new_df_dec_hidden, df_dec_hidden)
        coefs = torch.where(
            apply_df, 
            new_coefs.view(self.nb_df, -1, 2).permute(1, 0, 2), 
            self.zero_coefs
        )

        # Applying features
        current_spec = new_rolling_spec_buf_y[self.df_order - 1]
        current_spec = torch.where(
            torch.logical_or(apply_gains, apply_gain_zeros),
            self.apply_mask(current_spec.clone(), gains),
            current_spec
        )
        current_spec = torch.where(
            apply_df, 
            self.deep_filter(current_spec.clone(), coefs, new_rolling_spec_buf_x),
            current_spec
        )

        # Mixing some noisy channel
        # taken from https://github.com/Rikorose/DeepFilterNet/blob/59789e135cb5ed0eb86bb50e8f1be09f60859d5c/DeepFilterNet/df/enhance.py#L237
        if torch.abs(atten_lim_db) > 0:
            spec_noisy = rolling_spec_buf_x[max(self.lookahead, self.df_order) - self.lookahead - 1]
            lim = 10 ** (-torch.abs(atten_lim_db) / self.normalize_atten_lim)
            current_spec = torch.lerp(current_spec, spec_noisy, lim)

        enhanced_audio_frame, new_synthesis_mem = self.frame_synthesis(current_spec, synthesis_mem)

        new_states = [
            new_erb_norm_state, new_band_unit_norm_state,
            new_analysis_mem, new_synthesis_mem,
            new_rolling_erb_buf, new_rolling_feat_spec_buf, new_rolling_c0_buf,
            new_rolling_spec_buf_x, new_rolling_spec_buf_y,
            new_enc_hidden, new_erb_dec_hidden, new_df_dec_hidden
        ]
        new_states = torch.cat([x.flatten() for x in new_states])

        # RMS conditioning for better ONNX graph
        enhanced_audio_frame = torch.where(rms_non_silence_condition, enhanced_audio_frame, torch.zeros_like(enhanced_audio_frame))
        new_states = torch.where(rms_non_silence_condition, new_states, states)

        return enhanced_audio_frame, new_states, lsnr

class TorchDFPipeline(nn.Module):
    def __init__(
            self, nb_bands=32, hop_size=480, fft_size=960, 
            df_order=5, conv_lookahead=2, nb_df=96, model_base_dir='DeepFilterNet3',
            atten_lim_db=0.0, always_apply_all_stages=False, device='cpu'
        ):
        super().__init__()
        self.hop_size = hop_size
        self.fft_size = fft_size

        model, state, _ = init_df(config_allow_defaults=True, model_base_dir=model_base_dir)
        model.eval()
        self.sample_rate = state.sr()

        self.torch_streaming_model = ExportableStreamingTorchDF(
            nb_bands=nb_bands, hop_size=hop_size, fft_size=fft_size, 
            enc=model.enc, df_dec=model.df_dec, erb_dec=model.erb_dec, df_order=df_order,
            always_apply_all_stages=always_apply_all_stages,
            conv_lookahead=conv_lookahead, nb_df=nb_df, sr=self.sample_rate
        )
        self.torch_streaming_model = self.torch_streaming_model.to(device)
        self.states = torch.zeros(self.torch_streaming_model.states_full_len, device=device)

        self.atten_lim_db = torch.tensor(atten_lim_db, device=device)

    def forward(self, input_audio: Tensor, sample_rate: int) -> Tensor:
        """
        Denoising audio frame using exportable fully torch model.

        Parameters:
            input_audio:      Float[1, t] - Input audio
            sample_rate:      Int - Sample rate

        Returns:
            enhanced_audio:   Float[1, t] - Enhanced input audio
        """
        assert input_audio.shape[0] == 1, f'Only mono supported! Got wrong shape! {input_audio.shape}'
        assert sample_rate == self.sample_rate, f'Only {self.sample_rate} supported! Got wrong sample rate! {sample_rate}'

        input_audio = input_audio.squeeze(0)
        orig_len = input_audio.shape[0]

        # padding taken from
        # https://github.com/Rikorose/DeepFilterNet/blob/fa926662facea33657c255fd1f3a083ddc696220/DeepFilterNet/df/enhance.py#L229
        hop_size_divisible_padding_size = (self.hop_size - orig_len % self.hop_size) % self.hop_size
        orig_len += hop_size_divisible_padding_size
        input_audio = F.pad(input_audio, (0, self.fft_size + hop_size_divisible_padding_size))
        
        chunked_audio = torch.split(input_audio, self.hop_size)

        output_frames = []

        for input_frame in chunked_audio:
            (
                enhanced_audio_frame, self.states, lsnr
            ) = self.torch_streaming_model(
                input_frame, 
                self.states,
                self.atten_lim_db
            )
            
            output_frames.append(enhanced_audio_frame)

        enhanced_audio = torch.cat(output_frames).unsqueeze(0) # [t] -> [1, t] typical mono format

        # taken from 
        # https://github.com/Rikorose/DeepFilterNet/blob/fa926662facea33657c255fd1f3a083ddc696220/DeepFilterNet/df/enhance.py#L248
        d = self.fft_size - self.hop_size
        enhanced_audio = enhanced_audio[:, d : orig_len + d]

        return enhanced_audio


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    torch_df = TorchDFPipeline(device=args.device)

    # torchaudio normalize=True, fp32 return
    noisy_audio, sr = torchaudio.load(args.audio_path, channels_first=True)
    noisy_audio = noisy_audio.mean(dim=0).unsqueeze(0).to(args.device) # stereo to mono

    enhanced_audio = torch_df(noisy_audio, sr).detach().cpu()

    torchaudio.save(
        args.output_path, enhanced_audio, sr,
        encoding="PCM_S", bits_per_sample=16
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Denoising one audio with DF3 model using torch only'
    )
    parser.add_argument(
        '--audio-path', type=str, required=True, help='Path to audio file'
    )
    parser.add_argument(
        '--output-path', type=str, required=True, help='Path to output file'
    )
    parser.add_argument(
        '--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Device to run on'
    )

    main(parser.parse_args())    
        