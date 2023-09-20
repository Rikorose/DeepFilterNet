import copy
import torch
import torchaudio

from df import init_df, enhance
from torch_df_offline import TorchDF
from libdf import DFTractPy
from torch_df_streaming import TorchDFPipeline


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
DEVICE = torch.device("cpu")


class TestTorchStreaming():
    def __reset(self):
        torch.manual_seed(23)
        
        self.model, self.rust_state, _ = init_df(config_allow_defaults=True, model_base_dir='DeepFilterNet3')
        self.model.eval()

        self.torch_streaming_like_offline = TorchDFPipeline(always_apply_all_stages=True, device=DEVICE)

        pipeline_for_streaming = TorchDFPipeline(always_apply_all_stages=False, device=DEVICE)
        self.torch_streaming = pipeline_for_streaming.torch_streaming_model
        self.streaming_state = pipeline_for_streaming.states
        self.atten_lim_db = pipeline_for_streaming.atten_lim_db

        self.torch_offline = TorchDF(copy.deepcopy(self.model))
        self.torch_offline = self.torch_offline.to(DEVICE)

        self.df_tract = DFTractPy()

        self.noisy_audio, self.audio_sr = torchaudio.load('examples/A1CIM28ZUCA8RX_M_Street_Near_Regular_SP_Mobile_Primary.wav', channels_first=True)
        self.noisy_audio = self.noisy_audio.mean(dim=0).unsqueeze(0).to(DEVICE)

    def test_offline_with_enhance(self):
        """
        Compare torchDF offline implementation to enhance method
        """
        self.__reset()
        enhanced_audio_torch = self.torch_streaming_like_offline(self.noisy_audio, self.audio_sr).to(DEVICE)
        enhanced_audio_offline = enhance(self.model, self.rust_state, self.noisy_audio.cpu()).to(DEVICE)

        assert torch.allclose(enhanced_audio_torch, enhanced_audio_offline, atol=1e-3)

    def test_offline_with_streaming(self):
        """
        Compare torchDF streaming implementation to torchDF offline implementation
        """
        self.__reset()

        enhanced_audio_torch = self.torch_streaming_like_offline(self.noisy_audio, self.audio_sr).to(DEVICE)
        enhanced_audio_offline = self.torch_offline(self.noisy_audio.squeeze(0)).to(DEVICE)

        assert torch.allclose(enhanced_audio_torch, enhanced_audio_offline, atol=1e-3)

    def test_streaming_torch_with_tract(self):
        """
        Compare torchDF streaming implementation to tract streaming implementation
        
        always_apply_all_stages = False
        """
        self.__reset()

        chunked_audio = torch.split(self.noisy_audio.squeeze(0), 480)

        for i, chunk in enumerate(chunked_audio):
            onnx_output, self.streaming_state, _ = self.torch_streaming(chunk, self.streaming_state, self.atten_lim_db)
            rust_output = torch.from_numpy(self.df_tract.process(chunk.unsqueeze(0).cpu().numpy()))

            assert torch.allclose(onnx_output.to(DEVICE), rust_output.to(DEVICE), atol=1e-3), f'process failed - {i} iteration'
