from torch.nn import functional as F
from torch import nn
from onnxruntime.quantization import quantize_static, quant_pre_process, CalibrationDataReader, QuantType, QuantFormat
import onnx
import onnxruntime as ort
from df.io import load_audio
from df.enhance import df_features, init_df
from df.model import ModelParams
import warnings
warnings.filterwarnings("ignore")


def calibration_data_generator(model_type,input_model_path):
    for file_path in samples:
      audio, meta = load_audio(file_path, sr=df_state.sr())
      n_fft, hop = df_state.fft_size(), df_state.hop_size()
      audio = F.pad(audio, (0, n_fft))
      spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device="cpu")
      pad_feat = nn.ConstantPad2d((0, 0, -p.conv_lookahead, p.conv_lookahead), 0.0)
      feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2)
      feat_erb = pad_feat(feat_erb)
      ort_sess = ort.InferenceSession(input_model_path)
      e0,e1,e2,e3,emb,c0,lsnr = ort_sess.run(None, {'feat_erb':feat_erb.numpy(), 'feat_spec': feat_spec.numpy()})
      if model_type == 'enc':
          yield {'feat_erb': feat_erb.numpy(), 'feat_spec': feat_spec.numpy()}
      if model_type == 'erb_dec':
        yield {'emb':emb, 'e3':e3, 'e2':e2, 'e1':e1, 'e0':e0}
      if model_type == 'df_dec':
        yield {'emb':emb, 'c0':c0}

class DataReader(CalibrationDataReader):
    def __init__(self, data_generator, model_type, input_model_path):
        self.data_generator = data_generator
        self.iterator = iter(self.data_generator(model_type, input_model_path))

    def get_next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            return None

def quant_layer(model_type, input_model_path, quantized_model_path):
    preprocessed_model_path = "model_preprocessed.onnx"
    '''
    You could apply only to some layers
    model = onnx.load(input_model_path)
    nodes_to_exclude = []
    for node in model.graph.node:
        if node.op_type != "Conv":
            nodes_to_exclude.append(node.name)
    '''

    quant_pre_process(
        input_model_path=input_model_path,
        output_model_path=preprocessed_model_path
    )
    calibration_reader = DataReader(calibration_data_generator, model_type, input_model_path)
    quantize_static(
        preprocessed_model_path,
        quantized_model_path,
        calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8
        #nodes_to_exclude=nodes_to_exclude
    )


if __name__ == "__main__":
    import glob
    import random
    path_to_dataset = 'drive/MyDrive/BWAVN'
    files = glob.glob(f'{path_to_dataset}/*')
    input_model_path = "enc.onnx"
    quantized_model_path = "enc_quant_static.onnx"
    #model type is enc, erb_dec or df_dec
    model_type = 'enc'
    n_samples = 101
    random.seed(10)
    samples = random.sample(files, n_samples)
    layers = ['enc','erb_dec','df_dec']
    model, df_state, _ = init_df()
    p = ModelParams()
    quant_layer(model_type, input_model_path, quantized_model_path)