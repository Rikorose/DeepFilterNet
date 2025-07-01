from onnxruntime.quantization import quant_pre_process, quantize_dynamic, QuantType
import onnx


input_model_path = "enc.onnx"
preprocessed_model_path = "model_preprocessed.onnx"
quantized_model_path = "enc_quant.onnx"

model = onnx.load(input_model_path)
quant_pre_process(
    input_model_path=input_model_path,
    output_model_path=preprocessed_model_path
)
quantize_dynamic(
    preprocessed_model_path,
    quantized_model_path,
    weight_type=QuantType.QInt8
)