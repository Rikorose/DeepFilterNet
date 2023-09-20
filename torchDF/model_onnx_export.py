import os
import copy
import onnx
import argparse
import subprocess

import torch
import torchaudio
import numpy as np
import onnxruntime as ort
import torch.utils.benchmark as benchmark

from torch_df_streaming import TorchDFPipeline
from typing import Dict, Iterable

torch.manual_seed(0)

FRAME_SIZE = 480
INPUT_NAMES = [
    'input_frame', 
    'states',
    'atten_lim_db'
]
OUTPUT_NAMES = [
    'enhanced_audio_frame', 'out_states', 'lsnr'
]


def onnx_simplify(
    path: str, input_data: Dict[str, np.ndarray], input_shapes: Dict[str, Iterable[int]]
) -> str:
    """
    Simplify ONNX model using onnxsim and checking it
    
    Parameters:
        path:           str - Path to ONNX model
        input_data:     Dict[str, np.ndarray] - Input data for ONNX model
        input_shapes:   Dict[str, Iterable[int]] - Input shapes for ONNX model

    Returns:
        path:           str - Path to simplified ONNX model
    """
    import onnxsim

    model = onnx.load(path)
    model_simp, check = onnxsim.simplify(
        model,
        input_data=input_data,
        test_input_shapes=input_shapes,
    )
    assert check, "Simplified ONNX model could not be validated"
    onnx.checker.check_model(model_simp, full_check=True)
    onnx.save_model(model_simp, path)
    return path

def test_onnx_model(torch_model, ort_session, states, atten_lim_db):
    """
    Simple test that everything converted correctly

    Parameters:
        torch_model:    torch.nn.Module - Original torch model
        ort_session:    onnxruntime.InferenceSession - Inference Session for converted ONNX model
        input_features: Dict[str, np.ndarray] - Input features
    """
    states_torch = copy.deepcopy(states)
    states_onnx = copy.deepcopy(states)

    for i in range(30):
        input_frame = torch.randn(FRAME_SIZE)

        # torch
        output_torch = torch_model(input_frame, states_torch, atten_lim_db)

        # onnx
        output_onnx = ort_session.run(
            OUTPUT_NAMES,
            generate_onnx_features([input_frame, states_onnx, atten_lim_db]),
        )

        for (x, y, name) in zip(output_torch, output_onnx, OUTPUT_NAMES):
            y_tensor = torch.from_numpy(y)
            assert torch.allclose(x, y_tensor, atol=1e-3), f"out {name} - {i}, {x.flatten()[-5:]}, {y_tensor.flatten()[-5:]}"

def generate_onnx_features(input_features):
    return {
        x: y.detach().cpu().numpy() 
        for x, y in zip(INPUT_NAMES, input_features)
    }

def perform_benchmark(
    ort_session, input_features: Dict[str, np.ndarray],
):
    """
    Benchmark ONNX model performance

    Parameters:
        ort_session:    onnxruntime.InferenceSession - Inference Session for converted ONNX model
        input_features: Dict[str, np.ndarray] - Input features
    """
    def run_onnx():
        output = ort_session.run(
            OUTPUT_NAMES,
            input_features,
        )
    
    t0 = benchmark.Timer(
        stmt='run_onnx()',
        num_threads=1,
        globals={'run_onnx': run_onnx},
    )
    print(f"Median iteration time: {t0.blocked_autorange(min_run_time=10).median * 1e3:6.2f} ms / {480 / 48000 * 1000} ms")

def infer_onnx_model(streaming_pipeline, ort_session, inference_path):
    """
    Inference ONNX model with TorchDFPipeline
    """
    del streaming_pipeline.torch_streaming_model
    streaming_pipeline.torch_streaming_model = lambda *features: (
        torch.from_numpy(x) for x in ort_session.run(
            OUTPUT_NAMES,
            generate_onnx_features(list(features)),
        )
    )

    noisy_audio, sr = torchaudio.load(inference_path, channels_first=True)
    noisy_audio = noisy_audio.mean(dim=0).unsqueeze(0) # stereo to mono

    enhanced_audio = streaming_pipeline(noisy_audio, sr)

    torchaudio.save(
        inference_path.replace('.wav', '_onnx_infer.wav'), enhanced_audio, sr,
        encoding="PCM_S", bits_per_sample=16
    )    

def main(args):
    streaming_pipeline = TorchDFPipeline(always_apply_all_stages=args.always_apply_all_stages, device='cpu')
    torch_df = streaming_pipeline.torch_streaming_model
    states = streaming_pipeline.states
    atten_lim_db = streaming_pipeline.atten_lim_db

    input_frame = torch.rand(FRAME_SIZE)
    input_features = (
        input_frame, states, atten_lim_db
    )
    torch_df(*input_features) # check model

    torch_df_script = torch.jit.script(torch_df)
    torch.onnx.export(
        torch_df_script,
        input_features,
        args.output_path,
        verbose=False,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=14
    )
    print(f'Model exported to {args.output_path}!')

    input_features_onnx = generate_onnx_features(input_features)
    input_shapes_dict = {
        x: y.shape
        for x, y in input_features_onnx.items()
    }

    # Simplify not working!
    if args.simplify:
        raise NotImplementedError("Simplify not working for flatten states!")
        onnx_simplify(args.output_path, input_features_onnx, input_shapes_dict)
        print(f'Model simplified! {args.output_path}')

    if args.ort:
        if subprocess.run([
            'python', '-m', 'onnxruntime.tools.convert_onnx_models_to_ort',
            args.output_path,
            '--optimization_style', 'Fixed',
        ]).returncode != 0:
            raise RuntimeError("ONNX to ORT conversion failed!")
        print('Model converted to ORT format!')

    print('Checking model...')
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = args.output_path
    sess_options.intra_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    ort_session = ort.InferenceSession(args.output_path, sess_options, providers=['CPUExecutionProvider'])

    onnx_outputs = ort_session.run(
        OUTPUT_NAMES,
        input_features_onnx,
    )

    print(f'InferenceSession successful! Output shapes: {[x.shape for x in onnx_outputs]}')

    if args.test:
        test_onnx_model(torch_df, ort_session, input_features[1], input_features[2])
        print('Tests passed!')

    if args.performance:
        print('Performanse check...')
        perform_benchmark(ort_session, input_features_onnx)

    if args.inference_path:
        infer_onnx_model(streaming_pipeline, ort_session, args.inference_path)
        print(f'Audio from {args.inference_path} enhanced!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Exporting torchDF model to ONNX'
    )
    parser.add_argument(
        '--output-path', type=str, default='denoiser_model.onnx', help='Path to output onnx file'
    )
    parser.add_argument(
        '--simplify', action='store_true', help='Simplify the model'
    )
    parser.add_argument(
        '--test', action='store_true', help='Test the onnx model'
    )
    parser.add_argument(
        '--performance', action='store_true', help='Mesure median iteration time for onnx model'
    )
    parser.add_argument(
        '--inference-path', type=str, help='Run inference on example'
    )
    parser.add_argument(
        '--ort', action='store_true', help='Save to ort format'
    )
    parser.add_argument(
        '--always-apply-all-stages', action='store_true', help='Always apply stages'
    )
    main(parser.parse_args())
