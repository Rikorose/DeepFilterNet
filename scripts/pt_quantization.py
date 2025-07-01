import os
from copy import deepcopy

import torch
from torch.ao.quantization import (
    MinMaxObserver,
    get_default_qconfig_mapping,
    QConfig,
    QConfigMapping,
    quantize_dynamic
)
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from df.enhance import df_features, init_df
from df.io import load_audio
from df.model import ModelParams
from df.utils import get_device


@torch.no_grad()
def main(model, df_state, p, samples):
    model.eval()
    qconfig = QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.qint8),
        weight=MinMaxObserver.with_args(dtype=torch.qint8),
    )
    '''
    with default qmapping cannot export in onnx, to unsupported layer
    with this type of configurations no quantization is applied, just fusing layers
    #qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    '''
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    audio, _ = load_audio(samples[0], sr=df_state.sr())
    spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device=get_device())
    feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2)
    e0, e1, e2, e3, emb, c0, lsnr = model.enc(feat_erb, feat_spec)
    _ = model.erb_dec(emb, e3, e2, e1, e0)
    _ = model.df_dec(emb, c0)
    #In case of Dynamic Quant no need of calibration
    '''
    enc = quantize_dynamic(model.enc, qconfig_dict_dyn)
    erb_dec = quantize_dynamic(model.erb_dec, qconfig_dict_dyn)
    df_dec = quantize_dynamic(model.df_dec, qconfig_dict_dyn)
    '''
    enc = model.enc
    erb_dec = model.erb_dec
    df_dec = model.df_dec
    enc_prep = prepare_fx(enc, qconfig_mapping, example_inputs=(feat_erb, feat_spec))
    erb_dec_prep = prepare_fx(erb_dec, qconfig_mapping, example_inputs=(emb, e3, e2, e1, e0))
    df_dec_prep = prepare_fx(df_dec, qconfig_mapping, example_inputs=(emb, c0))
    for file_path in samples[1:]:
        audio, _ = load_audio(file_path, p.sr)
        spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device=get_device())
        feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2)
        with torch.no_grad():
            e0, e1, e2, e3, emb, c0, _ = enc_prep(feat_erb, feat_spec)
            _ = erb_dec_prep(emb, e3, e2, e1, e0)
            _ = df_dec_prep(emb, c0)
    enc_static = convert_fx(enc_prep)
    erb_dec_static = convert_fx(erb_dec_prep)
    df_dec_static = convert_fx(df_dec_prep)
    torch.save(enc_static, os.path.join("enc_quant.pt"))
    torch.save(erb_dec_static, os.path.join("erb_dec_quant.pt"))
    torch.save(df_dec_static, os.path.join("df_dec_quant.pt"))

    dynamic_axes = {
        "feat_erb": {2: "S"},
        "feat_spec": {2: "S"},
        "e0": {2: "S"},
        "e1": {2: "S"},
        "e2": {2: "S"},
        "e3": {2: "S"},
        "emb": {1: "S"},
        "c0": {2: "S"},
        "lsnr": {1: "S"},
    }

    torch.onnx.export(
        model=deepcopy(enc_static),
        f='enc_q.onnx',
        args=(feat_erb, feat_spec),
        input_names= ["feat_erb", "feat_spec"],
        dynamic_axes=dynamic_axes,
        output_names = ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"],
        opset_version=14,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        export_params=True
    )

    inputs = (emb.clone(), e3, e2, e1, e0)
    input_names = ["emb", "e3", "e2", "e1", "e0"]
    output_names = ["m"]
    dynamic_axes = {
        "emb": {1: "S"},
        "e3": {2: "S"},
        "e2": {2: "S"},
        "e1": {2: "S"},
        "e0": {2: "S"},
        "m": {2: "S"},
    }
    torch.onnx.export(
        model=deepcopy(erb_dec_static),
        f='erb_dec_q.onnx',
        args=inputs,
        input_names= input_names,
        dynamic_axes=dynamic_axes,
        output_names = output_names,
        opset_version=14,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        export_params=True
    )
    inputs = (emb.clone(), c0)
    input_names = ["emb", "c0"]
    output_names = ["coefs"]
    dynamic_axes = {
        "emb": {1: "S"},
        "c0": {2: "S"},
        "coefs": {1: "S"},
    }
    torch.onnx.export(
      model=deepcopy(df_dec_static),
      f='df_dec_q.onnx',
      args=inputs,
      input_names= input_names,
      dynamic_axes=dynamic_axes,
      output_names = output_names,
      opset_version=14,
      keep_initializers_as_inputs=False,
      do_constant_folding=True,
      export_params=True
    )


if __name__ == "__main__":
    import glob
    import random

    path_to_dataset = 'drive/MyDrive/BWAVN'
    files = glob.glob(f'{path_to_dataset}/*')
    n_samples = 101
    random.seed(10)
    samples = random.sample(files, n_samples)
    model, df_state, _ = init_df()
    p = ModelParams()
    main(model, df_state, p, samples)