import os
from copy import deepcopy
from timeit import timeit

import torch
from torch import nn
from torch.ao.quantization import (
    MinMaxObserver,
    QConfig,
    QConfigDynamic,
    QConfigMapping,
    float16_dynamic_qconfig,
    qconfig_mapping,
    quantize_dynamic,
)
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.fx import GraphModule, symbolic_trace

from df.config import config
from df.enhance import df_features, init_df, setup_df_argument_parser
from df.io import get_test_sample, load_audio, save_audio
from df.model import ModelParams
from df.utils import as_complex, get_device
from libdfdata import PytorchDataLoader as DataLoader


@torch.no_grad()
def main(args):
    model, df_state, _ = init_df(
        args.model_base_dir,
        post_filter=args.pf,
        log_level=args.log_level,
        log_file="export.log",
        config_allow_defaults=True,
        epoch=args.epoch,
    )
    model.eval()
    p = ModelParams()
    # for name, m in model.named_parameters():
    #     if hasattr(m, "data"):
    #         print(name, m.data.shape, m.data.min(), m.data.max())
    # model_traced: GraphModule = symbolic_trace(model)
    enc_traced: GraphModule = symbolic_trace(model.enc)
    erb_dec_traced: GraphModule = symbolic_trace(model.erb_dec)
    df_dec_traced: GraphModule = symbolic_trace(model.df_dec)

    model_float = deepcopy(model)

    qconfig = QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.qint8),
        weight=MinMaxObserver.with_args(dtype=torch.qint8),
    )
    qconfig_dyn = QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.qint8),
        weight=MinMaxObserver.with_args(dtype=torch.qint8),
    )
    # qconfig_dyn=float16_dynamic_qconfig
    qconfig_dict_dyn = {nn.GRU: qconfig_dyn}
    qconfig_dict = (
        QConfigMapping()
        .set_object_type(nn.Conv2d, qconfig)
        .set_object_type(nn.ConvTranspose2d, qconfig)
        .set_object_type(nn.ReLU, qconfig)
        .set_object_type(nn.BatchNorm2d, qconfig)
        .set_object_type(nn.Linear, qconfig)
    )

    audio = get_test_sample(p.sr)
    spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device=get_device())
    enhanced = as_complex(model(spec, feat_erb, feat_spec)[0].cpu().squeeze(1))
    enhanced = torch.as_tensor(df_state.synthesis(enhanced.numpy()))
    save_audio("out/enhanced_float.wav", enhanced, p.sr)
    feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2)
    e0, e1, e2, e3, emb, c0, lsnr = model.enc(feat_erb, feat_spec)
    _ = model.erb_dec(emb, e3, e2, e1, e0)
    _ = model.df_dec(emb, c0)

    enc = quantize_dynamic(model.enc, qconfig_dict_dyn)
    erb_dec = quantize_dynamic(model.erb_dec, qconfig_dict_dyn)
    df_dec = quantize_dynamic(model.df_dec, qconfig_dict_dyn)
    enc = prepare_fx(enc, qconfig_dict, example_inputs=(feat_erb, feat_spec))
    erb_dec = prepare_fx(erb_dec, qconfig_dict, example_inputs=(emb, e3, e2, e1, e0))
    df_dec = prepare_fx(df_dec, qconfig_dict, example_inputs=(emb, c0))

    sample = get_test_sample(df_state.sr())
    for i, sample in enumerate(args.noisy_files):
        if i > 10:
            break
        print(sample)
        audio, _ = load_audio(sample, p.sr)
        spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device=get_device())
        feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2)
        with torch.no_grad():
            e0, e1, e2, e3, emb, c0, _ = enc(feat_erb, feat_spec)
            _ = erb_dec(emb, e3, e2, e1, e0)
            _ = df_dec(emb, c0)

    enc = convert_fx(enc)
    erb_dec = convert_fx(erb_dec)
    df_dec = convert_fx(df_dec)
    torch.save(enc, os.path.join(args.export_dir, "enc_quant.pt"))
    torch.save(erb_dec, os.path.join(args.export_dir, "erb_dec_quant.pt"))
    torch.save(df_dec, os.path.join(args.export_dir, "df_dec_quant.pt"))

    # This failes:
    # torch.onnx.export(
    #     model=enc,
    #     f=os.path.join(args.export_dir, "enc_q.onnx"),
    #     args=(feat_erb, feat_spec),
    #     input_names=("feat_erb", "feat_spec"),
    # )

    model.enc = enc
    model.erb_dec = erb_dec
    model.df_dec = df_dec

    audio = get_test_sample(p.sr)
    spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device=get_device())
    enhanced = as_complex(model(spec, feat_erb, feat_spec)[0].cpu().squeeze(1))
    enhanced = torch.as_tensor(df_state.synthesis(enhanced.numpy()))
    save_audio("out/enhanced_quant.wav", enhanced, p.sr)

    print(timeit(lambda: model_float(spec, feat_erb, feat_spec), number=10))
    print(timeit(lambda: model(spec, feat_erb, feat_spec), number=10))


if __name__ == "__main__":
    parser = setup_df_argument_parser()
    parser.add_argument("export_dir")
    parser.add_argument(
        "noisy_files",
        type=str,
        nargs="+",
        help="Path to the directory containing noisy audio files for calibration.",
    )
    args = parser.parse_args()
    main(args)
