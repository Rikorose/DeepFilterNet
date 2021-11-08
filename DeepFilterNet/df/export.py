import argparse
import os

import onnx
import onnxruntime
import torch
from icecream import ic

from libdf import DF
from df import config
from df.dfop import DfDelaySpec, DfOpInitSpecBuf, DfOpTimeStep
from df.enhance import df_features
from df.logger import init_logger
from df.model import ModelParams
from df.modules import Mask, erb_fb
from df.train import load_model

icolnames = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds")


@torch.no_grad()
def export(
    net: torch.nn.Module, state: DF, output_dir: str, opset_version=12, constant_folding=True
):
    net.eval()
    p = ModelParams()

    audio = torch.randn((1, int(0.1 * p.sr)))
    spec, feat_erb, feat_spec = df_features(audio, state)
    ic(spec.shape)
    # only use `df_order` steps
    # o = ModelParams().df_order
    # spec = spec[:, :, :o]
    # feat_erb = feat_erb[:, :, :o]
    # feat_spec = feat_spec[:, :, :o]
    # ic(spec.shape, feat_erb.shape, feat_spec.shape)

    # run once for testing
    out, m, lsnr, df_alpha = net(spec, feat_erb, feat_spec)

    print("exporting encoder")
    feat_spec = feat_spec.transpose(1, 4).squeeze(4)  # re/im into channel axis
    ic(feat_spec.shape)
    e0, e1, e2, e3, emb, c0, lsnr = net.enc(feat_erb, feat_spec)
    enc = net.enc
    enc = torch.jit.script(enc)
    enc_path = os.path.join(output_dir, "enc.onnx")
    torch.onnx.export(
        model=enc,
        args=(feat_erb, feat_spec),
        input_names=["feat_erb", "feat_spec"],
        dynamic_axes={
            "feat_erb": {2: "time"},
            "e0": {2: "time"},
            "e1": {2: "time"},
            "e3": {2: "time"},
            "e4": {2: "time"},
            "emb": {1: "time"},
            "c0": {2: "time"},
            "lsnr": {1: "time"},
        },
        f=enc_path,
        example_outputs=(e0, e1, e2, e3, emb, c0, lsnr),
        do_constant_folding=constant_folding,
        opset_version=opset_version,
        output_names=["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"],
        verbose=False,
    )
    enc_onnx = onnx.load(enc_path)
    onnx.checker.check_model(enc_onnx)
    sess = onnxruntime.InferenceSession(enc_path)
    out = sess.run(
        ("e0", "e1", "e2", "e3", "emb", "c0", "lsnr"),
        {"feat_erb": feat_erb.numpy(), "feat_spec": feat_spec.numpy()},
    )
    torch.testing.assert_allclose(e0, out[0])
    torch.testing.assert_allclose(e1, out[1])
    torch.testing.assert_allclose(e2, out[2])
    torch.testing.assert_allclose(e3, out[3])
    torch.testing.assert_allclose(emb, out[4])
    torch.testing.assert_allclose(c0, out[5])
    torch.testing.assert_allclose(lsnr, out[6])

    print("exporting decoder")
    args = (emb.clone(), e3, e2, e1, e0)
    ic([arg.shape for arg in args])
    dec = torch.jit.script(net.erb_dec)
    # dec = net.erb_dec
    m = dec(*args)
    torch.onnx.export(
        model=dec,
        args=args,
        input_names=["emb", "e3", "e2", "e1", "e0"],
        output_names=["m"],
        training=torch.onnx.TrainingMode.EVAL,
        f=os.path.join(output_dir, "dec.onnx"),
        example_outputs=m,
        do_constant_folding=constant_folding,
        opset_version=opset_version,
        dynamic_axes={
            "emb": {1: "time"},
            "e3": {2: "time"},
            "e2": {2: "time"},
            "e1": {2: "time"},
            "e0": {2: "time"},
            "m": {2: "time"},
        },
        verbose=False,
    )
    sess = onnxruntime.InferenceSession(os.path.join(output_dir, "dec.onnx"))
    out = sess.run(
        ("m",),
        {
            "emb": emb.numpy(),
            "e3": e3.numpy(),
            "e2": e2.numpy(),
            "e1": e1.numpy(),
            "e0": e0.numpy(),
        },
    )
    torch.testing.assert_allclose(m, out[0])

    print("exporting mask op")
    erb_inverse = erb_fb(state.erb_widths(), p.sr, inverse=True)
    mask_op = Mask(erb_inverse, post_filter=p.mask_pf)
    # mask_op = torch.jit.script(mask_op)
    args = (spec, m)
    spec_m = mask_op(*args)
    ic(spec.shape, m.shape, spec_m.shape)
    torch.onnx.export(
        model=mask_op,
        args=args,
        input_names=["spec", "m"],
        f=os.path.join(output_dir, "mask_op.onnx"),
        do_constant_folding=constant_folding,
        opset_version=opset_version,
        output_names=["spec_m"],
        dynamic_axes={
            "spec": {2: "time"},
            "m": {2: "time"},
            "spec_m": {2: "time"},
        },
        verbose=False,
    )

    print("exporting dfnet")
    args = (emb, c0)
    coefs, alpha = net.df_dec(*args)
    ic(emb.shape, e0.shape, coefs.shape, alpha.shape)
    torch.onnx.export(
        model=net.df_dec,
        args=args,
        input_names=["emb", "c0"],
        f=os.path.join(output_dir, "dfnet.onnx"),
        do_constant_folding=constant_folding,
        opset_version=opset_version,
        output_names=["coefs", "alpha"],
        dynamic_axes={
            "emb": {1: "time"},
            "c0": {2: "time"},
            "coefs": {2: "time"},
            "alpha": {1: "time"},
        },
        verbose=False,
    )
    sess = onnxruntime.InferenceSession(os.path.join(output_dir, "dfnet.onnx"))
    out = sess.run(
        ("coefs", "alpha"),
        {"emb": emb.numpy(), "c0": c0.numpy()},
    )
    torch.testing.assert_allclose(coefs, out[0])
    torch.testing.assert_allclose(alpha, out[1])

    print("exporting df-op")
    args = (spec, coefs, alpha)
    spec_f_net = net.df_op(*args)
    # This prodices a range over time which is not supported by tract:
    # dfop = DfOpTimeLoop(p.df_order, p.df_lookahead, p.nb_df, p.fft_size // 2 + 1)
    # dfop = torch.jit.script(dfop)
    # torch.onnx.export(
    #     model=dfop,
    #     args=args,
    #     input_names=["spec", "coefs", "alpha"],
    #     f=os.path.join(output_dir, "dfop.onnx"),
    #     do_constant_folding=constant_folding,
    #     opset_version=opset_version,
    #     output_names=["spec_f"],
    #     dynamic_axes={
    #         "spec": {2: "time"},
    #         "coefs": {2: "time"},
    #         "alpha": {1: "time"},
    #         "spec_f": {2: "time"},
    #     },
    #     verbose=False,
    # )

    # dfop_delayspec = DfDelaySpec(lookahead=p.df_lookahead)
    # This needs to be set in for the torch.assert_allclose condition later.
    # However in tract, we cannot front-pad the spectrogram for the delay introduced in the
    # convolutions. In general, we are not able to frontpad since we imediatly get new gains, coefs
    # etc. Thus we need to delay the overall introduced delay consider this delay.
    delay = max(p.conv_lookahead, p.df_lookahead)
    delay = p.df_lookahead
    ic(delay)
    dfop_delayspec = DfDelaySpec(delay=delay)
    spec_d = dfop_delayspec(spec[0, 0])
    torch.onnx.export(
        model=dfop_delayspec,
        args=spec[0, 0],
        input_names=["spec"],
        f=os.path.join(output_dir, "dfop_delayspec.onnx"),
        do_constant_folding=constant_folding,
        opset_version=opset_version,
        output_names=["spec_d"],
        dynamic_axes={
            "spec": {2: "time"},
            "spec_d": {2: "time"},
        },
        verbose=False,
    )
    dfop_initbuf = DfOpInitSpecBuf(p.df_order, p.df_lookahead, p.nb_df, p.fft_size // 2 + 1)
    spec_buf = dfop_initbuf(spec[0, 0])
    # Let's skip initialization since it only affects a few time steps
    # torch.onnx.export(
    #     model=dfop_initbuf,
    #     args=spec,
    #     input_names=["spec"],
    #     f=os.path.join(output_dir, "dfop_initbuf.onnx"),
    #     do_constant_folding=constant_folding,
    #     opset_version=opset_version,
    #     output_names=["spec_d"],
    #     dynamic_axes={"spec": {2: "time"}, "spec_d": {2: "time"}},
    #     verbose=False,
    # )
    # print("ckecking dfop init onnx")
    # dfop_onnx = onnx.load(os.path.join(output_dir, "dfop_initbuf.onnx"))
    # ic(dfop_onnx)
    dfop_step = DfOpTimeStep(p.df_order, p.df_lookahead, p.nb_df, p.fft_size // 2 + 1)
    dfop_step = torch.jit.script(dfop_step)
    spec_f = torch.zeros_like(spec)
    ic(spec_d.shape, spec.shape)
    for t in range(spec.shape[2]):
        args = [spec_d[t + p.df_lookahead], coefs[0, t], alpha[0, t], spec_buf]
        spec_f[:, :, t], spec_buf = dfop_step(*args)
    # This will only hold if the delay op is changed as explained above.
    ic(torch.allclose(spec_f, spec_f_net))
    for arg in args:
        ic(arg.shape)
    t = 1
    torch.onnx.export(
        model=dfop_step,
        args=(spec_d[t + p.df_lookahead], coefs[0, t], alpha[0, t], spec_buf),
        input_names=["spec", "coefs", "alpha", "spec_buf_in"],
        f=os.path.join(output_dir, "dfop_step.onnx"),
        do_constant_folding=constant_folding,
        opset_version=opset_version,
        output_names=["spec_f", "spec_buf"],
        verbose=False,
    )
    print("ckecking dfop onnx")
    dfop_onnx = onnx.load(os.path.join(output_dir, "dfop_step.onnx"))
    print(dfop_onnx, file=open("out/dfopstep.json", "w"))
    onnx.checker.check_model(dfop_onnx)
    print("running dfop onnx")
    sess = onnxruntime.InferenceSession(os.path.join(output_dir, "dfop_step.onnx"))
    spec_d = dfop_delayspec(spec[0, 0])
    spec_buf = dfop_initbuf(spec[0, 0]).numpy()
    for t in range(spec.shape[2] - p.df_lookahead):
        out = sess.run(
            ("spec_f", "spec_buf"),
            {
                "spec": spec_d[t + p.df_lookahead].numpy(),
                "coefs": coefs[0, t].numpy(),
                "alpha": alpha[0, t].numpy(),
                "spec_buf_in": spec_buf,
            },
        )
        spec_buf = out[1]
        if t > p.df_lookahead:
            torch.testing.assert_allclose(torch.from_numpy(out[0]), spec_f[0, 0, t])
    return

    # print("exporting whole dfnet")
    # args = (spec, feat_erb, feat_spec, hemb, hdf)
    # torch.onnx.export(
    #     model=net,
    #     args=args,
    #     input_names=["spec", "feat_erb", "feat_spec"],
    #     f=os.path.join(output_dir, "dfnet.onnx"),
    #     do_constant_folding=constant_folding,
    #     opset_version=opset_version,
    #     dynamic_axes={
    #         "spec": {2: "time"},
    #         "feat_erb": {2: "time"},
    #         "feat_spec": {2: "time"},
    #     },
    #     verbose=False,
    # )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=str, help="Directory e.g. for checkpoint loading.")
    args = parser.parse_args()
    if not os.path.isdir(args.base_dir):
        NotADirectoryError("Base directory not found at {}".format(args.base_dir))
    init_logger(file=os.path.join(args.base_dir, "export.log"))
    cfg_path = os.path.join(args.base_dir, "config.ini")
    config.load(cfg_path)
    checkpoint_dir = os.path.join(args.base_dir, "checkpoints")
    out_dir = os.path.join(args.base_dir, "export")
    os.makedirs(out_dir, exist_ok=True)

    p = ModelParams()
    df_state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    model, _ = load_model(checkpoint_dir, df_state)
    export(model, df_state, out_dir, constant_folding=False, opset_version=13)
    config.save(cfg_path)


if __name__ == "__main__":
    main()
