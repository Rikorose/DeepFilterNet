import os
import shutil
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import onnx
import onnx.checker
import onnx.helper
import onnxruntime as ort
import torch
from loguru import logger
from torch import Tensor

from df.enhance import (
    ModelParams,
    df_features,
    enhance,
    get_model_basedir,
    init_df,
    setup_df_argument_parser,
)
from df.io import get_test_sample, save_audio
from libdf import DF


def shapes_dict(
    tensors: Tuple[Tensor], names: Union[Tuple[str], List[str]]
) -> Dict[str, Tuple[int]]:
    if len(tensors) != len(names):
        logger.warning(
            f"  Number of tensors ({len(tensors)}) does not match provided names: {names}"
        )
    return {k: v.shape for (k, v) in zip(names, tensors)}


def onnx_simplify(
    path: str, input_data: Dict[str, Tensor], input_shapes: Dict[str, Iterable[int]]
) -> str:
    import onnxsim

    model = onnx.load(path)
    model_simp, check = onnxsim.simplify(
        model,
        input_data=input_data,
        test_input_shapes=input_shapes,
    )
    model_n = os.path.splitext(os.path.basename(path))[0]
    assert check, "Simplified ONNX model could not be validated"
    logger.debug(model_n + ": " + onnx.helper.printable_graph(model.graph))
    try:
        onnx.checker.check_model(model_simp, full_check=True)
    except Exception as e:
        logger.error(f"Failed to simplify model {model_n}. Skipping: {e}")
        return path
    # new_path = os.path.join(os.path.dirname(path), model_n + "_simplified.onnx")
    onnx.save_model(model_simp, path)
    return path


def onnx_check(path: str, input_dict: Dict[str, Tensor], output_names: Tuple[str]):
    model = onnx.load(path)
    logger.debug(os.path.basename(path) + ": " + onnx.helper.printable_graph(model.graph))
    onnx.checker.check_model(model, full_check=True)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(output_names, {k: v.numpy() for (k, v) in input_dict.items()})


def export_impl(
    path: str,
    model: torch.nn.Module,
    inputs: Tuple[Tensor, ...],
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
    jit: bool = True,
    opset_version=14,
    check: bool = True,
    simplify: bool = True,
    print_graph: bool = False,
):
    export_dir = os.path.dirname(path)
    if not os.path.isdir(export_dir):
        logger.info(f"Creating export directory: {export_dir}")
        os.makedirs(export_dir)
    model_name = os.path.splitext(os.path.basename(path))[0]
    logger.info(f"Exporting model '{model_name}' to {export_dir}")

    input_shapes = shapes_dict(inputs, input_names)
    logger.info(f"  Input shapes: {input_shapes}")

    outputs = model(*inputs)
    output_shapes = shapes_dict(outputs, output_names)
    logger.info(f"  Output shapes: {output_shapes}")

    if jit:
        model = torch.jit.script(model, example_inputs=[tuple(a for a in inputs)])

    logger.info(f"  Dynamic axis: {dynamic_axes}")
    torch.onnx.export(
        model=deepcopy(model),
        f=path,
        args=inputs,
        input_names=input_names,
        dynamic_axes=dynamic_axes,
        output_names=output_names,
        opset_version=opset_version,
        keep_initializers_as_inputs=False,
    )

    input_dict = {k: v for (k, v) in zip(input_names, inputs)}
    if check:
        onnx_outputs = onnx_check(path, input_dict, tuple(output_names))
        for name, out, onnx_out in zip(output_names, outputs, onnx_outputs):
            try:
                np.testing.assert_allclose(
                    out.numpy().squeeze(), onnx_out.squeeze(), rtol=1e-6, atol=1e-5
                )
            except AssertionError as e:
                logger.warning(f"  Elements not close for {name}: {e}")
    if simplify:
        path = onnx_simplify(path, input_dict, shapes_dict(inputs, input_names))
        logger.info(f"  Saved simplified model {path}")
    if print_graph:
        onnx.helper.printable_graph(onnx.load_model(path).graph)

    return outputs


@torch.no_grad()
def export(
    model,
    export_dir: str,
    df_state: DF,
    check: bool = True,
    simplify: bool = True,
    opset=14,
    export_full: bool = False,
    print_graph: bool = False,
):
    model = deepcopy(model).to("cpu")
    model.eval()
    p = ModelParams()
    audio = torch.randn((1, 1 * p.sr))
    spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device="cpu")

    # Export full model
    if export_full:
        path = os.path.join(export_dir, "deepfilternet2.onnx")
        input_names = ["spec", "feat_erb", "feat_spec"]
        dynamic_axes = {
            "spec": {2: "S"},
            "feat_erb": {2: "S"},
            "feat_spec": {2: "S"},
            "enh": {2: "S"},
            "m": {2: "S"},
            "lsnr": {1: "S"},
        }
        inputs = (spec, feat_erb, feat_spec)
        output_names = ["enh", "m", "lsnr", "coefs"]
        export_impl(
            path,
            model,
            inputs=inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            jit=False,
            check=check,
            simplify=simplify,
            opset_version=opset,
            print_graph=print_graph,
        )

    # Export encoder
    feat_spec = feat_spec.transpose(1, 4).squeeze(4)  # re/im into channel axis
    path = os.path.join(export_dir, "enc.onnx")
    inputs = (feat_erb, feat_spec)
    input_names = ["feat_erb", "feat_spec"]
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
    output_names = ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"]
    e0, e1, e2, e3, emb, c0, lsnr = export_impl(
        path,
        model.enc,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        jit=True,
        check=check,
        simplify=simplify,
        opset_version=opset,
        print_graph=print_graph,
    )
    np.savez_compressed(
        os.path.join(export_dir, "enc_input.npz"),
        feat_erb=feat_erb.numpy(),
        feat_spec=feat_spec.numpy(),
    )
    np.savez_compressed(
        os.path.join(export_dir, "enc_output.npz"),
        e0=e0.numpy(),
        e1=e1.numpy(),
        e2=e2.numpy(),
        e3=e3.numpy(),
        emb=emb.numpy(),
        c0=c0.numpy(),
        lsnr=lsnr.numpy(),
    )

    # Export erb decoder
    np.savez_compressed(
        os.path.join(export_dir, "erb_dec_input.npz"),
        emb=emb.numpy(),
        e0=e0.numpy(),
        e1=e1.numpy(),
        e2=e2.numpy(),
        e3=e3.numpy(),
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
    path = os.path.join(export_dir, "erb_dec.onnx")
    m = export_impl(  # noqa
        path,
        model.erb_dec,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        jit=True,
        check=check,
        simplify=simplify,
        opset_version=opset,
        print_graph=print_graph,
    )
    np.savez_compressed(os.path.join(export_dir, "erb_dec_output.npz"), m=m.numpy())

    # Export df decoder
    np.savez_compressed(
        os.path.join(export_dir, "df_dec_input.npz"), emb=emb.numpy(), c0=c0.numpy()
    )
    inputs = (emb.clone(), c0)
    input_names = ["emb", "c0"]
    output_names = ["coefs"]
    dynamic_axes = {
        "emb": {1: "S"},
        "c0": {2: "S"},
        "coefs": {1: "S"},
    }
    path = os.path.join(export_dir, "df_dec.onnx")
    coefs = export_impl(  # noqa
        path,
        model.df_dec,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        jit=False,
        check=check,
        simplify=simplify,
        opset_version=opset,
        print_graph=print_graph,
    )
    np.savez_compressed(os.path.join(export_dir, "df_dec_output.npz"), coefs=coefs.numpy())


def main(args):
    try:
        import monkeytype  # noqa: F401
    except ImportError:
        print("Failed to import monkeytype. Please install it via")
        print("$ pip install MonkeyType")
        exit(1)

    print(args)
    model, df_state, _, epoch = init_df(
        args.model_base_dir,
        post_filter=args.pf,
        log_level=args.log_level,
        log_file="export.log",
        config_allow_defaults=True,
        epoch=args.epoch,
    )
    sample = get_test_sample(df_state.sr())
    enhanced = enhance(model, df_state, sample, True)
    out_dir = Path("out")
    if out_dir.is_dir():
        # attempt saving enhanced audio
        save_audio(os.path.join(out_dir, "enhanced.wav"), enhanced, df_state.sr())
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    export(
        model,
        export_dir,
        df_state=df_state,
        opset=args.opset,
        check=args.check,
        simplify=args.simplify,
    )
    model_base_dir = get_model_basedir(args.model_base_dir)
    if model_base_dir != args.export_dir:
        shutil.copyfile(
            os.path.join(model_base_dir, "config.ini"),
            os.path.join(args.export_dir, "config.ini"),
        )
    model_name = Path(model_base_dir).name
    version_file = os.path.join(args.export_dir, "version.txt")
    with open(version_file, "w") as f:
        f.write(f"{model_name}_epoch_{epoch}")
    tar_name = export_dir / (Path(model_base_dir).name + "_onnx.tar.gz")
    with tarfile.open(tar_name, mode="w:gz") as f:
        f.add(os.path.join(args.export_dir, "enc.onnx"))
        f.add(os.path.join(args.export_dir, "erb_dec.onnx"))
        f.add(os.path.join(args.export_dir, "df_dec.onnx"))
        f.add(os.path.join(args.export_dir, "config.ini"))
        f.add(os.path.join(args.export_dir, "version.txt"))


if __name__ == "__main__":
    parser = setup_df_argument_parser()
    parser.add_argument("export_dir", help="Directory for exporting the onnx model.")
    parser.add_argument(
        "--no-check",
        help="Don't check models with onnx checker.",
        action="store_false",
        dest="check",
    )
    parser.add_argument("--simplify", help="Simply onnx models using onnxsim.", action="store_true")
    parser.add_argument("--opset", help="ONNX opset version", type=int, default=12)
    args = parser.parse_args()
    main(args)
