import os
from copy import deepcopy

import torch

from df.enhance import ModelParams, df_features, init_df, setup_df_argument_parser
from libdf import DF


@torch.no_grad()
def export(model, export_dir: str, df_state: DF):
    model = deepcopy(model).to("cpu")
    model.eval()
    p = ModelParams()
    audio = torch.randn((1, 1 * p.sr))
    spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device="cpu")
    enh = model(spec, feat_erb, feat_spec)[0]
    print(enh.shape)
    torch.onnx.export(
        model=deepcopy(model),
        f=os.path.join(export_dir, "deepfilternet2.onnx"),
        args=(spec, feat_erb, feat_spec),
        input_names=["spec", "feat_erb", "feat_spec"],
        dynamic_axes={
            "spec": {3: "time"},
            "feat_erb": {3: "time"},
            "feat_spec": {3: "time"},
            "enh": {3: "time"},
            "m": {3: "time"},
            "lsnr": {2: "time"},
        },
        output_names=["enh", "m", "lsnr"],
        opset_version=14,
    )


def main(args):
    print(args)
    model, df_state, _ = init_df(
        args.model_base_dir,
        post_filter=args.pf,
        log_level=args.log_level,
        log_file="export.log",
        config_allow_defaults=True,
    )
    if not os.path.isdir(args.export_dir):
        os.makedirs(args.export_dir)
    export(model, args.export_dir, df_state=df_state)


if __name__ == "__main__":
    parser = setup_df_argument_parser()
    parser.add_argument("export_dir", help="Directory for exporting the onnx model.")
    args = parser.parse_args()
    main(args)
