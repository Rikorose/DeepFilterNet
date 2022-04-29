import glob
import os

from loguru import logger

from df.enhance import init_df, save_audio, setup_df_argument_parser
from df.evaluation_utils import evaluation_loop_dns
from df.model import ModelParams


def main(args):
    model, df_state, suffix = init_df(
        args.model_base_dir,
        post_filter=args.pf,
        log_level=args.log_level,
        log_file="voicebank-test.log",
        config_allow_defaults=True,
        epoch=args.epoch,
    )
    assert os.path.isdir(args.dataset_dir)
    sr = ModelParams().sr
    noisy_dir = args.dataset_dir
    noisy_files = glob.glob(noisy_dir + "/*.wav")

    def save_audio_callback(cleanfn: str, enh):
        save_audio(os.path.basename(cleanfn), enh, sr, output_dir=args.output_dir, suffix=suffix)

    methods = list(args.methods) if isinstance(args.methods, (list, tuple)) else [args.methods]
    metrics = evaluation_loop_dns(
        df_state,
        model,
        noisy_files,
        n_workers=args.metric_workers,
        save_audio_callback=save_audio_callback if args.output_dir is not None else None,
        metrics=methods,
    )
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")


if __name__ == "__main__":
    parser = setup_df_argument_parser()
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Voicebank Demand Test set directory. Must contain 'noisy_testset_wav' and 'clean_testset_wav'.",
    )
    parser.add_argument(
        "--metric-workers",
        "-w",
        type=int,
        default=4,
        help="Number of worker processes for metric calculation.",
    )
    parser.add_argument(
        "--methods",
        default="p835",
        nargs="*",
        choices=["p808", "p835"],
        help="Choose which method to compute P.808 or P.835. Default is P.808",
    )
    args = parser.parse_args()
    main(args)
