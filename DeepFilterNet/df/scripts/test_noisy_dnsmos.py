import glob
import os

import torch
from loguru import logger

from df.enhance import init_df, save_audio, setup_df_argument_parser
from df.evaluation_utils import evaluation_loop_dns
from df.model import ModelParams


def main(args):
    model, df_state, suffix = init_df(
        args.model_base_dir,
        post_filter=args.pf,
        log_level=args.log_level,
        log_file="dnsmos-test.log",
        config_allow_defaults=True,
        epoch=args.epoch,
    )
    assert os.path.isdir(args.dataset_dir)
    sr = ModelParams().sr
    noisy_dir = args.dataset_dir
    noisy_files = glob.glob(noisy_dir + "/*.wav")
    n = len(noisy_files)
    logger.info(f"Evaluating DNSMOS on enhanced noisy files for dir: {noisy_dir} ({n} files)")
    assert n == 859
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    def save_audio_callback(cleanfn: str, enh):
        save_audio(
            os.path.basename(cleanfn),
            enh,
            sr,
            output_dir=args.output_dir,
            suffix=suffix,
            dtype=torch.float32,
        )

    metrics = evaluation_loop_dns(
        df_state,
        model,
        noisy_files,
        n_workers=args.metric_workers,
        save_audio_callback=save_audio_callback if args.output_dir is not None else None,
        metrics=["p835_local"],
        csv_path_enh=args.csv_path_enh,
        csv_path_noisy=args.csv_path_noisy,
        eval_noisy=False,
    )
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")


if __name__ == "__main__":
    parser = setup_df_argument_parser()
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Test set directory containing noisy audio files.",
    )
    parser.add_argument(
        "--metric-workers",
        "-w",
        type=int,
        default=4,
        help="Number of worker processes for metric calculation.",
    )
    parser.add_argument(
        "--csv-path-enh",
        type=str,
        default=None,
        help="Path to csv score file containing metrics of enhanced audios.",
    )
    parser.add_argument(
        "--csv-path-noisy",
        type=str,
        default=None,
        help="Path to csv score file containing metrics of noisy audios.",
    )
    args = parser.parse_args()
    main(args)
