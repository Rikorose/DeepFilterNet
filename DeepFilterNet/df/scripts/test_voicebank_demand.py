import glob
import os

from loguru import logger
from torch import Tensor

from df.deepfilternet import ModelParams
from df.enhance import init_df, save_audio, setup_df_argument_parser
from df.evaluation_utils import evaluation_loop


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
    noisy_dir = os.path.join(args.dataset_dir, "noisy_testset_wav")
    clean_dir = os.path.join(args.dataset_dir, "clean_testset_wav")
    assert os.path.isdir(noisy_dir) and os.path.isdir(clean_dir)
    clean_files = sorted(glob.glob(clean_dir + "/*.wav"))
    noisy_files = sorted(glob.glob(noisy_dir + "/*.wav"))
    if args.output_dir is not None:
        logger.debug(f"Setting up output dir: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)

    def save_audio_callback(cleanfn: str, enh: Tensor):
        save_audio(os.path.basename(cleanfn), enh, sr, output_dir=args.output_dir, suffix=suffix)

    metrics = evaluation_loop(
        df_state,
        model,
        clean_files,
        noisy_files,
        n_workers=args.metric_workers,
        save_audio_callback=save_audio_callback if args.output_dir is not None else None,
        metrics=["stoi", "composite", "sisdr"],
        csv_path_enh=args.csv_path_enh,
        csv_path_noisy=args.csv_path_noisy,
        noisy_metric=args.compute_noisy_metric,
        sleep_ms=args.sleep_ms,
    )
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")
    print("".join(f"{m}," for k, m in metrics.items() if "SSNR" not in k)[:-1])


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
    parser.add_argument("--compute-noisy-metric", action="store_true")
    parser.add_argument("--sleep-ms", type=int, default=0)
    args = parser.parse_args()
    main(args)
