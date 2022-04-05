import glob
import os

from loguru import logger

from df.enhance import init_df, save_audio, setup_df_argument_parser
from df.evaluation_utils import evaluation_loop
from df.model import ModelParams


def main(args):
    model, df_state, suffix = init_df(
        args.model_base_dir,
        post_filter=args.pf,
        log_level=args.log_level,
        log_file="dns2020-test.log",
        config_allow_defaults=True,
        epoch=args.epoch,
    )
    assert os.path.isdir(args.dataset_dir)
    sr = ModelParams().sr
    datasets = [os.path.join(args.dataset_dir, "no_reverb")]
    if args.reverb:
        datasets.append(os.path.join(args.dataset_dir, "with_reverb"))

    save_audio_callback = None  # type: ignore
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

        def save_audio_callback(cleanfn: str, enh):
            save_audio(
                os.path.basename(cleanfn), enh, sr, output_dir=args.output_dir, suffix=suffix
            )

    for ds_dir in datasets:
        logger.info(f"Evaluating dataset {os.path.basename(ds_dir)}")
        assert os.path.isdir(ds_dir), ds_dir
        noisy_dir = os.path.join(ds_dir, "noisy")
        clean_dir = os.path.join(ds_dir, "clean")
        assert os.path.isdir(noisy_dir) and os.path.isdir(clean_dir)
        noisy_files = sorted(glob.iglob(noisy_dir + "/*.wav"), key=extract_fileid)
        clean_files = sorted(glob.glob(clean_dir + "/*.wav"), key=extract_fileid)

        metrics = evaluation_loop(
            df_state,
            model,
            clean_files,
            noisy_files,
            n_workers=args.metric_workers,
            save_audio_callback=save_audio_callback,
        )
        for k, v in metrics.items():
            logger.info(f"{k}: {v}")


def extract_fileid(fn: str) -> int:
    return int(os.path.splitext(fn)[0].split("_")[-1])


if __name__ == "__main__":
    parser = setup_df_argument_parser()
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="DNS 2020 Test set directory. Must contain 'no_reverb/clean' and 'no_reverb/noisy'.",
    )
    parser.add_argument(
        "--metric-workers",
        "-w",
        type=int,
        default=4,
        help="Number of worker processes for metric calculation.",
    )
    parser.add_argument(
        "--with-reverb", action="store_true", help="Also test on the reverb dataset", dest="reverb"
    )
    args = parser.parse_args()
    main(args)
