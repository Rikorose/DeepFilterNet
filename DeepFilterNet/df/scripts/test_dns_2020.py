import glob
import os
import re

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

    save_audio_callback = None  # type: ignore  # noqa: F811
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

        def save_audio_callback(cleanfn: str, enh):  # noqa: F811
            save_audio(
                os.path.basename(cleanfn), enh, sr, output_dir=args.output_dir, suffix=suffix
            )

    for ds_dir in datasets:
        logger.info(f"Evaluating dataset {os.path.basename(ds_dir)}")
        assert os.path.isdir(ds_dir), ds_dir
        noisy_dir = os.path.join(ds_dir, "noisy")
        clean_dir = os.path.join(ds_dir, "clean")
        assert os.path.isdir(noisy_dir) and os.path.isdir(clean_dir)
        expr = re.compile(r"clnsp.*_fileid")
        noisy_files = glob.glob(noisy_dir + "/*.wav")
        clean_files = [
            re.sub(expr, "clean_fileid", f.replace("noisy", "clean")) for f in noisy_files
        ]
        assert len(clean_files) == 150

        metrics = evaluation_loop(
            df_state,
            model,
            clean_files,
            noisy_files,
            n_workers=args.metric_workers,
            save_audio_callback=save_audio_callback,
            metrics=["stoi", "composite", "sisdr"],
        )
        for k, v in metrics.items():
            logger.info(f"{k}: {v}")


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
