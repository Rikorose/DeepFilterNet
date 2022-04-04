import glob
import os
from typing import List

import torch
import torch.multiprocessing as mp
from loguru import logger

from df.enhance import df_features, init_df, load_audio, save_audio, setup_df_argument_parser
from df.evaluation_utils import HAS_OCTAVE, CompositeMetric, SiSDRMetric, StoiMetric, tqdm
from df.model import ModelParams
from df.modules import get_device
from df.utils import as_complex


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
    if not HAS_OCTAVE:
        logger.warning("Running without octave. Skipping composite metrics")
    sr = ModelParams().sr
    noisy_dir = os.path.join(args.dataset_dir, "noisy_testset_wav")
    clean_dir = os.path.join(args.dataset_dir, "clean_testset_wav")
    assert os.path.isdir(noisy_dir) and os.path.isdir(clean_dir)
    with mp.Pool(processes=args.metric_workers) as pool:
        metrics: List[Metric] = [StoiMetric(sr, pool), SiSDRMetric(pool), CompositeMetric(sr, pool)]  # type: ignore
        noisy_files = glob.glob(noisy_dir + "/*wav")
        clean_files = glob.glob(clean_dir + "/*wav")
        for noisyfn, cleanfn in tqdm(zip(noisy_files, clean_files), total=len(noisy_files)):
            noisy, _ = load_audio(noisyfn, sr)
            clean, _ = load_audio(cleanfn, sr)
            enh = enhance(model, df_state, noisy)[0]
            clean = df_state.synthesis(df_state.analysis(clean.numpy()))[0]
            noisy = df_state.synthesis(df_state.analysis(noisy.numpy()))[0]
            for m in metrics:
                m.add(clean=clean, enhanced=enh, noisy=noisy)
            enh = torch.as_tensor(enh).to(torch.float32).view(1, -1)
            if args.output_dir is not None:
                save_audio(
                    os.path.basename(cleanfn),
                    enh,
                    sr,
                    output_dir=args.output_dir,
                    suffix=f"{suffix}",
                )
        del model, noisy, clean, enh
        torch.cuda.empty_cache()
        logger.info("Waiting for metrics computation completion. This could take a few minutes.")
        for m in metrics:
            for k, v in m.mean().items():
                logger.info(f"{k}: {v}")


@torch.no_grad()
def enhance(model, df_state, audio):
    model.eval()
    if hasattr(model, "reset_h0"):
        model.reset_h0(batch_size=1, device=get_device())
    spec, erb_feat, spec_feat = df_features(audio, df_state, get_device())
    spec = model(spec, erb_feat, spec_feat)[0]
    return df_state.synthesis(as_complex(spec.squeeze(0)).cpu().numpy())


if __name__ == "__main__":
    parser = setup_df_argument_parser()
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Voicebank Demand Test set directory. Must contain 'noisy_testset_wav' and 'clean_testset_wav'.",
    )
    parser.add_argument(
        "--metric-workers",
        type=int,
        default=4,
        help="Number of worker processes for metric calculation.",
    )
    args = parser.parse_args()
    main(args)
