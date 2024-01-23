from typing import List, Tuple

import torch

from df.enhance import init_df, setup_df_argument_parser
from df.logger import log_model_summary


def main(args):
    model, _, _ = init_df(
        args.model_base_dir,
        post_filter=args.pf,
        log_level=args.log_level,
        log_file=None,
        config_allow_defaults=True,
        epoch=None,
    )
    if args.type == "torch":
        print(model)
    elif args.type == "ptflops":
        log_model_summary(model, verbose=True, force=True)
    elif args.type == "table":
        for line in model_summary_table(model):
            print(line)
    else:
        raise NotImplementedError()


def model_summary_table(m: torch.nn.Module):
    def n_params(m: torch.nn.Module):
        return sum(p.numel() for p in m.parameters())

    childs = list(m.named_modules())[1:]
    max_len_name = max(len(n) for n, _ in childs) + 1
    out: List[Tuple[str, str]] = []
    # Heading
    out.append(("Name".ljust(max_len_name, " "), "Parameters"))
    out.append(("-" * len(out[0][0]), "-" * len(out[0][1])))
    # Add each module that has trainable parameters
    for name, c_m in childs:
        # Filter out parent modules that have children containing parameters
        if any(lambda c: n_params(c) > 0 for c in c_m.children()):
            continue
        if n_params(c_m) > 0:
            out.append((name.ljust(max_len_name, " "), f"{n_params(c_m):,}"))
    # Add sum of all parameters
    out.append(("-" * len(out[0][0]), "-" * len(out[0][1])))
    out.append(("Sum".ljust(max_len_name), f"{n_params(m):,}"))
    # Calculate max len of the parameters
    max_len_params = max(len(p) for _, p in out)
    # Left pad params and concat the strings
    for i in range(len(out)):
        out[i] = out[i][0] + out[i][1].rjust(max_len_params)
    return out


if __name__ == "__main__":
    parser = setup_df_argument_parser("INFO")
    parser.add_argument("--type", choices=["torch", "ptflops", "table"], default="torch")
    main(parser.parse_args())
