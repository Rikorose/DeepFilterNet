import os
import sys

from df.enhance import init_df


def main():
    dir = sys.argv[1] if len(sys.argv) > 1 else None
    if dir is not None:
        assert os.path.isdir(dir), f"Model base directory not found: {dir}"
    model, _, _ = init_df(dir, log_level="ERROR", config_allow_defaults=True)
    print(model)


if __name__ == "__main__":
    main()
