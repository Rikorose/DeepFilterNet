import os
import sys
from configparser import ConfigParser
from socket import gethostname
from typing import List, Tuple


def update_batch_size(
    host_key: str,
    config_parser: ConfigParser,
    host_bs_parser: ConfigParser,
    config_key: str,
    batchconfig_key: str,
) -> Tuple[bool, bool]:
    current_bs = None
    if config_parser.has_section("train"):
        current_bs = config_parser.get("train", config_key, fallback=None)
    batchconfig_bs = host_bs_parser.get(host_key, batchconfig_key, fallback=None)
    config_changed = False
    host_bs_changed = False
    if batchconfig_bs is not None:
        # Overwrite training config
        if current_bs is not None and batchconfig_bs != current_bs:
            print(
                f"Found host specific {batchconfig_key} ({batchconfig_bs}) for host {host_key}. "
                "Updating config."
            )
            if not config_parser.has_section("train"):
                config_parser.add_section("train")
            config_parser.set("train", config_key, batchconfig_bs)
            config_changed = True
    elif current_bs is not None:
        # No host_key specific batch config found. Store current batch size for current host
        print(
            f"Host specific {batchconfig_key} not found for host {host_key}. "
            "Updating host batch size config."
        )
        if not host_bs_parser.has_section(host_key):
            host_bs_parser.add_section(host_key)
        host_bs_parser.set(host_key, batchconfig_key, current_bs)
        host_bs_changed = True
    return config_changed, host_bs_changed


def cast_bool(value):
    value = str(value).lower()
    if value in {"true", "yes", "y", "on", "1"}:
        return True  # type: ignore
    elif value in {"false", "no", "n", "off", "0"}:
        return False  # type: ignore


def main(config_path, host_bs_config, host_key=None):
    """Sets host specific batch sizes in the specified config."""
    if not os.path.isfile(config_path):
        print(f"Config not found at path {config_path}.")
        exit(2)
    if not os.path.isfile(host_bs_config):
        print(f"Host specific batch size config not found at path {host_bs_config}.")
        print("Creating default.")
        open(host_bs_config, "w").close()
    if host_key is None:
        host_key = gethostname()
    config_parser = ConfigParser()
    host_bs_parser = ConfigParser()
    with open(config_path, "r") as f:
        config_parser.read_file(f)
    with open(host_bs_config, "r") as f:
        host_bs_parser.read_file(f)
    # For eval, don't distinguish between autocasting
    changed: List[Tuple[bool, bool]] = []
    changed.append(
        update_batch_size(
            host_key, config_parser, host_bs_parser, "batch_size_eval", "batch_size_eval"
        )
    )
    autocast_enabled = cast_bool(config_parser.get("train", "train_autocast", fallback=False))
    if autocast_enabled:
        changed.append(
            update_batch_size(
                host_key, config_parser, host_bs_parser, "batch_size", "batch_size_autocast_train"
            )
        )
    else:
        changed.append(
            update_batch_size(
                host_key, config_parser, host_bs_parser, "batch_size", "batch_size_train"
            )
        )
    # check whether the training config was changed
    if any(c[0] for c in changed):
        with open(config_path, "w") as f:
            config_parser.write(f)
    # check whether the host batch size config was changed
    if any(c[1] for c in changed):
        with open(host_bs_config, "w") as f:
            host_bs_parser.write(f)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: set_batch_size <config-path> <host-bs-config-path>")
        exit(1)
    main(sys.argv[1], sys.argv[2])
