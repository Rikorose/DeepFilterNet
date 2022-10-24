import os
import re

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    # for Python <3.8 add 'importlib_metadata' as a dependency
    try:
        import importlib_metadata  # type: ignore
    except ModuleNotFoundError:
        importlib_metadata = None

VERSION_REGEX = r"\s*version\s*=\s*[\"']\s*([-.\w]{3,})\s*[\"']\s*"

try:
    if importlib_metadata is None:
        version = None
    else:
        version = importlib_metadata.version("deepfilternet")
except Exception:
    version = None
    path = os.path.join(os.path.dirname(__file__), os.pardir, "pyproject.toml")
    compiled_version_regex = re.compile(VERSION_REGEX)
    if os.path.isfile(path):
        for line in open(path):
            v_ = compiled_version_regex.search(line)
            if v_ is not None:
                version = v_.group(1).strip()
                break
