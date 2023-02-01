import os
import string
from configparser import ConfigParser
from shlex import shlex
from typing import Any, List, Optional, Tuple, Type, TypeVar, Union

from loguru import logger

T = TypeVar("T")


class DfParams:
    def __init__(self):
        # Sampling rate used for training
        self.sr: int = config("SR", cast=int, default=48_000, section="DF")
        # FFT size in samples
        self.fft_size: int = config("FFT_SIZE", cast=int, default=960, section="DF")
        # STFT Hop size in samples
        self.hop_size: int = config("HOP_SIZE", cast=int, default=480, section="DF")
        # Number of ERB bands
        self.nb_erb: int = config("NB_ERB", cast=int, default=32, section="DF")
        # Number of deep filtering bins; DF is applied from 0th to nb_df-th frequency bins
        self.nb_df: int = config("NB_DF", cast=int, default=96, section="DF")
        # Normalization decay factor; used for complex and erb features
        self.norm_tau: float = config("NORM_TAU", 1, float, section="DF")
        # Local SNR minimum value, ground truth will be truncated
        self.lsnr_max: int = config("LSNR_MAX", 35, int, section="DF")
        # Local SNR maximum value, ground truth will be truncated
        self.lsnr_min: int = config("LSNR_MIN", -15, int, section="DF")
        # Minimum number of frequency bins per ERB band
        self.min_nb_freqs = config("MIN_NB_ERB_FREQS", 2, int, section="DF")
        # Deep Filtering order
        self.df_order: int = config("DF_ORDER", cast=int, default=5, section="DF")
        # Deep Filtering look-ahead
        self.df_lookahead: int = config("DF_LOOKAHEAD", cast=int, default=0, section="DF")
        # Pad mode. By default, padding will be handled on the input side:
        # - `input`, which pads the input features passed to the model
        # - `output`, which pads the output spectrogram corresponding to `df_lookahead`
        self.pad_mode: str = config("PAD_MODE", default="input", section="DF")


class Config:
    """Adopted from python-decouple"""

    DEFAULT_SECTION = "settings"

    def __init__(self):
        self.parser: ConfigParser = None  # type: ignore
        self.path = ""
        self.modified = False
        self.allow_defaults = True

    def load(
        self, path: Optional[str], config_must_exist=False, allow_defaults=True, allow_reload=False
    ):
        self.allow_defaults = allow_defaults
        if self.parser is not None and not allow_reload:
            raise ValueError("Config already loaded")
        self.parser = ConfigParser()
        self.path = path
        if path is not None and os.path.isfile(path):
            with open(path) as f:
                self.parser.read_file(f)
        else:
            if config_must_exist:
                raise ValueError(f"No config file found at '{path}'.")
        if not self.parser.has_section(self.DEFAULT_SECTION):
            self.parser.add_section(self.DEFAULT_SECTION)
        self._fix_clc()
        self._fix_df()

    def use_defaults(self, allow_reload=False):
        self.load(path=None, config_must_exist=False, allow_reload=allow_reload)

    def save(self, path: str):
        if not self.modified:
            logger.debug("Config not modified. No need to overwrite on disk.")
            return
        if self.parser is None:
            self.parser = ConfigParser()
        for section in self.parser.sections():
            if len(self.parser[section]) == 0:
                self.parser.remove_section(section)
        with open(path, mode="w") as f:
            self.parser.write(f)

    def tostr(self, value, cast):
        if isinstance(cast, Csv) and isinstance(value, (tuple, list)):
            return "".join(str(v) + cast.delimiter for v in value)[:-1]
        return str(value)

    def set(self, option: str, value: T, cast: Type[T], section: Optional[str] = None) -> T:
        section = self.DEFAULT_SECTION if section is None else section
        section = section.lower()
        if not self.parser.has_section(section):
            self.parser.add_section(section)
        if self.parser.has_option(section, option):
            if value == self.cast(self.parser.get(section, option), cast):
                return value
        self.modified = True
        self.parser.set(section, option, self.tostr(value, cast))
        return value

    def __call__(
        self,
        option: str,
        default: Any = None,
        cast: Type[T] = str,
        save: bool = True,
        section: Optional[str] = None,
    ) -> T:
        # Get value either from an ENV or from the .ini file
        section = self.DEFAULT_SECTION if section is None else section
        value = None
        if self.parser is None:
            raise ValueError("No configuration loaded")
        if not self.parser.has_section(section.lower()):
            self.parser.add_section(section.lower())
        if option.upper() in os.environ:
            value = os.environ[option.upper()]
            if save:
                self.parser.set(section, option, self.tostr(value, cast))
        elif self.parser.has_option(section, option):
            value = self.parser.get(section, option)
        elif self.parser.has_option(section.lower(), option):
            value = self.parser.get(section.lower(), option)
        elif self.parser.has_option(self.DEFAULT_SECTION, option):
            logger.warning(
                f"Couldn't find option {option} in section {section}. "
                "Falling back to default settings section."
            )
            value = self.parser.get(self.DEFAULT_SECTION, option)
        elif default is None:
            raise ValueError("Value {} not found.".format(option))
        elif not self.allow_defaults and save:
            raise ValueError(f"Value '{option}' not found in config (defaults not allowed).")
        else:
            value = default
            if save:
                self.set(option, value, cast, section)
        return self.cast(value, cast)

    def cast(self, value, cast):
        # Do the casting to get the correct type
        if cast is bool:
            value = str(value).lower()
            if value in {"true", "yes", "y", "on", "1"}:
                return True  # type: ignore
            elif value in {"false", "no", "n", "off", "0"}:
                return False  # type: ignore
            raise ValueError("Parse error")
        return cast(value)

    def get(self, option: str, cast: Type[T] = str, section: Optional[str] = None) -> T:
        section = self.DEFAULT_SECTION if section is None else section
        if not self.parser.has_section(section):
            raise KeyError(section)
        if not self.parser.has_option(section, option):
            raise KeyError(option)
        return self.cast(self.parser.get(section, option), cast)

    def overwrite(self, section: str, option: str, value: Any):
        if not self.parser.has_section(section):
            return ValueError(f"Section not found: '{section}'")
        if not self.parser.has_option(section, option):
            return ValueError(f"Option not found '{option}' in section '{section}'")
        self.modified = True
        cast = type(value)
        return self.parser.set(section, option, self.tostr(value, cast))

    def _fix_df(self):
        """Renaming of some groups/options for compatibility with old models."""
        if self.parser.has_section("deepfilternet") and self.parser.has_section("df"):
            sec_deepfilternet = self.parser["deepfilternet"]
            sec_df = self.parser["df"]
            if "df_order" in sec_deepfilternet:
                sec_df["df_order"] = sec_deepfilternet["df_order"]
                del sec_deepfilternet["df_order"]
            if "df_lookahead" in sec_deepfilternet:
                sec_df["df_lookahead"] = sec_deepfilternet["df_lookahead"]
                del sec_deepfilternet["df_lookahead"]
        if self.parser.has_section("train") and "p_reverb" in self.parser["train"]:
            if not self.parser.has_section("distortion"):
                self.parser.add_section("distortion")
            self.parser["distortion"]["p_reverb"] = self.parser["train"]["p_reverb"]
            del self.parser["train"]["p_reverb"]

    def _fix_clc(self):
        """Renaming of some groups/options for compatibility with old models."""
        if (
            not self.parser.has_section("deepfilternet")
            and self.parser.has_section("train")
            and self.parser.get("train", "model") == "convgru5"
        ):
            self.overwrite("train", "model", "deepfilternet")
            self.parser.add_section("deepfilternet")
            self.parser["deepfilternet"] = self.parser["convgru"]
            del self.parser["convgru"]
        if not self.parser.has_section("df") and self.parser.has_section("clc"):
            self.parser["df"] = self.parser["clc"]
            del self.parser["clc"]
        for section in self.parser.sections():
            for k, v in self.parser[section].items():
                if "clc" in k.lower():
                    self.parser.set(section, k.lower().replace("clc", "df"), v)
                    del self.parser[section][k]

    def __repr__(self):
        msg = ""
        for section in self.parser.sections():
            msg += f"{section}:\n"
            for k, v in self.parser[section].items():
                msg += f"  {k}: {v}\n"
        return msg


config = Config()


class Csv(object):
    """
    Produces a csv parser that return a list of transformed elements. From python-decouple.
    """

    def __init__(
        self, cast: Type[T] = str, delimiter=",", strip=string.whitespace, post_process=list
    ):
        """
        Parameters:
        cast -- callable that transforms the item just before it's added to the list.
        delimiter -- string of delimiters chars passed to shlex.
        strip -- string of non-relevant characters to be passed to str.strip after the split.
        post_process -- callable to post process all casted values. Default is `list`.
        """
        self.cast: Type[T] = cast
        self.delimiter = delimiter
        self.strip = strip
        self.post_process = post_process

    def __call__(self, value: Union[str, Tuple[T], List[T]]) -> List[T]:
        """The actual transformation"""
        if isinstance(value, (tuple, list)):
            # if default value is a list
            value = "".join(str(v) + self.delimiter for v in value)[:-1]

        def transform(s):
            return self.cast(s.strip(self.strip))

        splitter = shlex(value, posix=True)
        splitter.whitespace = self.delimiter
        splitter.whitespace_split = True

        return self.post_process(transform(s) for s in splitter)
