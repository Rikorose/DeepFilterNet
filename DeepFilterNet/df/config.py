import os
import string
from configparser import ConfigParser
from shlex import shlex
from typing import Any, List, Optional, Tuple, Type, TypeVar, Union

from loguru import logger

T = TypeVar("T")


class DfParams:
    def __init__(self):
        self.sr: int = config("SR", cast=int, default=48_000, section="DF")
        self.fft_size: int = config("FFT_SIZE", cast=int, default=384, section="DF")
        self.hop_size: int = config("HOP_SIZE", cast=int, default=192, section="DF")
        self.nb_erb: int = config("NB_ERB", cast=int, default=16, section="DF")
        self.nb_df: int = config("NB_DF", cast=int, default=24, section="DF")
        self.norm_tau: float = config("NORM_TAU", 1, float, section="DF")
        self.lsnr_max: int = config("LSNR_MAX", 35, int, section="DF")
        self.lsnr_min: int = config("LSNR_MIN", -15, int, section="DF")
        self.min_nb_freqs = config("MIN_NB_ERB_FREQS", 1, int, section="DF")


class Config:
    """Adopted from python-decouple"""

    DEFAULT_SECTION = "settings"

    def __init__(self):
        self.parser: ConfigParser = None  # type: ignore
        self.path = ""
        self.modified = False

    def load(self, path: Optional[str], doraise=False):
        if self.parser is not None:
            raise ValueError("Config already loaded")
        self.parser = ConfigParser()
        self.path = path
        if path is not None and os.path.isfile(path):
            with open(path) as f:
                self.parser.read_file(f)
        else:
            if doraise:
                raise ValueError("No config file found.")
        if not self.parser.has_section(self.DEFAULT_SECTION):
            self.parser.add_section(self.DEFAULT_SECTION)

    def use_defaults(self):
        self.load(path=None, doraise=False)

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

    def set(self, section: str, option: str, value: T, cast: Type[T]):
        section = section.lower()
        if not self.parser.has_section(section):
            raise ValueError(f"Section not found: {section}")
        if self.parser.has_option(section, option):
            if value == self.cast(self.parser.get(section, option), cast):
                return
        self.modified = True
        self.parser.set(section, option, self.tostr(value, cast))

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
        if option in os.environ:
            value = os.environ[option]
            if save:
                self.parser.set(section, option, self.tostr(value, cast))
        elif self.parser.has_option(section, option):
            value = self.read_from_section(section, option, default, cast, save)
        elif self.parser.has_option(section.lower(), option):
            value = self.read_from_section(section.lower(), option, cast, save)
        elif self.parser.has_option(self.DEFAULT_SECTION, option):
            logger.warning(
                f"Couldn't find option {option} in section {section}. "
                "Falling back to default settings section."
            )
            value = self.read_from_section(self.DEFAULT_SECTION, option, cast=cast, save=save)
        elif default is None:
            raise ValueError("Value {} not found.".format(option))
        else:
            value = default
            if save:
                self.set(section, option, value, cast)
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

    def read_from_section(
        self, section: str, option: str, default: Any = None, cast: Type = str, save: bool = True
    ) -> str:
        value = self.parser.get(section, option)
        if not save:
            # Set to default or remove to not read it at trainig start again
            if default is None:
                self.parser.remove_option(section, option)
            else:
                self.parser.set(section, option, self.tostr(default, cast))
        elif section.lower() != section:
            self.parser.set(section.lower(), option, self.tostr(value, cast))
            self.parser.remove_option(section, option)
            self.modified = True
        return value

    def overwrite(self, section: str, option: str, value: str):
        if not self.parser.has_section(section):
            return ValueError(f"Section not found: '{section}'")
        if not self.parser.has_option(section, option):
            return ValueError(f"Option not found '{option}' in section '{section}'")
        self.modified = True
        cast = type(value)
        return self.parser.set(section, option, self.tostr(value, cast))

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
