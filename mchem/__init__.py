"""M-Chem front-end package."""

from importlib.metadata import PackageNotFoundError, version as _package_version

try:
    __version__ = _package_version("mchem")
except PackageNotFoundError:  # running from a source tree that is not installed
    try:
        from ._version import __version__  # type: ignore[no-redef]
    except ImportError:
        __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
