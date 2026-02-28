_BASE_VERSION = "0.1.0"

try:
    from ._version import __version__
except ImportError:
    __version__ = _BASE_VERSION
