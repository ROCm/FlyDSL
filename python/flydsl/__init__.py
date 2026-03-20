_BASE_VERSION = "0.1.1"

try:
    from ._version import __version__
except ImportError:
    __version__ = _BASE_VERSION
