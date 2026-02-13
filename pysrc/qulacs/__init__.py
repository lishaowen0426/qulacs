from qulacs_core import *

try:
    from qulacs._version import __version__, __version_tuple__
except ModuleNotFoundError:
    __version__ = "0.0.0"
    __version_tuple__ = (0, 0, 0)
