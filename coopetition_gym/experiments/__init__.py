"""Installed experiment commands; see the repository REPRODUCE.md."""
from . import config

__version__ = config.VERSION
__all__ = ["config"]


def get_algorithm_class(class_name: str):
    """Lazily load algorithms and their training dependencies."""
    from . import algorithms
    return algorithms.get_algorithm_class(class_name)
