"""Reproducibility package for the Coopetition-Gym NeurIPS 2026 submission.

This package contains the consolidated orchestration, evaluation, analysis,
and validation code used to produce the 25,708-file training dataset and
the 1,116-file behavioral audit dataset.

See :doc:`../REPRODUCE` for reproduction instructions and :doc:`README` for
module layout and design principles.
"""

import os
import sys

# When this package is run from the repository root
# (e.g. ``python -m experiments.audit``), Python inserts the repo root at
# the front of ``sys.path``. Because the repository layout has a top-level
# folder named ``coopetition_gym/`` with no ``__init__.py`` at the top level
# (the package lives at ``coopetition_gym/coopetition_gym/``), that folder
# is found as a namespace package and shadows the installed editable
# ``coopetition_gym``. Move the repo root to the end of ``sys.path`` so that
# site-packages resolves ``coopetition_gym`` correctly.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_remaining = [p for p in sys.path if os.path.abspath(p or ".") != _repo_root]
sys.path = _remaining + [_repo_root]

from experiments import config  # noqa: E402

__version__ = config.VERSION
__all__ = ["config"]


def get_algorithm_class(class_name: str):
    """Convenience re-export of :func:`experiments.algorithms.get_algorithm_class`.

    Importing :mod:`experiments.algorithms` has heavier dependencies (torch,
    gymnasium), so it is lazy-loaded the first time this helper is called.
    """
    from experiments import algorithms
    return algorithms.get_algorithm_class(class_name)
