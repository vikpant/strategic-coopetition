# =============================================================================
# THREAD LIMITING - MUST BE SET BEFORE ANY IMPORTS
# =============================================================================
# PyTorch, NumPy, and other libraries spawn many threads by default. On
# many-core cloud instances (100+ vCPUs) with 80+ worker processes this
# causes severe CPU oversubscription. Limit threads per process.
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
# =============================================================================

"""Unified campaign orchestrator for the NeurIPS 2026 benchmark.

Single entry point for running training campaigns across all three reward
configurations used in the paper. Consolidates the original
``orchestrator.py`` and ``orchestrator_reward_ablation.py`` scripts into one
command-line tool with subcommands:

* ``baseline`` — Main experimental campaign with integrated reward.
* ``private`` — Private-reward ablation (:math:`U_i = \\pi_i`,
  :math:`D_{ij}=0`).
* ``cooperative`` — Cooperative-reward ablation (fully shared reward).

The network sensitivity analysis is handled separately by
:mod:`experiments.sensitivity` because it varies algorithm hyperparameters
per-experiment (``net_arch`` override) rather than running a single
configuration against all algorithms.

Safety defaults (all enabled by default; each has an opt-out flag):

* Checkpoints every 100,000 steps (``--no-checkpoints`` to disable)
* GPU memory monitoring (``--no-monitoring`` to disable)
* Dynamic backpressure on memory pressure (``--no-backpressure`` to disable)
* Thermal monitoring (``--no-thermal-monitoring`` to disable)

These defaults reflect lessons learned from the original campaign where
disabled checkpoints caused lost GPU-hours and disk pressure filled
instances repeatedly.

Reward-type ablation mechanism:
    The reward type is passed via the ``COOPETITION_REWARD_TYPE`` environment
    variable. A ``.pth``-triggered patcher (``reward_type_patcher.py`` in
    site-packages) reads this variable at Python startup in every process
    (including ``multiprocessing.spawn`` children) and patches
    ``coopetition_gym.make()`` to apply ``env.reward_type`` post-construction.
    The patcher must be installed in the venv's site-packages; see
    :doc:`../REPRODUCE` for setup instructions.

Usage::

    # Main campaign
    python -m experiments.campaign baseline --output data/training/baseline_integrated/

    # Private reward ablation
    python -m experiments.campaign private --output data/training/ablation_private/

    # Cooperative reward ablation
    python -m experiments.campaign cooperative --output data/training/ablation_cooperative/

    # Single TR tier with fewer workers
    python -m experiments.campaign baseline --mode tr3 --max-workers 16

    # Dry run to see the experiment matrix
    python -m experiments.campaign baseline --dry-run

The underlying orchestrator preserves every dynamic resource management
feature of the original: GPU memory tracking with bin-packing allocation,
round-robin GPU distribution, CPU-only pool for heuristic algorithms,
memory-aware queue sorting, active load balancing via work-stealing,
backpressure on memory/thermal pressure, and safe worker limits based on
hardware detection.
"""

import os
import sys
import json
import time
import random
import signal
import logging
import argparse
import traceback
import subprocess
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from threading import Lock, Thread, Event
from collections import defaultdict
from queue import Queue

import numpy as np


# ============================================================================
# CUSTOM JSON ENCODER FOR NUMPY TYPES
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


# ============================================================================
# PATH SETUP
# ============================================================================

def _setup_path():
    """Ensure ``coopetition_gym`` and related modules are importable.

    The repository layout has a top-level folder ``coopetition_gym/`` with
    the actual package at ``coopetition_gym/coopetition_gym/``. When running
    from the repository root (or a multiprocessing worker launched from
    there), Python resolves ``coopetition_gym`` to the outer folder as a
    namespace package, shadowing the installed editable package. Prepending
    the inner-package parent to ``sys.path`` and dropping any stale
    namespace-package import lets the import machinery resolve the installed
    package correctly.
    """
    experiments_dir = Path(__file__).resolve().parent  # .../experiments
    repo_root = experiments_dir.parent                 # repository root
    coopetition_gym_parent = repo_root / "coopetition_gym"

    # Drop any stale namespace-package import so the next import re-resolves.
    sys.modules.pop("coopetition_gym", None)

    # Prepend the inner-package parent; order matters so it takes precedence
    # over the namespace-package shadowing from the repo root entry.
    if str(coopetition_gym_parent) not in sys.path:
        sys.path.insert(0, str(coopetition_gym_parent))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(experiments_dir) not in sys.path:
        sys.path.insert(0, str(experiments_dir))


_setup_path()


# ============================================================================
# ENVIRONMENT DEFINITIONS BY TECHNICAL REPORT
# ============================================================================

# TR-1: Interdependence and Complementarity (arXiv:2510.18802)
# Environments focusing on value functions, synergy, and strategic interdependence
# From docs/environments/index.md: PartnerHoldUp, PlatformEcosystem, DynamicPartnerSelection, SynergySearch, RenaultNissan
TR1_ENVIRONMENTS = [
    {"id": "PartnerHoldUp-v0", "horizon": 100, "category": "dyadic", "n_agents": 2, "tr": "tr1"},
    {"id": "PlatformEcosystem-v0", "horizon": 100, "category": "ecosystem", "n_agents": 5, "tr": "tr1"},
    {"id": "DynamicPartnerSelection-v0", "horizon": 100, "category": "ecosystem", "n_agents": 4, "tr": "tr1"},
    {"id": "SynergySearch-v0", "horizon": 100, "category": "benchmark", "n_agents": 2, "tr": "tr1"},
    {"id": "RenaultNissan-v0", "horizon": 60, "category": "validated", "n_agents": 2, "tr": "tr1"},
]

# TR-2: Trust and Reputation Dynamics (arXiv:2510.24909)
# Environments focusing on trust evolution, reputation, and information asymmetry
# From docs/environments/index.md: TrustDilemma, RecoveryRace, SLCD, CooperativeNegotiation, ReputationMarket
TR2_ENVIRONMENTS = [
    {"id": "TrustDilemma-v0", "horizon": 100, "category": "dyadic", "n_agents": 2, "tr": "tr2"},
    {"id": "RecoveryRace-v0", "horizon": 150, "category": "benchmark", "n_agents": 2, "tr": "tr2"},
    {"id": "SLCD-v0", "horizon": 40, "category": "validated", "n_agents": 2, "tr": "tr2"},
    {"id": "CooperativeNegotiation-v0", "horizon": 100, "category": "extended", "n_agents": 2, "tr": "tr2"},
    {"id": "ReputationMarket-v0", "horizon": 100, "category": "extended", "n_agents": 2, "tr": "tr2"},
]

# TR-3: Collective Action and Loyalty (arXiv:2601.16237)
# Environments focusing on team production, free-rider dynamics, and coalition formation
TR3_ENVIRONMENTS = [
    {"id": "TeamProduction-v0", "horizon": 100, "category": "collective_action", "n_agents": 4, "tr": "tr3"},
    {"id": "LoyaltyTeam-v0", "horizon": 100, "category": "collective_action", "n_agents": 4, "tr": "tr3"},
    {"id": "CoalitionFormation-v0", "horizon": 150, "category": "collective_action", "n_agents": 6, "tr": "tr3"},
    {"id": "ApacheProject-v0", "horizon": 60, "category": "collective_action", "n_agents": 5, "tr": "tr3"},
    {"id": "PublicGoods-v0", "horizon": 100, "category": "collective_action", "n_agents": 4, "tr": "tr3"},
]

# TR-4: Sequential Interaction and Reciprocity (forthcoming)
# Environments focusing on conditional cooperation, memory-windowed reciprocity, bounded responses
TR4_ENVIRONMENTS = [
    {"id": "ReciprocalDilemma-v0", "horizon": 100, "category": "dyadic", "n_agents": 2, "tr": "tr4"},
    {"id": "GiftExchange-v0", "horizon": 100, "category": "dyadic", "n_agents": 2, "tr": "tr4"},
    {"id": "IndirectReciprocity-v0", "horizon": 150, "category": "reciprocity", "n_agents": 4, "tr": "tr4"},
    {"id": "GraduatedSanction-v0", "horizon": 200, "category": "reciprocity", "n_agents": 6, "tr": "tr4"},
    {"id": "AppleAppStore-v0", "horizon": 66, "category": "reciprocity", "n_agents": 3, "tr": "tr4"},
]

# Environment lookup by mode
ENVIRONMENTS_BY_MODE = {
    "tr1": TR1_ENVIRONMENTS,
    "tr2": TR2_ENVIRONMENTS,
    "tr3": TR3_ENVIRONMENTS,
    "tr4": TR4_ENVIRONMENTS,
}


# ============================================================================
# ALGORITHM DEFINITIONS WITH RESOURCE REQUIREMENTS
# ============================================================================

HEURISTIC_ALGORITHMS = [
    {"name": "Random", "class": "RandomPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {}},
    # Constant cooperation policies - comprehensive 0-100% sweep in 1% increments
    # Enables fine-grained analysis of monotonicity and nonlinearity of cooperation-return relationship
    {"name": "Constant_00", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.00}},
    {"name": "Constant_01", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.01}},
    {"name": "Constant_02", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.02}},
    {"name": "Constant_03", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.03}},
    {"name": "Constant_04", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.04}},
    {"name": "Constant_05", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.05}},
    {"name": "Constant_06", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.06}},
    {"name": "Constant_07", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.07}},
    {"name": "Constant_08", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.08}},
    {"name": "Constant_09", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.09}},
    {"name": "Constant_10", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.10}},
    {"name": "Constant_11", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.11}},
    {"name": "Constant_12", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.12}},
    {"name": "Constant_13", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.13}},
    {"name": "Constant_14", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.14}},
    {"name": "Constant_15", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.15}},
    {"name": "Constant_16", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.16}},
    {"name": "Constant_17", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.17}},
    {"name": "Constant_18", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.18}},
    {"name": "Constant_19", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.19}},
    {"name": "Constant_20", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.20}},
    {"name": "Constant_21", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.21}},
    {"name": "Constant_22", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.22}},
    {"name": "Constant_23", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.23}},
    {"name": "Constant_24", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.24}},
    {"name": "Constant_25", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.25}},
    {"name": "Constant_26", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.26}},
    {"name": "Constant_27", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.27}},
    {"name": "Constant_28", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.28}},
    {"name": "Constant_29", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.29}},
    {"name": "Constant_30", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.30}},
    {"name": "Constant_31", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.31}},
    {"name": "Constant_32", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.32}},
    {"name": "Constant_33", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.33}},
    {"name": "Constant_34", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.34}},
    {"name": "Constant_35", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.35}},
    {"name": "Constant_36", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.36}},
    {"name": "Constant_37", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.37}},
    {"name": "Constant_38", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.38}},
    {"name": "Constant_39", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.39}},
    {"name": "Constant_40", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.40}},
    {"name": "Constant_41", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.41}},
    {"name": "Constant_42", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.42}},
    {"name": "Constant_43", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.43}},
    {"name": "Constant_44", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.44}},
    {"name": "Constant_45", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.45}},
    {"name": "Constant_46", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.46}},
    {"name": "Constant_47", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.47}},
    {"name": "Constant_48", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.48}},
    {"name": "Constant_49", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.49}},
    {"name": "Constant_50", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.50}},
    {"name": "Constant_51", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.51}},
    {"name": "Constant_52", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.52}},
    {"name": "Constant_53", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.53}},
    {"name": "Constant_54", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.54}},
    {"name": "Constant_55", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.55}},
    {"name": "Constant_56", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.56}},
    {"name": "Constant_57", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.57}},
    {"name": "Constant_58", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.58}},
    {"name": "Constant_59", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.59}},
    {"name": "Constant_60", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.60}},
    {"name": "Constant_61", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.61}},
    {"name": "Constant_62", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.62}},
    {"name": "Constant_63", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.63}},
    {"name": "Constant_64", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.64}},
    {"name": "Constant_65", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.65}},
    {"name": "Constant_66", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.66}},
    {"name": "Constant_67", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.67}},
    {"name": "Constant_68", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.68}},
    {"name": "Constant_69", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.69}},
    {"name": "Constant_70", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.70}},
    {"name": "Constant_71", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.71}},
    {"name": "Constant_72", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.72}},
    {"name": "Constant_73", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.73}},
    {"name": "Constant_74", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.74}},
    {"name": "Constant_75", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.75}},
    {"name": "Constant_76", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.76}},
    {"name": "Constant_77", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.77}},
    {"name": "Constant_78", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.78}},
    {"name": "Constant_79", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.79}},
    {"name": "Constant_80", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.80}},
    {"name": "Constant_81", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.81}},
    {"name": "Constant_82", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.82}},
    {"name": "Constant_83", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.83}},
    {"name": "Constant_84", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.84}},
    {"name": "Constant_85", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.85}},
    {"name": "Constant_86", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.86}},
    {"name": "Constant_87", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.87}},
    {"name": "Constant_88", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.88}},
    {"name": "Constant_89", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.89}},
    {"name": "Constant_90", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.90}},
    {"name": "Constant_91", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.91}},
    {"name": "Constant_92", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.92}},
    {"name": "Constant_93", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.93}},
    {"name": "Constant_94", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.94}},
    {"name": "Constant_95", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.95}},
    {"name": "Constant_96", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.96}},
    {"name": "Constant_97", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.97}},
    {"name": "Constant_98", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.98}},
    {"name": "Constant_99", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 0.99}},
    {"name": "Constant_100", "class": "ConstantPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {"level": 1.00}},
    {"name": "TitForTat", "class": "TitForTatPolicy", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {}},
    # Oracle baselines - TR-specific equilibrium computations
    # TR-1: Static Coopetitive Equilibrium (interdependence only, no trust dynamics)
    {"name": "Oracle_Equilibrium", "class": "CoopetitiveEquilibriumOracle", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {},
     "applicable_trs": ["tr1"]},
    # TR-2: Trust-Aware Equilibrium (accounts for TR-2 trust dynamics)
    {"name": "Oracle_TrustAware", "class": "TrustAwareEquilibriumOracle", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {},
     "applicable_trs": ["tr2"]},
    # TR-3: Team production equilibria
    {"name": "Oracle_Nash", "class": "NashEquilibriumOracle", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {},
     "applicable_trs": ["tr3"]},
    {"name": "Oracle_SocialOptimum", "class": "SocialOptimumOracle", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {},
     "applicable_trs": ["tr3"]},
    {"name": "Oracle_Loyalty", "class": "LoyaltyAugmentedOracle", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {},
     "applicable_trs": ["tr3"]},
    # TR-4: Reciprocity equilibria
    {"name": "Oracle_ReciprocityEquilibrium", "class": "ReciprocityEquilibriumOracle", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {},
     "applicable_trs": ["tr4"]},
    {"name": "Oracle_BoundedReciprocity", "class": "BoundedReciprocityOracle", "requires_training": False,
     "gpu_memory_gb": 0.0, "cpu_only": True, "params": {},
     "applicable_trs": ["tr4"]},
]

TRAINING_ALGORITHMS = [
    # =========================================================================
    # CPU-ONLY ALGORITHMS (MLP policies - faster on CPU per SB3 recommendation)
    # See: https://github.com/DLR-RM/stable-baselines3/issues/1245
    # =========================================================================

    # Independent Learning - MLP policies run faster on CPU
    {"name": "IPPO", "class": "IndependentPPO", "requires_training": True,
     "gpu_memory_gb": 0.0, "cpu_only": True, "speed": "fast",
     "params": {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
                "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
                "vf_coef": 0.5, "max_grad_norm": 0.5, "net_arch": [128, 128]}},
    {"name": "IA2C", "class": "IndependentA2C", "requires_training": True,
     "gpu_memory_gb": 0.0, "cpu_only": True, "speed": "fast",
     "params": {"learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0,
                "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5, "net_arch": [128, 128]}},

    # CTDE MLP-based - CPU for efficiency
    {"name": "MAPPO", "class": "MAPPO", "requires_training": True,
     "gpu_memory_gb": 0.0, "cpu_only": True, "speed": "medium",
     "params": {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
                "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
                "share_critic": True, "net_arch": [128, 128]}},

    # Value Decomposition MLP-based - GPU for counterfactual baseline computation
    {"name": "COMA", "class": "COMA", "requires_training": True,
     "gpu_memory_gb": 1.5, "cpu_only": False, "speed": "fast",
     "params": {"learning_rate": 5e-4, "gamma": 0.99}},

    # Independent Policy Gradient - CPU for efficiency
    {"name": "IndependentREINFORCE", "class": "IndependentREINFORCE", "requires_training": True,
     "gpu_memory_gb": 0.0, "cpu_only": True, "speed": "fast",
     "params": {"learning_rate": 1e-3, "gamma": 0.99, "net_arch": [128, 128]}},

    # Opponent Modeling MLP-based - CPU for efficiency
    {"name": "LOLA", "class": "LOLA", "requires_training": True,
     "gpu_memory_gb": 0.0, "cpu_only": True, "speed": "medium",
     "params": {"learning_rate": 1e-3, "opponent_lr": 1e-3, "n_lookahead": 1,
                "gamma": 0.99, "net_arch": [128, 128]}},

    # Population MLP-based - CPU for efficiency
    {"name": "SelfPlay_PPO", "class": "SelfPlayPPO", "requires_training": True,
     "gpu_memory_gb": 0.0, "cpu_only": True, "speed": "slow",
     "params": {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,
                "gamma": 0.99, "opponent_update_freq": 10000, "net_arch": [128, 128]}},
    {"name": "FCP", "class": "FictitiousCoPlay", "requires_training": True,
     "gpu_memory_gb": 0.0, "cpu_only": True, "speed": "slow",
     "params": {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,
                "gamma": 0.99, "checkpoint_freq": 50000, "sample_recent_prob": 0.5,
                "net_arch": [128, 128]}},

    # Mean Field MLP-based - CPU for efficiency
    # Restricted to 3+ agent environments: mean-field approximation (Yang et al. 2018)
    # replaces joint opponent effect with population average, which is degenerate for N=2.
    {"name": "MeanFieldAC", "class": "MeanFieldActorCritic", "requires_training": True,
     "gpu_memory_gb": 0.0, "cpu_only": True, "speed": "slow",
     "applicable_categories": ["ecosystem", "collective_action", "reciprocity"],
     "params": {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "net_arch": [128, 128]}},

    # =========================================================================
    # GPU ALGORITHMS (Large replay buffers benefit from GPU memory)
    # These use off-policy learning with large experience replay buffers
    # =========================================================================

    # Independent Learning with replay buffer - GPU beneficial
    {"name": "ISAC", "class": "IndependentSAC", "requires_training": True,
     "gpu_memory_gb": 3.0, "cpu_only": False, "speed": "medium",
     "params": {"learning_rate": 3e-4, "buffer_size": 100000, "batch_size": 256,
                "tau": 0.005, "gamma": 0.99, "net_arch": [128, 128]}},

    # CTDE with replay buffers - GPU beneficial for buffer storage
    {"name": "MADDPG", "class": "MADDPG", "requires_training": True,
     "gpu_memory_gb": 4.0, "cpu_only": False, "speed": "slow",
     "params": {"learning_rate_actor": 1e-4, "learning_rate_critic": 1e-3,
                "buffer_size": 100000, "batch_size": 256, "tau": 0.005, "gamma": 0.99,
                "net_arch": [128, 128]}},
    {"name": "MATD3", "class": "MATD3", "requires_training": True,
     "gpu_memory_gb": 4.0, "cpu_only": False, "speed": "slow",
     "params": {"learning_rate_actor": 1e-4, "learning_rate_critic": 1e-3,
                "buffer_size": 100000, "batch_size": 256, "tau": 0.005, "gamma": 0.99,
                "policy_noise": 0.2, "noise_clip": 0.5, "policy_delay": 2, "net_arch": [128, 128]}},
    {"name": "MASAC", "class": "MASAC", "requires_training": True,
     "gpu_memory_gb": 4.0, "cpu_only": False, "speed": "slow",
     "params": {"learning_rate": 3e-4, "buffer_size": 100000, "batch_size": 256,
                "tau": 0.005, "gamma": 0.99, "net_arch": [128, 128]}},

    # Value Decomposition with replay buffers - GPU beneficial
    {"name": "QMIX", "class": "QMIX", "requires_training": True,
     "gpu_memory_gb": 2.5, "cpu_only": False, "speed": "medium",
     "params": {"learning_rate": 5e-4, "buffer_size": 5000, "batch_size": 32,
                "gamma": 0.99, "action_bins": 11}},
    {"name": "VDN", "class": "VDN", "requires_training": True,
     "gpu_memory_gb": 2.0, "cpu_only": False, "speed": "fast",
     "params": {"learning_rate": 5e-4, "buffer_size": 5000, "batch_size": 32,
                "gamma": 0.99, "action_bins": 11}},

    # Opponent Modeling with replay buffer - GPU beneficial
    {"name": "M3DDPG", "class": "M3DDPG", "requires_training": True,
     "gpu_memory_gb": 4.0, "cpu_only": False, "speed": "slow",
     "params": {"learning_rate_actor": 1e-4, "learning_rate_critic": 1e-3,
                "buffer_size": 100000, "batch_size": 256, "gamma": 0.99,
                "minimax_weight": 0.5, "net_arch": [128, 128]}},
]

ALL_ALGORITHMS = HEURISTIC_ALGORITHMS + TRAINING_ALGORITHMS


# ============================================================================
# RESOURCE CONFIGURATION
# ============================================================================

# Training timesteps by environment category
# All categories normalized to ~250K steps per agent for consistency
TIMESTEPS_BY_CATEGORY = {
    'dyadic': 500000,          # 2 agents → 250K/agent
    'ecosystem': 1000000,      # 4-5 agents → 200-250K/agent
    'benchmark': 500000,       # 2 agents → 250K/agent
    'validated': 500000,       # 2 agents → 250K/agent
    'extended': 500000,        # 2 agents → 250K/agent
    'collective_action': 1000000,  # 4-6 agents → 167-250K/agent (increased from 750K)
    'reciprocity': 1000000,        # 3-6 agents → 167-333K/agent (TR-4 multi-agent)
}

# GPU memory safety margin (GB) - never use more than this per GPU
GPU_MEMORY_SAFETY_MARGIN_GB = 4.0  # Reserve 4GB per GPU for system/overhead

# Lambda Cloud 8x A100 configuration
# OPTIMIZED: More CPU workers for MLP-based algorithms (IPPO, MAPPO, etc.)
# GPU reserved only for replay-buffer algorithms (MADDPG, MATD3, etc.)
LAMBDA_CLOUD_CONFIG = {
    "max_workers": 80,  # Increased for better parallelization
    "gpu_workers": 24,  # 8 GPUs × 3 per GPU for replay-buffer algorithms
    "cpu_workers": 56,  # Increased: 124 vCPUs - 8 system - 60 usable for MLP algos
    "vram_per_gpu_gb": 40,
    "num_gpus": 8,
    "vcpus": 124,
    "ram_gb": 1800,
}

# Backpressure configuration
BACKPRESSURE_CONFIG = {
    "memory_threshold_pct": 85,  # Trigger backpressure above this
    "critical_threshold_pct": 95,  # Emergency stop threshold
    "cooldown_seconds": 30,  # Wait time after reducing workers
    "reduction_factor": 0.75,  # Reduce workers by 25% on OOM
    "min_workers": 4,  # Never go below this
}

# Thermal monitoring configuration
THERMAL_CONFIG = {
    "warning_threshold_c": 75,  # Log warning above this
    "throttle_threshold_c": 80,  # Trigger backpressure above this
    "critical_threshold_c": 85,  # Emergency stop above this
}

# System RAM configuration
RAM_CONFIG = {
    "warning_threshold_pct": 70,  # Log warning above this
    "backpressure_threshold_pct": 80,  # Trigger backpressure above this
    "critical_threshold_pct": 90,  # Emergency stop above this
}

# Algorithm time estimates (minutes) for priority scheduling
# UPDATED: MLP algorithms (IPPO, MAPPO, etc.) are faster on CPU than GPU
ALGO_TIME_ESTIMATES = {
    # Heuristics (very fast, CPU-only, evaluation only)
    "Random": 1, "TitForTat": 1,
    # Constant policies - comprehensive 0-100% sweep in 1% increments (101 variants)
    "Constant_00": 1, "Constant_01": 1, "Constant_02": 1, "Constant_03": 1, "Constant_04": 1,
    "Constant_05": 1, "Constant_06": 1, "Constant_07": 1, "Constant_08": 1, "Constant_09": 1,
    "Constant_10": 1, "Constant_11": 1, "Constant_12": 1, "Constant_13": 1, "Constant_14": 1,
    "Constant_15": 1, "Constant_16": 1, "Constant_17": 1, "Constant_18": 1, "Constant_19": 1,
    "Constant_20": 1, "Constant_21": 1, "Constant_22": 1, "Constant_23": 1, "Constant_24": 1,
    "Constant_25": 1, "Constant_26": 1, "Constant_27": 1, "Constant_28": 1, "Constant_29": 1,
    "Constant_30": 1, "Constant_31": 1, "Constant_32": 1, "Constant_33": 1, "Constant_34": 1,
    "Constant_35": 1, "Constant_36": 1, "Constant_37": 1, "Constant_38": 1, "Constant_39": 1,
    "Constant_40": 1, "Constant_41": 1, "Constant_42": 1, "Constant_43": 1, "Constant_44": 1,
    "Constant_45": 1, "Constant_46": 1, "Constant_47": 1, "Constant_48": 1, "Constant_49": 1,
    "Constant_50": 1, "Constant_51": 1, "Constant_52": 1, "Constant_53": 1, "Constant_54": 1,
    "Constant_55": 1, "Constant_56": 1, "Constant_57": 1, "Constant_58": 1, "Constant_59": 1,
    "Constant_60": 1, "Constant_61": 1, "Constant_62": 1, "Constant_63": 1, "Constant_64": 1,
    "Constant_65": 1, "Constant_66": 1, "Constant_67": 1, "Constant_68": 1, "Constant_69": 1,
    "Constant_70": 1, "Constant_71": 1, "Constant_72": 1, "Constant_73": 1, "Constant_74": 1,
    "Constant_75": 1, "Constant_76": 1, "Constant_77": 1, "Constant_78": 1, "Constant_79": 1,
    "Constant_80": 1, "Constant_81": 1, "Constant_82": 1, "Constant_83": 1, "Constant_84": 1,
    "Constant_85": 1, "Constant_86": 1, "Constant_87": 1, "Constant_88": 1, "Constant_89": 1,
    "Constant_90": 1, "Constant_91": 1, "Constant_92": 1, "Constant_93": 1, "Constant_94": 1,
    "Constant_95": 1, "Constant_96": 1, "Constant_97": 1, "Constant_98": 1, "Constant_99": 1,
    "Constant_100": 1,
    # MLP-based CPU algorithms (faster than GPU for small networks)
    "IPPO": 10, "IA2C": 8, "MAPPO": 20, "IndependentREINFORCE": 10,
    "LOLA": 15, "SelfPlay_PPO": 40, "FCP": 40, "MeanFieldAC": 35,
    # Replay-buffer and counterfactual GPU algorithms (benefit from GPU memory/compute)
    "ISAC": 25, "MADDPG": 35, "MATD3": 35, "MASAC": 35, "M3DDPG": 35,
    "QMIX": 20, "VDN": 15, "COMA": 12,
    # Oracle algorithms (heuristic, CPU-only, evaluation only)
    "Oracle_Equilibrium": 1, "Oracle_TrustAware": 1, "Oracle_Nash": 1,
    "Oracle_SocialOptimum": 1, "Oracle_Loyalty": 1,
    "Oracle_ReciprocityEquilibrium": 1, "Oracle_BoundedReciprocity": 1,
}

# Reduced buffer parameters for memory pressure levels
REDUCED_BUFFER_PARAMS = {
    # Level 1: Moderate reduction (75% of original)
    1: {
        "ISAC": {"buffer_size": 75000},
        "MADDPG": {"buffer_size": 75000},
        "MATD3": {"buffer_size": 75000},
        "MASAC": {"buffer_size": 75000},
        "M3DDPG": {"buffer_size": 75000},
        "QMIX": {"buffer_size": 3750},
        "VDN": {"buffer_size": 3750},
    },
    # Level 2: Aggressive reduction (50% of original)
    2: {
        "ISAC": {"buffer_size": 50000, "batch_size": 128},
        "MADDPG": {"buffer_size": 50000, "batch_size": 128},
        "MATD3": {"buffer_size": 50000, "batch_size": 128},
        "MASAC": {"buffer_size": 50000, "batch_size": 128},
        "M3DDPG": {"buffer_size": 50000, "batch_size": 128},
        "QMIX": {"buffer_size": 2500, "batch_size": 16},
        "VDN": {"buffer_size": 2500, "batch_size": 16},
    },
}

# Checkpoint configuration
CHECKPOINT_CONFIG = {
    "interval_steps": 100000,  # Save checkpoint every N steps
    "max_checkpoints": 3,  # Keep only last N checkpoints per experiment
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ExperimentResult:
    """Container for experiment results - unified format for all TRs."""
    algorithm: str
    environment: str
    training_seed: int
    status: str
    error_message: Optional[str] = None
    training_time_seconds: float = 0.0
    evaluation_time_seconds: float = 0.0
    metrics: Optional[Dict[str, Any]] = None
    timestamp: str = ""
    gpu_id: int = -1
    tr_mode: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    modes: List[str]  # List of ["tr1", "tr2", "tr3"]
    output_dir: Path
    algorithms: Optional[List[str]] = None
    environments: Optional[List[str]] = None
    seeds: List[int] = field(default_factory=lambda: [100, 101, 102, 103, 104])
    n_eval_episodes: int = 100
    resume: bool = False
    dry_run: bool = False
    shuffle_seed: int = 42

    # Worker limits (explicit overrides)
    max_workers: Optional[int] = None  # Total worker cap (GPU + CPU)
    max_gpu_workers: Optional[int] = None  # GPU worker cap
    max_cpu_workers: Optional[int] = None  # CPU worker cap

    # Monitoring options
    enable_memory_monitoring: bool = True
    memory_poll_interval: float = 5.0  # seconds
    memory_threshold_pct: float = 85.0  # Backpressure threshold
    critical_threshold_pct: float = 95.0  # Emergency threshold

    # Backpressure options
    enable_backpressure: bool = True
    backpressure_cooldown: float = 30.0  # seconds

    # System resource monitoring
    enable_ram_monitoring: bool = True
    ram_threshold_pct: float = 80.0  # Backpressure threshold
    ram_critical_pct: float = 90.0  # Emergency threshold

    # Thermal monitoring
    enable_thermal_monitoring: bool = True
    thermal_throttle_c: float = 80.0  # Backpressure threshold
    thermal_critical_c: float = 85.0  # Emergency threshold

    # Adaptive buffer sizes
    enable_adaptive_buffers: bool = True

    # Priority scheduling — fast-first reduces wall-clock time by completing
    # quick experiments early, freeing GPU slots for slow algorithms sooner
    enable_priority_queue: bool = True
    prioritize_fast_algorithms: bool = True  # Fast algos first to avoid convoy effect

    # NVLink-aware scheduling
    enable_nvlink_scheduling: bool = True

    # Checkpoint recovery
    enable_checkpoints: bool = True  # Enabled by default for crash recovery
    checkpoint_interval: int = 100000  # Steps between checkpoints
    checkpoint_dir: Optional[Path] = None  # Directory for checkpoints

    # GPU isolation
    enable_gpu_isolation: bool = True  # Use CUDA_VISIBLE_DEVICES


# ============================================================================
# HARDWARE DETECTION
# ============================================================================

def detect_hardware() -> Dict[str, Any]:
    """Detect available hardware resources with detailed GPU info."""
    hardware = {
        "device": "cpu",
        "num_gpus": 0,
        "num_vcpus": mp.cpu_count(),
        "gpu_names": [],
        "gpu_memory_gb": [],
        "total_gpu_memory_gb": 0.0,
        "usable_gpu_memory_gb": [],  # After safety margin
    }

    try:
        import torch
        if torch.cuda.is_available():
            hardware["device"] = "cuda"
            hardware["num_gpus"] = torch.cuda.device_count()
            total = 0.0
            for i in range(hardware["num_gpus"]):
                hardware["gpu_names"].append(torch.cuda.get_device_name(i))
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / (1024**3)
                hardware["gpu_memory_gb"].append(mem_gb)
                usable = mem_gb - GPU_MEMORY_SAFETY_MARGIN_GB
                hardware["usable_gpu_memory_gb"].append(usable)
                total += mem_gb
            hardware["total_gpu_memory_gb"] = total
    except ImportError:
        pass

    return hardware


def compute_safe_worker_limits(
    hardware: Dict[str, Any],
    max_workers: Optional[int] = None,
    max_gpu_workers: Optional[int] = None,
    max_cpu_workers: Optional[int] = None,
) -> Dict[str, int]:
    """
    Compute safe worker limits based on hardware with optional overrides.

    OPTIMIZED FOR MLP-ON-CPU PARADIGM:
    - MLP-based algorithms (IPPO, MAPPO, IA2C, etc.) run on CPU for efficiency
    - Only replay-buffer algorithms (MADDPG, MATD3, MASAC, etc.) use GPU
    - GPU workers are CPU-bound (90% time in env.step()), allowing high concurrency

    Lambda Cloud instance examples:

    A100-80GB (8 GPU, 240 vCPU, 1800GB RAM):
    - Each A100 has 80GB, usable ~76GB
    - Observed ~1.3GB per worker (CPU-bound workload)
    - Tier cap: 24 workers/GPU for 80GB-class GPUs
    - Total GPU workers = 8 * 24 = 192
    - CPU workers = 240 - 8(sys) - 192(gpu) = 40
    - Total: 192 GPU + 40 CPU = 232 concurrent experiments

    A100-40GB (8 GPU, 124 vCPU, 1800GB RAM):
    - Each A100 has 40GB, usable ~36GB
    - Tier cap: 12 workers/GPU for 40GB-class GPUs
    - Total GPU workers = 8 * 12 = 96
    - CPU workers = 124 - 8(sys) - 96(gpu) = 20
    - Total: 96 GPU + 20 CPU = 116 concurrent experiments

    Safety is ensured by runtime GPU memory monitoring + backpressure, not
    conservative static limits. The tier caps prevent OOM while maximizing throughput.

    Parameters
    ----------
    hardware : Dict
        Hardware detection results
    max_workers : int, optional
        Hard cap on total workers (GPU + CPU combined)
    max_gpu_workers : int, optional
        Hard cap on GPU workers only
    max_cpu_workers : int, optional
        Hard cap on CPU workers only

    Returns
    -------
    Dict with gpu_workers, cpu_workers, max_per_gpu, total_workers
    """
    num_gpus = hardware.get("num_gpus", 0)
    num_vcpus = hardware.get("num_vcpus", 4)
    usable_memory = hardware.get("usable_gpu_memory_gb", [])

    if num_gpus == 0:
        # CPU-only mode - allocate most vCPUs for training
        # PATCHED: Removed hardcoded cap of 64 - let max_cpu_workers override
        system_reserve = max(8, num_vcpus // 30)
        computed_cpu = num_vcpus - system_reserve

        # Apply explicit overrides (allow INCREASE beyond default)
        if max_cpu_workers is not None:
            computed_cpu = max_cpu_workers  # Override, not min()
        if max_workers is not None:
            computed_cpu = min(computed_cpu, max_workers)

        return {
            "gpu_workers": 0,
            "cpu_workers": computed_cpu,
            "max_per_gpu": 0,
            "total_workers": computed_cpu,
        }

    # Calculate safe workers per GPU based on minimum usable memory
    # REALISTIC: Observed peak ~1.3GB/worker on A100 (model params + optimizer stay
    # allocated, but workers spend 90% of time in CPU env.step()). Use 2.0GB as
    # realistic peak with safety margin instead of 4.0GB theoretical max.
    # Runtime GPU memory monitoring + backpressure provide the actual safety net.
    min_usable = min(usable_memory) if usable_memory else 36.0
    realistic_algo_memory = 2.0  # Observed 1.3GB peak, 2.0GB with safety margin
    safe_per_gpu = max(1, int(min_usable / realistic_algo_memory))

    # Dynamic cap based on GPU memory tier (runtime monitoring handles actual safety)
    # A100-80GB (76GB usable): up to 24 per GPU — workload is CPU-bound, not GPU-bound
    # A100-40GB (36GB usable): up to 12 per GPU
    # Smaller GPUs (V100, etc.): up to 6 per GPU
    if min_usable >= 72:
        max_per_gpu_cap = 24
    elif min_usable >= 32:
        max_per_gpu_cap = 12
    else:
        max_per_gpu_cap = 6
    safe_per_gpu = min(safe_per_gpu, max_per_gpu_cap)

    gpu_workers = num_gpus * safe_per_gpu

    # CPU workers: account for GPU workers consuming vCPUs
    # env.step() is single-threaded so effective vCPU usage ~1.0 per worker process.
    # OMP_NUM_THREADS=2 adds brief torch bursts but scheduler handles this.
    system_reserve = max(8, num_vcpus // 30)  # At least 8, scales with instance size
    vcpus_used_by_gpu = gpu_workers  # ~1 vCPU per GPU worker process
    available_for_cpu = max(0, num_vcpus - system_reserve - vcpus_used_by_gpu)
    cpu_workers = max(4, available_for_cpu)

    # Apply explicit overrides (allow BOTH increase and decrease)
    if max_gpu_workers is not None:
        gpu_workers = max_gpu_workers
        safe_per_gpu = max(1, max_gpu_workers // max(num_gpus, 1))
        # Recompute CPU workers to account for changed GPU worker count
        vcpus_used_by_gpu = gpu_workers
        available_for_cpu = max(0, num_vcpus - system_reserve - vcpus_used_by_gpu)
        cpu_workers = max(4, available_for_cpu)

    if max_cpu_workers is not None:
        # PATCHED: Allow override to INCREASE cpu_workers, not just decrease
        cpu_workers = max_cpu_workers

    # Apply total max_workers constraint
    if max_workers is not None:
        total = gpu_workers + cpu_workers
        if total > max_workers:
            # Reduce proportionally, but prioritize GPU workers
            ratio = max_workers / total
            new_gpu = max(1, int(gpu_workers * ratio))
            new_cpu = max_workers - new_gpu
            gpu_workers = new_gpu
            cpu_workers = max(1, new_cpu)
            safe_per_gpu = max(1, gpu_workers // max(num_gpus, 1))

    return {
        "gpu_workers": gpu_workers,
        "cpu_workers": cpu_workers,
        "max_per_gpu": safe_per_gpu,
        "total_workers": gpu_workers + cpu_workers,
    }


# ============================================================================
# GPU MEMORY MANAGER (DYNAMIC)
# ============================================================================

class GPUMemoryManager:
    """
    Dynamic GPU memory manager with per-GPU tracking.

    Implements:
    - Real-time memory tracking per GPU
    - Round-robin allocation with memory checking (distributes load evenly)
    - Automatic load balancing
    """

    def __init__(self, num_gpus: int, memory_per_gpu: List[float]):
        self.num_gpus = num_gpus
        self.total_memory = memory_per_gpu.copy()  # Total GB per GPU
        self.used_memory = [0.0] * num_gpus  # Currently allocated GB
        self.next_gpu = 0  # Round-robin counter for fair distribution
        self.active_jobs = [0] * num_gpus  # Number of active jobs per GPU
        self.lock = Lock()

    def allocate(self, algo_name: str, required_gb: float) -> int:
        """
        Allocate GPU for an algorithm using round-robin distribution.

        Round-robin ensures all GPUs are used evenly, preventing scenarios where
        GPUs 0-4 get all jobs while GPUs 5-7 sit idle (which happened with
        best-fit bin-packing on Lambda Cloud 8x A100).

        Returns GPU ID or -1 if no GPU has enough memory.
        """
        with self.lock:
            # Try round-robin starting from next_gpu, wrapping around
            for i in range(self.num_gpus):
                gpu_id = (self.next_gpu + i) % self.num_gpus
                available = self.total_memory[gpu_id] - self.used_memory[gpu_id]

                if available >= required_gb:
                    # Found a GPU with enough memory
                    self.used_memory[gpu_id] += required_gb
                    self.active_jobs[gpu_id] += 1
                    # Advance round-robin counter for next allocation
                    self.next_gpu = (gpu_id + 1) % self.num_gpus
                    return gpu_id

            # No GPU has enough memory
            return -1

    def release(self, gpu_id: int, required_gb: float):
        """Release GPU memory after job completion."""
        with self.lock:
            if gpu_id >= 0 and gpu_id < self.num_gpus:
                self.used_memory[gpu_id] = max(0, self.used_memory[gpu_id] - required_gb)
                self.active_jobs[gpu_id] = max(0, self.active_jobs[gpu_id] - 1)

    def get_status(self) -> Dict[str, Any]:
        """Get current GPU memory status."""
        with self.lock:
            return {
                "used_memory_gb": self.used_memory.copy(),
                "total_memory_gb": self.total_memory.copy(),
                "active_jobs": self.active_jobs.copy(),
                "utilization_pct": [
                    (u / t * 100) if t > 0 else 0
                    for u, t in zip(self.used_memory, self.total_memory)
                ],
            }

    def get_least_loaded_gpu(self) -> int:
        """Get GPU with most available memory (for load balancing)."""
        with self.lock:
            best_gpu = 0
            best_available = 0
            for gpu_id in range(self.num_gpus):
                available = self.total_memory[gpu_id] - self.used_memory[gpu_id]
                if available > best_available:
                    best_available = available
                    best_gpu = gpu_id
            return best_gpu

    def reconcile_with_actual(self, actual_used_gb: List[float], logger=None):
        """Reconcile tracked allocations with actual GPU memory from nvidia-smi.

        If tracked used_memory diverges from actual by more than 2 GB per GPU
        (indicating crashed experiments that didn't call release()), reset the
        tracker to match reality. This prevents phantom allocations from blocking
        future experiment submissions.
        """
        with self.lock:
            for gpu_id in range(min(self.num_gpus, len(actual_used_gb))):
                tracked = self.used_memory[gpu_id]
                actual = actual_used_gb[gpu_id]
                divergence = tracked - actual
                # If tracked exceeds actual by >2 GB, a release() was missed
                if divergence > 2.0:
                    if logger:
                        logger.info(
                            f"GPU {gpu_id} memory reconciled: tracked={tracked:.1f}GB "
                            f"actual={actual:.1f}GB (freed {divergence:.1f}GB phantom)"
                        )
                    self.used_memory[gpu_id] = actual


# ============================================================================
# GPU MEMORY MONITOR (RUNTIME via nvidia-smi)
# ============================================================================

class GPUMemoryMonitor:
    """
    Runtime GPU memory monitor using nvidia-smi.

    Provides:
    - Real-time memory usage via nvidia-smi queries
    - Background monitoring thread with periodic logging
    - Backpressure signals when memory exceeds thresholds
    - OOM detection and recovery coordination
    """

    def __init__(
        self,
        num_gpus: int,
        poll_interval: float = 5.0,
        memory_threshold_pct: float = 85.0,
        critical_threshold_pct: float = 95.0,
        thermal_throttle_c: float = 80.0,
        thermal_critical_c: float = 85.0,
        enable_thermal: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.num_gpus = num_gpus
        self.poll_interval = poll_interval
        self.memory_threshold_pct = memory_threshold_pct
        self.critical_threshold_pct = critical_threshold_pct
        self.thermal_throttle_c = thermal_throttle_c
        self.thermal_critical_c = thermal_critical_c
        self.enable_thermal = enable_thermal
        self.logger = logger or logging.getLogger(__name__)

        # State
        self.lock = Lock()
        self.running = False
        self.monitor_thread: Optional[Thread] = None
        self.stop_event = Event()

        # Memory readings (per GPU)
        self.current_used_mb: List[float] = [0.0] * num_gpus
        self.current_total_mb: List[float] = [0.0] * num_gpus
        self.current_utilization_pct: List[float] = [0.0] * num_gpus

        # Temperature readings (per GPU)
        self.current_temp_c: List[float] = [0.0] * num_gpus
        self.thermal_throttle_active = False
        self.thermal_critical_state = False

        # Backpressure state
        self.backpressure_active = False
        self.critical_state = False
        self.oom_count = 0
        self.last_oom_time: Optional[float] = None

        # Callbacks for backpressure
        self.backpressure_callback: Optional[callable] = None
        self.critical_callback: Optional[callable] = None
        self.thermal_callback: Optional[callable] = None
        self.thermal_critical_callback: Optional[callable] = None  # Separate from memory critical!

        # History for trend analysis
        self.memory_history: List[Dict[str, Any]] = []
        self.max_history_size = 100

    def _query_nvidia_smi(self) -> Optional[List[Dict[str, float]]]:
        """Query nvidia-smi for current GPU memory usage and temperature."""
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                self.logger.warning(f"nvidia-smi failed: {result.stderr}")
                return None

            gpu_data = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpu_data.append({
                        'index': int(parts[0]),
                        'used_mb': float(parts[1]),
                        'total_mb': float(parts[2]),
                        'utilization_pct': float(parts[3]),
                        'temperature_c': float(parts[4]),
                    })
                elif len(parts) >= 4:
                    # Fallback without temperature
                    gpu_data.append({
                        'index': int(parts[0]),
                        'used_mb': float(parts[1]),
                        'total_mb': float(parts[2]),
                        'utilization_pct': float(parts[3]),
                        'temperature_c': 0.0,
                    })

            return gpu_data

        except subprocess.TimeoutExpired:
            self.logger.warning("nvidia-smi query timed out")
            return None
        except FileNotFoundError:
            self.logger.warning("nvidia-smi not found")
            return None
        except Exception as e:
            self.logger.warning(f"nvidia-smi query error: {e}")
            return None

    def _update_readings(self, gpu_data: List[Dict[str, float]]):
        """Update internal readings from nvidia-smi data."""
        with self.lock:
            for data in gpu_data:
                idx = data['index']
                if idx < self.num_gpus:
                    self.current_used_mb[idx] = data['used_mb']
                    self.current_total_mb[idx] = data['total_mb']
                    self.current_temp_c[idx] = data.get('temperature_c', 0.0)
                    if data['total_mb'] > 0:
                        self.current_utilization_pct[idx] = (
                            data['used_mb'] / data['total_mb'] * 100
                        )
                    else:
                        self.current_utilization_pct[idx] = data['utilization_pct']

            # Record history
            self.memory_history.append({
                'timestamp': time.time(),
                'used_mb': self.current_used_mb.copy(),
                'total_mb': self.current_total_mb.copy(),
                'utilization_pct': self.current_utilization_pct.copy(),
                'temperature_c': self.current_temp_c.copy(),
            })

            # Trim history
            if len(self.memory_history) > self.max_history_size:
                self.memory_history = self.memory_history[-self.max_history_size:]

    def _check_thresholds(self):
        """Check memory and thermal thresholds and trigger callbacks."""
        with self.lock:
            max_utilization = max(self.current_utilization_pct) if self.current_utilization_pct else 0
            max_temp = max(self.current_temp_c) if self.current_temp_c else 0

            # Check critical memory threshold
            if max_utilization >= self.critical_threshold_pct:
                if not self.critical_state:
                    self.critical_state = True
                    self.logger.error(
                        f"CRITICAL: GPU memory at {max_utilization:.1f}% "
                        f"(threshold: {self.critical_threshold_pct}%)"
                    )
                    if self.critical_callback:
                        self.critical_callback(max_utilization)
            else:
                self.critical_state = False

            # Check memory backpressure threshold
            if max_utilization >= self.memory_threshold_pct:
                if not self.backpressure_active:
                    self.backpressure_active = True
                    self.logger.warning(
                        f"BACKPRESSURE: GPU memory at {max_utilization:.1f}% "
                        f"(threshold: {self.memory_threshold_pct}%)"
                    )
                    if self.backpressure_callback:
                        self.backpressure_callback(max_utilization)
            else:
                if self.backpressure_active:
                    self.logger.info(
                        f"Backpressure released: GPU memory at {max_utilization:.1f}%"
                    )
                self.backpressure_active = False

            # Check thermal thresholds (if enabled)
            if self.enable_thermal and max_temp > 0:
                # Critical thermal state
                if max_temp >= self.thermal_critical_c:
                    if not self.thermal_critical_state:
                        self.thermal_critical_state = True
                        self.logger.error(
                            f"THERMAL CRITICAL: GPU at {max_temp:.1f}°C "
                            f"(threshold: {self.thermal_critical_c}°C)"
                        )
                        # BUG FIX: Use thermal_critical_callback, NOT critical_callback!
                        # critical_callback is for memory and expects a percentage value.
                        if self.thermal_critical_callback:
                            self.thermal_critical_callback(max_temp)
                else:
                    self.thermal_critical_state = False

                # Thermal throttle threshold
                if max_temp >= self.thermal_throttle_c:
                    if not self.thermal_throttle_active:
                        self.thermal_throttle_active = True
                        self.logger.warning(
                            f"THERMAL THROTTLE: GPU at {max_temp:.1f}°C "
                            f"(threshold: {self.thermal_throttle_c}°C)"
                        )
                        if self.thermal_callback:
                            self.thermal_callback(max_temp)
                else:
                    if self.thermal_throttle_active:
                        self.logger.info(
                            f"Thermal throttle released: GPU at {max_temp:.1f}°C"
                        )
                    self.thermal_throttle_active = False

    def _monitor_loop(self):
        """Background monitoring loop."""
        self.logger.info("GPU memory monitor started")

        while not self.stop_event.is_set():
            gpu_data = self._query_nvidia_smi()

            if gpu_data:
                self._update_readings(gpu_data)
                self._check_thresholds()

            self.stop_event.wait(self.poll_interval)

        self.logger.info("GPU memory monitor stopped")

    def start(self):
        """Start the background monitor thread."""
        if self.running:
            return

        self.running = True
        self.stop_event.clear()
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop the background monitor thread."""
        if not self.running:
            return

        self.running = False
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            self.monitor_thread = None

    def get_current_status(self) -> Dict[str, Any]:
        """Get current GPU memory and thermal status."""
        with self.lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'num_gpus': self.num_gpus,
                'per_gpu': [
                    {
                        'gpu_id': i,
                        'used_mb': self.current_used_mb[i],
                        'total_mb': self.current_total_mb[i],
                        'utilization_pct': self.current_utilization_pct[i],
                        'used_gb': self.current_used_mb[i] / 1024,
                        'free_gb': (self.current_total_mb[i] - self.current_used_mb[i]) / 1024,
                        'temperature_c': self.current_temp_c[i],
                    }
                    for i in range(self.num_gpus)
                ],
                'max_utilization_pct': max(self.current_utilization_pct) if self.current_utilization_pct else 0,
                'avg_utilization_pct': np.mean(self.current_utilization_pct) if self.current_utilization_pct else 0,
                'max_temperature_c': max(self.current_temp_c) if self.current_temp_c else 0,
                'avg_temperature_c': np.mean(self.current_temp_c) if self.current_temp_c else 0,
                'backpressure_active': self.backpressure_active,
                'critical_state': self.critical_state,
                'thermal_throttle_active': self.thermal_throttle_active,
                'thermal_critical_state': self.thermal_critical_state,
                'oom_count': self.oom_count,
            }

    def set_thermal_callback(self, callback: callable):
        """Set callback for thermal throttle events."""
        self.thermal_callback = callback

    def log_status(self, prefix: str = ""):
        """Log current GPU memory and thermal status to console."""
        status = self.get_current_status()

        lines = [f"{prefix}GPU Memory & Thermal Status:"]
        for gpu in status['per_gpu']:
            temp_str = f" {gpu['temperature_c']:.0f}°C" if gpu['temperature_c'] > 0 else ""
            lines.append(
                f"  GPU {gpu['gpu_id']}: {gpu['used_gb']:.1f}/{gpu['total_mb']/1024:.1f} GB "
                f"({gpu['utilization_pct']:.1f}%){temp_str}"
            )
        lines.append(
            f"  Mem: Max={status['max_utilization_pct']:.1f}% Avg={status['avg_utilization_pct']:.1f}% | "
            f"Temp: Max={status['max_temperature_c']:.0f}°C | "
            f"BP: {'ACTIVE' if status['backpressure_active'] else 'off'} | "
            f"Thermal: {'THROTTLE' if status['thermal_throttle_active'] else 'ok'} | "
            f"OOM: {status['oom_count']}"
        )

        self.logger.info('\n'.join(lines))

    def record_oom_error(self):
        """Record an OOM error occurrence."""
        with self.lock:
            self.oom_count += 1
            self.last_oom_time = time.time()
            self.logger.error(f"OOM error recorded (total count: {self.oom_count})")

    def set_backpressure_callback(self, callback: callable):
        """Set callback for backpressure events."""
        self.backpressure_callback = callback

    def set_critical_callback(self, callback: callable):
        """Set callback for critical memory events."""
        self.critical_callback = callback

    def set_thermal_critical_callback(self, callback: callable):
        """Set callback for thermal critical events (separate from memory critical!)."""
        self.thermal_critical_callback = callback


# ============================================================================
# DYNAMIC BACKPRESSURE CONTROLLER
# ============================================================================

class BackpressureController:
    """
    Dynamic backpressure controller that adjusts worker count based on
    GPU memory pressure and OOM errors.

    Features:
    - Reduces workers when memory exceeds threshold
    - Emergency halt on critical memory state
    - Gradual recovery after pressure relief
    - OOM error tracking and response
    """

    def __init__(
        self,
        initial_gpu_workers: int,
        initial_cpu_workers: int,
        min_gpu_workers: int = 48,       # PATCHED: Was 2, now 48 (higher floor)
        min_cpu_workers: int = 8,         # PATCHED: Was 2, now 8
        reduction_factor: float = 0.75,
        recovery_factor: float = 1.5,     # PATCHED: Was 1.1, now 1.5 (faster recovery)
        cooldown_seconds: float = 30.0,
        logger: Optional[logging.Logger] = None,
    ):
        self.initial_gpu_workers = initial_gpu_workers
        self.initial_cpu_workers = initial_cpu_workers
        # PATCHED: Dynamic floor - at least 25% of initial workers
        self.min_gpu_workers = max(min_gpu_workers, initial_gpu_workers // 4)
        self.min_cpu_workers = max(min_cpu_workers, initial_cpu_workers // 4)
        self.reduction_factor = reduction_factor
        self.recovery_factor = recovery_factor
        self.cooldown_seconds = cooldown_seconds
        self.logger = logger or logging.getLogger(__name__)

        # Current worker counts
        self.current_gpu_workers = initial_gpu_workers
        self.current_cpu_workers = initial_cpu_workers

        # State
        self.lock = Lock()
        self.last_reduction_time: Optional[float] = None
        self.last_recovery_time: Optional[float] = None
        self.reduction_count = 0
        self.recovery_count = 0
        self.paused = False
        self.oom_count = 0  # PATCHED: Track OOM errors

    def reduce_workers(self, reason: str = "memory pressure") -> Tuple[int, int]:
        """
        Reduce worker count due to memory pressure.

        Returns new (gpu_workers, cpu_workers) tuple.
        """
        with self.lock:
            now = time.time()

            # Check cooldown
            if self.last_reduction_time and (now - self.last_reduction_time) < self.cooldown_seconds:
                self.logger.debug(f"Reduction cooldown active, skipping")
                return self.current_gpu_workers, self.current_cpu_workers

            # Reduce GPU workers first (more impactful on memory)
            new_gpu = max(
                self.min_gpu_workers,
                int(self.current_gpu_workers * self.reduction_factor)
            )

            # Only reduce CPU workers if GPU workers at minimum
            new_cpu = self.current_cpu_workers
            if new_gpu == self.current_gpu_workers and new_gpu == self.min_gpu_workers:
                new_cpu = max(
                    self.min_cpu_workers,
                    int(self.current_cpu_workers * self.reduction_factor)
                )

            if new_gpu < self.current_gpu_workers or new_cpu < self.current_cpu_workers:
                self.logger.warning(
                    f"BACKPRESSURE: Reducing workers due to {reason}: "
                    f"GPU {self.current_gpu_workers}→{new_gpu}, "
                    f"CPU {self.current_cpu_workers}→{new_cpu}"
                )
                self.current_gpu_workers = new_gpu
                self.current_cpu_workers = new_cpu
                self.last_reduction_time = now
                self.reduction_count += 1

            return self.current_gpu_workers, self.current_cpu_workers

    def try_recover_workers(self) -> Tuple[int, int]:
        """
        Try to recover workers if memory pressure has subsided.

        PATCHED: Uses same cooldown as reduction (was 2×), ensures minimum +1 increase.
        Returns new (gpu_workers, cpu_workers) tuple.
        """
        with self.lock:
            now = time.time()

            # PATCHED: Use same cooldown as reduction (was 2×)
            if self.last_reduction_time and (now - self.last_reduction_time) < self.cooldown_seconds:
                return self.current_gpu_workers, self.current_cpu_workers

            # Already at maximum?
            if (self.current_gpu_workers >= self.initial_gpu_workers and
                self.current_cpu_workers >= self.initial_cpu_workers):
                return self.current_gpu_workers, self.current_cpu_workers

            # PATCHED: Ensure at least +1 increase to avoid stuck recovery
            new_gpu = min(
                self.initial_gpu_workers,
                max(self.current_gpu_workers + 1, int(self.current_gpu_workers * self.recovery_factor))
            )
            new_cpu = min(
                self.initial_cpu_workers,
                max(self.current_cpu_workers + 1, int(self.current_cpu_workers * self.recovery_factor))
            )

            if new_gpu > self.current_gpu_workers or new_cpu > self.current_cpu_workers:
                self.logger.info(
                    f"RECOVERY: Increasing workers: "
                    f"GPU {self.current_gpu_workers}→{new_gpu}, "
                    f"CPU {self.current_cpu_workers}→{new_cpu}"
                )
                self.current_gpu_workers = new_gpu
                self.current_cpu_workers = new_cpu
                self.last_recovery_time = now
                self.recovery_count += 1

            return self.current_gpu_workers, self.current_cpu_workers

    def handle_oom_error(self) -> Tuple[int, int]:
        """Handle OOM error with moderate reduction (25% instead of 50%).

        PATCHED: Changed from // 2 (50% reduction) to * 0.75 (25% reduction)
        to prevent excessive worker starvation on high-memory GPUs.
        """
        with self.lock:
            self.oom_count += 1
            self.logger.error(
                f"OOM ERROR #{self.oom_count}: Reducing workers by 25% "
                f"(current GPU: {self.current_gpu_workers})"
            )

            # PATCHED: Reduce by 25% instead of 50%
            new_gpu = max(
                self.min_gpu_workers,
                int(self.current_gpu_workers * 0.75)
            )
            new_cpu = max(
                self.min_cpu_workers,
                int(self.current_cpu_workers * 0.85)
            )

            if new_gpu < self.current_gpu_workers:
                self.logger.warning(
                    f"BACKPRESSURE: GPU workers {self.current_gpu_workers} → {new_gpu}"
                )

            self.current_gpu_workers = new_gpu
            self.current_cpu_workers = new_cpu
            self.last_reduction_time = time.time()
            self.reduction_count += 1

            return self.current_gpu_workers, self.current_cpu_workers

    def pause_submissions(self):
        """Temporarily pause new job submissions."""
        with self.lock:
            self.paused = True
            self.logger.warning("Job submissions PAUSED due to memory pressure")

    def resume_submissions(self):
        """Resume job submissions."""
        with self.lock:
            self.paused = False
            self.logger.info("Job submissions RESUMED")

    def is_paused(self) -> bool:
        """Check if submissions are paused."""
        with self.lock:
            return self.paused

    def get_current_workers(self) -> Tuple[int, int]:
        """Get current worker counts."""
        with self.lock:
            return self.current_gpu_workers, self.current_cpu_workers

    def get_status(self) -> Dict[str, Any]:
        """Get current backpressure status."""
        with self.lock:
            return {
                'gpu_workers': self.current_gpu_workers,
                'cpu_workers': self.current_cpu_workers,
                'initial_gpu_workers': self.initial_gpu_workers,
                'initial_cpu_workers': self.initial_cpu_workers,
                'reduction_count': self.reduction_count,
                'recovery_count': self.recovery_count,
                'paused': self.paused,
                'at_minimum': (
                    self.current_gpu_workers == self.min_gpu_workers and
                    self.current_cpu_workers == self.min_cpu_workers
                ),
            }


# ============================================================================
# SYSTEM RESOURCE MONITOR (RAM/Disk)
# ============================================================================

class SystemResourceMonitor:
    """
    System resource monitor for RAM and disk usage.

    Provides:
    - Real-time RAM usage monitoring
    - Disk space monitoring for output directory
    - Backpressure signals when resources are low
    """

    def __init__(
        self,
        output_dir: Path,
        poll_interval: float = 10.0,
        ram_threshold_pct: float = 80.0,
        ram_critical_pct: float = 90.0,
        disk_min_gb: float = 50.0,
        logger: Optional[logging.Logger] = None,
    ):
        self.output_dir = output_dir
        self.poll_interval = poll_interval
        self.ram_threshold_pct = ram_threshold_pct
        self.ram_critical_pct = ram_critical_pct
        self.disk_min_gb = disk_min_gb
        self.logger = logger or logging.getLogger(__name__)

        # State
        self.lock = Lock()
        self.running = False
        self.monitor_thread: Optional[Thread] = None
        self.stop_event = Event()

        # Current readings
        self.current_ram_used_gb: float = 0.0
        self.current_ram_total_gb: float = 0.0
        self.current_ram_pct: float = 0.0
        self.current_disk_free_gb: float = 0.0
        self.current_disk_total_gb: float = 0.0

        # Backpressure state
        self.ram_backpressure_active = False
        self.ram_critical_state = False
        self.disk_warning_active = False

        # Callbacks
        self.ram_backpressure_callback: Optional[callable] = None
        self.ram_critical_callback: Optional[callable] = None
        self.disk_warning_callback: Optional[callable] = None

    def _query_system_resources(self) -> Dict[str, float]:
        """Query current system resource usage."""
        try:
            import psutil

            # RAM usage
            mem = psutil.virtual_memory()
            ram_total_gb = mem.total / (1024**3)
            ram_used_gb = mem.used / (1024**3)
            ram_pct = mem.percent

            # Disk usage for output directory
            disk = psutil.disk_usage(str(self.output_dir))
            disk_total_gb = disk.total / (1024**3)
            disk_free_gb = disk.free / (1024**3)

            return {
                'ram_total_gb': ram_total_gb,
                'ram_used_gb': ram_used_gb,
                'ram_pct': ram_pct,
                'disk_total_gb': disk_total_gb,
                'disk_free_gb': disk_free_gb,
            }
        except ImportError:
            self.logger.warning("psutil not available, using fallback resource check")
            return self._query_fallback()
        except Exception as e:
            self.logger.warning(f"Resource query error: {e}")
            return {}

    def _query_fallback(self) -> Dict[str, float]:
        """Fallback resource query using /proc/meminfo and df."""
        result = {}

        # RAM from /proc/meminfo (Linux)
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(':')
                        value = int(parts[1])  # in KB
                        meminfo[key] = value

                total_kb = meminfo.get('MemTotal', 0)
                available_kb = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))
                used_kb = total_kb - available_kb

                result['ram_total_gb'] = total_kb / (1024**2)
                result['ram_used_gb'] = used_kb / (1024**2)
                result['ram_pct'] = (used_kb / total_kb * 100) if total_kb > 0 else 0
        except Exception:
            pass

        # Disk from df command
        try:
            df_result = subprocess.run(
                ['df', '-B1', str(self.output_dir)],
                capture_output=True, text=True, timeout=5
            )
            if df_result.returncode == 0:
                lines = df_result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    parts = lines[1].split()
                    if len(parts) >= 4:
                        result['disk_total_gb'] = int(parts[1]) / (1024**3)
                        result['disk_free_gb'] = int(parts[3]) / (1024**3)
        except Exception:
            pass

        return result

    def _update_readings(self, data: Dict[str, float]):
        """Update internal readings from query data."""
        with self.lock:
            self.current_ram_total_gb = data.get('ram_total_gb', 0)
            self.current_ram_used_gb = data.get('ram_used_gb', 0)
            self.current_ram_pct = data.get('ram_pct', 0)
            self.current_disk_total_gb = data.get('disk_total_gb', 0)
            self.current_disk_free_gb = data.get('disk_free_gb', 0)

    def _check_thresholds(self):
        """Check resource thresholds and trigger callbacks."""
        with self.lock:
            # Check RAM critical threshold
            if self.current_ram_pct >= self.ram_critical_pct:
                if not self.ram_critical_state:
                    self.ram_critical_state = True
                    self.logger.error(
                        f"CRITICAL: RAM at {self.current_ram_pct:.1f}% "
                        f"({self.current_ram_used_gb:.1f}/{self.current_ram_total_gb:.1f} GB)"
                    )
                    if self.ram_critical_callback:
                        self.ram_critical_callback(self.current_ram_pct)
            else:
                self.ram_critical_state = False

            # Check RAM backpressure threshold
            if self.current_ram_pct >= self.ram_threshold_pct:
                if not self.ram_backpressure_active:
                    self.ram_backpressure_active = True
                    self.logger.warning(
                        f"RAM BACKPRESSURE: {self.current_ram_pct:.1f}% "
                        f"({self.current_ram_used_gb:.1f}/{self.current_ram_total_gb:.1f} GB)"
                    )
                    if self.ram_backpressure_callback:
                        self.ram_backpressure_callback(self.current_ram_pct)
            else:
                if self.ram_backpressure_active:
                    self.logger.info(f"RAM backpressure released: {self.current_ram_pct:.1f}%")
                self.ram_backpressure_active = False

            # Check disk space
            if self.current_disk_free_gb < self.disk_min_gb:
                if not self.disk_warning_active:
                    self.disk_warning_active = True
                    self.logger.warning(
                        f"LOW DISK SPACE: {self.current_disk_free_gb:.1f} GB free "
                        f"(minimum: {self.disk_min_gb} GB)"
                    )
                    if self.disk_warning_callback:
                        self.disk_warning_callback(self.current_disk_free_gb)
            else:
                self.disk_warning_active = False

    def _monitor_loop(self):
        """Background monitoring loop."""
        self.logger.info("System resource monitor started")

        while not self.stop_event.is_set():
            data = self._query_system_resources()
            if data:
                self._update_readings(data)
                self._check_thresholds()

            self.stop_event.wait(self.poll_interval)

        self.logger.info("System resource monitor stopped")

    def start(self):
        """Start the background monitor thread."""
        if self.running:
            return

        self.running = True
        self.stop_event.clear()
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop the background monitor thread."""
        if not self.running:
            return

        self.running = False
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            self.monitor_thread = None

    def get_current_status(self) -> Dict[str, Any]:
        """Get current system resource status."""
        with self.lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'ram_used_gb': self.current_ram_used_gb,
                'ram_total_gb': self.current_ram_total_gb,
                'ram_pct': self.current_ram_pct,
                'ram_free_gb': self.current_ram_total_gb - self.current_ram_used_gb,
                'disk_free_gb': self.current_disk_free_gb,
                'disk_total_gb': self.current_disk_total_gb,
                'ram_backpressure_active': self.ram_backpressure_active,
                'ram_critical_state': self.ram_critical_state,
                'disk_warning_active': self.disk_warning_active,
            }

    def log_status(self, prefix: str = ""):
        """Log current system resource status."""
        status = self.get_current_status()
        self.logger.info(
            f"{prefix}System Resources: "
            f"RAM {status['ram_used_gb']:.1f}/{status['ram_total_gb']:.1f} GB ({status['ram_pct']:.1f}%) | "
            f"Disk {status['disk_free_gb']:.1f} GB free"
        )

    def set_ram_backpressure_callback(self, callback: callable):
        """Set callback for RAM backpressure events."""
        self.ram_backpressure_callback = callback

    def set_ram_critical_callback(self, callback: callable):
        """Set callback for RAM critical events."""
        self.ram_critical_callback = callback


# ============================================================================
# NVLINK TOPOLOGY DETECTION
# ============================================================================

class NVLinkTopology:
    """
    NVLink topology detection for optimal GPU pairing.

    On systems with NVLink (like Lambda Cloud 8x A100), this detects
    which GPUs are connected via NVLink for optimal scheduling of
    multi-GPU algorithms.
    """

    def __init__(self, num_gpus: int, logger: Optional[logging.Logger] = None):
        self.num_gpus = num_gpus
        self.logger = logger or logging.getLogger(__name__)

        # NVLink adjacency matrix (True if GPUs are connected via NVLink)
        self.nvlink_matrix: List[List[bool]] = [
            [False] * num_gpus for _ in range(num_gpus)
        ]

        # GPU pairs connected via NVLink
        self.nvlink_pairs: List[Tuple[int, int]] = []

        # Detect topology
        self._detect_topology()

    def _detect_topology(self):
        """Detect NVLink topology using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', 'topo', '-m'],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                self.logger.debug("nvidia-smi topo command not available")
                return

            # Parse topology matrix
            # Format: GPU0  GPU1  GPU2 ... (header)
            #         X     NV4   SYS  ... (GPU0 row)
            lines = result.stdout.strip().split('\n')

            # Find the matrix portion
            matrix_start = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('GPU'):
                    matrix_start = i
                    break

            if matrix_start < 0:
                return

            # Parse each GPU row
            for i, line in enumerate(lines[matrix_start + 1:]):
                if not line.strip().startswith('GPU'):
                    break

                parts = line.split()
                gpu_idx = int(parts[0].replace('GPU', ''))

                if gpu_idx >= self.num_gpus:
                    continue

                # Check connections to other GPUs
                for j, conn in enumerate(parts[1:]):
                    if j >= self.num_gpus or j == gpu_idx:
                        continue

                    # NVLink connections are marked as NV1, NV2, NV3, NV4, etc.
                    if conn.startswith('NV'):
                        self.nvlink_matrix[gpu_idx][j] = True
                        if gpu_idx < j:  # Avoid duplicates
                            self.nvlink_pairs.append((gpu_idx, j))

            if self.nvlink_pairs:
                self.logger.info(f"Detected NVLink pairs: {self.nvlink_pairs}")
            else:
                self.logger.debug("No NVLink connections detected")

        except subprocess.TimeoutExpired:
            self.logger.debug("nvidia-smi topo timed out")
        except FileNotFoundError:
            self.logger.debug("nvidia-smi not found")
        except Exception as e:
            self.logger.debug(f"NVLink detection error: {e}")

    def are_linked(self, gpu_a: int, gpu_b: int) -> bool:
        """Check if two GPUs are connected via NVLink."""
        if gpu_a >= self.num_gpus or gpu_b >= self.num_gpus:
            return False
        return self.nvlink_matrix[gpu_a][gpu_b]

    def get_linked_gpus(self, gpu_id: int) -> List[int]:
        """Get list of GPUs connected to given GPU via NVLink."""
        if gpu_id >= self.num_gpus:
            return []
        return [i for i in range(self.num_gpus) if self.nvlink_matrix[gpu_id][i]]

    def get_best_pair(self, available_gpus: List[int]) -> Optional[Tuple[int, int]]:
        """
        Get best GPU pair from available GPUs.
        Prefers NVLink-connected pairs.
        """
        # Try to find NVLink pair first
        for gpu_a, gpu_b in self.nvlink_pairs:
            if gpu_a in available_gpus and gpu_b in available_gpus:
                return (gpu_a, gpu_b)

        # Fall back to any pair
        if len(available_gpus) >= 2:
            return (available_gpus[0], available_gpus[1])

        return None

    def has_nvlink(self) -> bool:
        """Check if system has any NVLink connections."""
        return len(self.nvlink_pairs) > 0


# ============================================================================
# PRIORITIZED EXPERIMENT
# ============================================================================

@dataclass
class PrioritizedExperiment:
    """Experiment with priority score for queue ordering."""
    algo_config: Dict[str, Any]
    env_config: Dict[str, Any]
    seed: int
    priority: float  # Higher = run first (or later if prioritize_fast=False)
    estimated_minutes: float
    memory_gb: float

    def to_tuple(self) -> Tuple:
        return (self.algo_config, self.env_config, self.seed)


def compute_experiment_priority(
    algo_config: Dict[str, Any],
    env_config: Dict[str, Any],
    prioritize_fast: bool = False,
) -> Tuple[float, float]:
    """
    Compute priority and time estimate for an experiment.

    Returns (priority, estimated_minutes).
    If prioritize_fast=True, fast algorithms get higher priority.
    If prioritize_fast=False, slow algorithms get higher priority (for early completion).
    """
    algo_name = algo_config["name"]

    # Get time estimate (default to 30 minutes for unknown)
    estimated_minutes = ALGO_TIME_ESTIMATES.get(algo_name, 30)

    # Adjust for environment category
    category = env_config.get("category", "dyadic")
    if category == "ecosystem":
        estimated_minutes *= 1.5
    elif category == "collective_action":
        estimated_minutes *= 1.25

    # Compute priority (higher = earlier in queue)
    if prioritize_fast:
        # Fast algorithms first: priority = inverse of time
        priority = 1000.0 / max(estimated_minutes, 1)
    else:
        # Slow algorithms first: priority = time
        priority = estimated_minutes

    return priority, estimated_minutes


# ============================================================================
# ALGORITHM FACTORY
# ============================================================================

def get_algorithm_class(algo_config: Dict[str, Any]):
    """Import and return the algorithm class from algorithms.py."""
    from algorithms import (
        RandomPolicy, ConstantPolicy, TitForTatPolicy,
        CoopetitiveEquilibriumOracle, NashEquilibriumOracle, SocialOptimumOracle,
        TrustAwareEquilibriumOracle, LoyaltyAugmentedOracle,
        ReciprocityEquilibriumOracle, BoundedReciprocityOracle,
        IndependentPPO, IndependentSAC, IndependentA2C,
        IndependentREINFORCE,
        MAPPO, MADDPG, MATD3, MASAC,
        QMIX, VDN, COMA,
        LOLA, M3DDPG,
        SelfPlayPPO, FictitiousCoPlay,
        MeanFieldActorCritic
    )

    class_map = {
        "RandomPolicy": RandomPolicy,
        "ConstantPolicy": ConstantPolicy,
        "TitForTatPolicy": TitForTatPolicy,
        "CoopetitiveEquilibriumOracle": CoopetitiveEquilibriumOracle,
        "NashEquilibriumOracle": NashEquilibriumOracle,
        "SocialOptimumOracle": SocialOptimumOracle,
        "TrustAwareEquilibriumOracle": TrustAwareEquilibriumOracle,
        "LoyaltyAugmentedOracle": LoyaltyAugmentedOracle,
        "ReciprocityEquilibriumOracle": ReciprocityEquilibriumOracle,
        "BoundedReciprocityOracle": BoundedReciprocityOracle,
        "IndependentPPO": IndependentPPO,
        "IndependentSAC": IndependentSAC,
        "IndependentA2C": IndependentA2C,
        "IndependentREINFORCE": IndependentREINFORCE,
        "MAPPO": MAPPO,
        "MADDPG": MADDPG,
        "MATD3": MATD3,
        "MASAC": MASAC,
        "QMIX": QMIX,
        "VDN": VDN,
        "COMA": COMA,
        "LOLA": LOLA,
        "M3DDPG": M3DDPG,
        "SelfPlayPPO": SelfPlayPPO,
        "FictitiousCoPlay": FictitiousCoPlay,
        "MeanFieldActorCritic": MeanFieldActorCritic,
    }

    class_name = algo_config["class"]
    if class_name not in class_map:
        raise ValueError(f"Unknown algorithm class: {class_name}")

    return class_map[class_name]


def create_environment(env_id: str, seed: int = None):
    """Create a coopetition_gym environment."""
    try:
        _setup_path()
        import coopetition_gym
        env = coopetition_gym.make(env_id)
        if seed is not None:
            env.reset(seed=seed)
        return env, None
    except Exception as e:
        return None, f"Failed to create environment {env_id}: {str(e)}\n{traceback.format_exc()}"


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(
    algo_config: Dict[str, Any],
    env_config: Dict[str, Any],
    training_seed: int,
    n_eval_episodes: int,
    gpu_id: int = -1,
    enable_gpu_isolation: bool = True,
    reduced_buffer_level: int = 0,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_interval: int = 100000,
    log_file: Optional[str] = None,
    progress_dir: Optional[Path] = None,
) -> ExperimentResult:
    """
    Run a single experiment with proper GPU assignment and resource management.

    Parameters
    ----------
    algo_config : Dict
        Algorithm configuration
    env_config : Dict
        Environment configuration
    training_seed : int
        Random seed for training
    n_eval_episodes : int
        Number of evaluation episodes
    gpu_id : int
        GPU ID to use (-1 for CPU)
    enable_gpu_isolation : bool
        If True, set CUDA_VISIBLE_DEVICES for process isolation
    reduced_buffer_level : int
        Memory pressure level (0=normal, 1=moderate reduction, 2=aggressive)
    checkpoint_dir : Path, optional
        Directory for checkpoints (None = no checkpoints)
    checkpoint_interval : int
        Steps between checkpoints
    log_file : str, optional
        Path to log file for worker process logging. Required because
        multiprocessing.spawn creates fresh processes that don't inherit
        the parent's logging configuration.
    progress_dir : Path, optional
        Directory for progress/pulsecheck files (None = no progress files)
    """
    _setup_path()

    algo_name = algo_config["name"]
    env_id = env_config["id"]
    env_category = env_config.get("category", "dyadic")
    tr_mode = env_config.get("tr", "unknown")
    n_agents = env_config.get("n_agents", 2)

    # Setup experiment logger for this worker process.
    # With multiprocessing.spawn, child processes start fresh — the parent's
    # logging configuration (FileHandler on "orchestrator" logger) is NOT
    # inherited. We must configure the "experiment" logger here so messages
    # like "Training complete" and "Metrics recorded:" reach the log file.
    exp_logger = logging.getLogger("experiment")
    if not exp_logger.handlers and log_file:
        exp_logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
        ))
        exp_logger.addHandler(fh)
    log_prefix = f"[{algo_name}] {env_id} seed={training_seed}"

    result = ExperimentResult(
        algorithm=algo_name,
        environment=env_id,
        training_seed=training_seed,
        status="failed",
        timestamp=datetime.now().isoformat(),
        gpu_id=gpu_id,
        tr_mode=tr_mode,
    )

    experiment_start_time = time.time()

    try:
        import torch

        # GPU Isolation: Set CUDA_VISIBLE_DEVICES before any CUDA operations
        # This ensures the process only sees its assigned GPU.
        # Design note: Each experiment runs on a single GPU (no DataParallel/DDP).
        # Multi-GPU parallelism comes from running N experiments simultaneously
        # across N GPUs via round-robin allocation, not from splitting one
        # experiment across GPUs. For 128x128 MLP policies, inter-GPU communication
        # overhead would exceed any parallelism benefit from DDP.
        if enable_gpu_isolation and gpu_id >= 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            # After setting CUDA_VISIBLE_DEVICES, the GPU appears as device 0
            effective_device_id = 0
        else:
            effective_device_id = gpu_id

        # Clear CUDA cache at start to release any leftover memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Set device based on GPU assignment
        if torch.cuda.is_available() and gpu_id >= 0:
            torch.cuda.set_device(effective_device_id)
            device = f"cuda:{effective_device_id}"
        else:
            device = "cpu"

        # LOG: Experiment started
        device_str = f"GPU {gpu_id}" if gpu_id >= 0 else "CPU"
        exp_logger.info(f"{log_prefix} | STARTED | {device_str} | {n_agents} agents | {tr_mode}")

        # Create environment
        env, error = create_environment(env_id, seed=training_seed)
        if error:
            result.error_message = error
            exp_logger.error(f"{log_prefix} | FAILED | Environment creation: {error[:100]}")
            return result

        exp_logger.info(f"{log_prefix} | Environment created | horizon={env_config.get('horizon', 'N/A')}")

        # Get algorithm class
        try:
            AlgoClass = get_algorithm_class(algo_config)
        except Exception as e:
            result.error_message = f"Failed to load algorithm: {str(e)}"
            exp_logger.error(f"{log_prefix} | FAILED | Algorithm load: {str(e)[:100]}")
            env.close()
            return result

        # Initialize algorithm with adaptive buffer sizes
        algo_params = algo_config.get("params", {}).copy()  # Make a copy to modify
        requires_training = algo_config.get("requires_training", True)

        # Apply reduced buffer parameters if under memory pressure
        if reduced_buffer_level > 0 and algo_name in REDUCED_BUFFER_PARAMS.get(reduced_buffer_level, {}):
            reduced_params = REDUCED_BUFFER_PARAMS[reduced_buffer_level][algo_name]
            for key, value in reduced_params.items():
                algo_params[key] = value
            logging.getLogger(__name__).info(
                f"[{algo_name}] Applied reduced buffer level {reduced_buffer_level}: {reduced_params}"
            )

        try:
            agent = AlgoClass(
                env=env,
                device=device,
                seed=training_seed,
                **algo_params
            )
            exp_logger.info(f"{log_prefix} | Agent initialized | device={device}")
        except Exception as e:
            result.error_message = f"Failed to initialize algorithm: {str(e)}\n{traceback.format_exc()}"
            exp_logger.error(f"{log_prefix} | FAILED | Agent init: {str(e)[:100]}")
            env.close()
            return result

        # Training phase with optional checkpointing
        training_start = time.time()
        if requires_training:
            # Allow a global override via environment variable. Set by the
            # parent process before dispatch; read here in the worker where
            # multiprocessing.spawn has reset Python state but inherited env.
            override = os.environ.get("COOPETITION_TIMESTEPS_OVERRIDE")
            if override is not None:
                try:
                    timesteps = int(override)
                except ValueError:
                    timesteps = TIMESTEPS_BY_CATEGORY.get(env_category, 500000)
            else:
                timesteps = TIMESTEPS_BY_CATEGORY.get(env_category, 500000)
            exp_logger.info(f"{log_prefix} | Training started | {timesteps:,} timesteps")

            # Check for existing checkpoint to resume from
            checkpoint_path = None
            resume_step = 0
            if checkpoint_dir:
                checkpoint_dir = Path(checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_pattern = f"{algo_name}_{env_id}_{training_seed}_step_*.pt"
                existing_checkpoints = sorted(
                    checkpoint_dir.glob(checkpoint_pattern),
                    key=lambda p: int(p.stem.split('_step_')[-1])
                )
                if existing_checkpoints:
                    checkpoint_path = existing_checkpoints[-1]
                    resume_step = int(checkpoint_path.stem.split('_step_')[-1])
                    logging.getLogger(__name__).info(
                        f"[{algo_name}] Found checkpoint at step {resume_step}: {checkpoint_path}"
                    )

            try:
                # Load checkpoint if available
                if checkpoint_path and hasattr(agent, 'load'):
                    try:
                        agent.load(str(checkpoint_path))
                        logging.getLogger(__name__).info(f"[{algo_name}] Resumed from checkpoint")
                    except Exception as e:
                        logging.getLogger(__name__).warning(
                            f"[{algo_name}] Failed to load checkpoint: {e}, starting fresh"
                        )
                        resume_step = 0

                # Training with progress logging and optional checkpointing
                remaining_steps = max(0, timesteps - resume_step)
                if remaining_steps > 0:
                    # Progress logging configuration — log every 50K steps or 5 minutes
                    progress_interval = max(remaining_steps // 20, 10000)
                    last_progress_log = [training_start]  # Use list for closure mutability
                    last_logged_step = [0]

                    # Pulsecheck/progress file identifier
                    _progress_id = f"{algo_name}_{env_id}_{training_seed}"

                    def combined_callback(step):
                        """Combined callback for progress logging, checkpointing, and observability."""
                        nonlocal checkpoint_dir, checkpoint_interval

                        # Progress logging (every ~5% or every 5 minutes)
                        now = time.time()
                        time_since_last = now - last_progress_log[0]
                        should_log_progress = (
                            (step - last_logged_step[0] >= progress_interval) or
                            (time_since_last >= 300)  # 5 minutes
                        )

                        elapsed = now - training_start
                        pct = 100 * step / remaining_steps if remaining_steps > 0 else 100
                        rate = step / max(elapsed, 1)

                        if step > 0 and should_log_progress:
                            eta_seconds = (remaining_steps - step) / max(rate, 1)
                            exp_logger.info(
                                f"{log_prefix} | Progress: {step:,}/{remaining_steps:,} ({pct:.0f}%) | "
                                f"{rate:.0f} steps/s | ETA: {eta_seconds/60:.0f}m"
                            )
                            last_progress_log[0] = now
                            last_logged_step[0] = step

                        # Pulsecheck file — written every callback (every 5000 steps)
                        # Lightweight: external monitors can stat this file for liveness
                        if progress_dir:
                            try:
                                pulse_file = progress_dir / f"{_progress_id}.pulse"
                                pulse_file.write_text(
                                    f"{step}/{remaining_steps} {pct:.1f}% "
                                    f"{rate:.0f}sps {elapsed:.0f}s\n"
                                )
                            except Exception:
                                pass  # Non-critical, never crash training

                        # Progress JSON — written every progress_interval for detailed monitoring
                        if progress_dir and step > 0 and should_log_progress:
                            try:
                                progress_file = progress_dir / f"{_progress_id}.json"
                                progress_data = {
                                    "algorithm": algo_name,
                                    "environment": env_id,
                                    "seed": training_seed,
                                    "gpu_id": gpu_id,
                                    "current_step": step,
                                    "total_steps": remaining_steps,
                                    "percent_complete": round(pct, 1),
                                    "steps_per_second": round(rate, 1),
                                    "elapsed_seconds": round(elapsed, 1),
                                    "eta_seconds": round((remaining_steps - step) / max(rate, 1), 1),
                                    "timestamp": datetime.utcnow().isoformat() + "Z",
                                }
                                # Atomic write: write to temp then rename
                                tmp_file = progress_file.with_suffix('.tmp')
                                tmp_file.write_text(json.dumps(progress_data))
                                tmp_file.rename(progress_file)
                            except Exception:
                                pass  # Non-critical

                        # TrainingMetrics flush to disk — alongside progress JSON
                        if progress_dir and step > 0 and should_log_progress:
                            if hasattr(agent, 'training_metrics') and agent.training_metrics.history:
                                try:
                                    metrics_file = progress_dir / f"{_progress_id}_metrics.json"
                                    tmp_file = metrics_file.with_suffix('.tmp')
                                    tmp_file.write_text(json.dumps(agent.training_metrics.to_dict()))
                                    tmp_file.rename(metrics_file)
                                except Exception:
                                    pass  # Non-critical

                        # Checkpointing (if enabled)
                        if checkpoint_dir and checkpoint_interval > 0:
                            if step > 0 and step % checkpoint_interval == 0:
                                ckpt_file = checkpoint_dir / f"{algo_name}_{env_id}_{training_seed}_step_{step}.pt"
                                if hasattr(agent, 'save'):
                                    try:
                                        agent.save(str(ckpt_file))
                                    except Exception as e:
                                        exp_logger.warning(f"{log_prefix} | Checkpoint save failed: {e}")
                                    # Clean up old checkpoints (keep last N)
                                    max_ckpts = CHECKPOINT_CONFIG.get("max_checkpoints", 3)
                                    all_ckpts = sorted(
                                        checkpoint_dir.glob(f"{algo_name}_{env_id}_{training_seed}_step_*.pt"),
                                        key=lambda p: int(p.stem.split('_step_')[-1])
                                    )
                                    for old_ckpt in all_ckpts[:-max_ckpts]:
                                        old_ckpt.unlink()

                    # Use callback-based training if available (for progress logging)
                    if hasattr(agent, 'train_with_callback'):
                        agent.train_with_callback(
                            total_timesteps=remaining_steps,
                            callback=combined_callback
                        )
                    else:
                        # Fallback to plain training without progress logging
                        exp_logger.info(f"{log_prefix} | Training in progress (no callback support)...")
                        agent.train(total_timesteps=remaining_steps)

                    # Save final checkpoint
                    if checkpoint_dir and hasattr(agent, 'save'):
                        final_ckpt = checkpoint_dir / f"{algo_name}_{env_id}_{training_seed}_step_{timesteps}.pt"
                        agent.save(str(final_ckpt))

                    # Clean up progress/pulse files (training complete)
                    if progress_dir:
                        for suffix in ['.pulse', '.json', '_metrics.json']:
                            pf = progress_dir / f"{_progress_id}{suffix}"
                            if pf.exists():
                                try:
                                    pf.unlink()
                                except Exception:
                                    pass

            except Exception as e:
                result.error_message = f"Training failed: {str(e)}\n{traceback.format_exc()}"
                exp_logger.error(f"{log_prefix} | FAILED | Training: {str(e)[:100]}")
                env.close()
                return result

        result.training_time_seconds = time.time() - training_start

        # LOG: Training completed
        if requires_training:
            train_hours = result.training_time_seconds / 3600
            exp_logger.info(f"{log_prefix} | Training complete | {train_hours:.2f}h ({result.training_time_seconds:.0f}s)")

        # Capture training curve data (returns and timesteps for proper learning curves)
        if hasattr(agent, 'training_returns') and agent.training_returns:
            training_curve = agent.training_returns
        else:
            training_curve = []

        if hasattr(agent, 'training_timesteps') and agent.training_timesteps:
            training_timesteps = agent.training_timesteps
        else:
            training_timesteps = []

        # Capture algorithm-level training metrics (losses, Q-values, entropy, etc.)
        if hasattr(agent, 'training_metrics') and agent.training_metrics.history:
            training_metrics_data = agent.training_metrics.to_dict()
            metric_names = list(agent.training_metrics.history.keys())
            exp_logger.info(f"{log_prefix} | Metrics recorded: {', '.join(metric_names)}")
        else:
            training_metrics_data = {}

        # Evaluation phase
        exp_logger.info(f"{log_prefix} | Evaluation started | {n_eval_episodes} episodes")
        eval_start = time.time()
        try:
            metrics = evaluate_agent(agent, env_id, n_eval_episodes, training_seed)
            if training_curve:
                metrics['training_returns'] = training_curve
            if training_timesteps:
                metrics['training_timesteps'] = training_timesteps
            if training_metrics_data:
                metrics['training_metrics'] = training_metrics_data
            result.metrics = metrics
            result.status = "success"
        except Exception as e:
            result.error_message = f"Evaluation failed: {str(e)}\n{traceback.format_exc()}"
            exp_logger.error(f"{log_prefix} | FAILED | Evaluation: {str(e)[:100]}")
            env.close()
            return result

        result.evaluation_time_seconds = time.time() - eval_start
        exp_logger.info(f"{log_prefix} | Evaluation complete | {result.evaluation_time_seconds:.1f}s")

        # Cleanup
        env.close()
        if hasattr(agent, 'close'):
            agent.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # LOG: Final result summary
        total_time = time.time() - experiment_start_time
        mean_return = result.metrics.get('mean_return', 0) if result.metrics else 0
        exp_logger.info(
            f"{log_prefix} | SUCCESS | return={mean_return:.2f} | "
            f"total={total_time:.1f}s ({total_time/3600:.2f}h)"
        )

        return result

    except Exception as e:
        result.error_message = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        total_time = time.time() - experiment_start_time
        exp_logger.error(f"{log_prefix} | FAILED | Unexpected error after {total_time:.1f}s: {str(e)[:100]}")
        return result


def _extract_tr_specific_metrics(info: Dict[str, Any], env_id: str) -> Dict[str, Any]:
    """
    Extract TR-specific metrics from environment info based on environment ID.

    TR-1 (Value & Interdependence): synergy, integrated utility
    TR-2 (Trust & Reputation): trust dynamics, reputation scores
    TR-3 (Collective Action): loyalty, efficiency, free-riders, coalitions
    """
    tr_metrics = {}

    # TR-1: Dyadic and Benchmark environments
    if env_id in ["TrustDilemma-v0", "PartnerHoldUp-v0"]:
        # Dyadic: defection/cooperation counts, trust asymmetry
        tr_metrics["defection_counts"] = info.get("defection_counts", [])
        tr_metrics["cooperation_counts"] = info.get("cooperation_counts", [])
        tr_metrics["cooperation_rates"] = info.get("cooperation_rates", [])
        if env_id == "PartnerHoldUp-v0":
            tr_metrics["power_asymmetry"] = info.get("power_asymmetry", 0.0)
            tr_metrics["trust_asymmetry"] = info.get("trust_asymmetry", 0.0)

    elif env_id in ["RecoveryRace-v0"]:
        # Recovery dynamics
        tr_metrics["peak_trust"] = info.get("peak_trust", 0.0)
        tr_metrics["recovery_step"] = info.get("recovery_step", 0)
        tr_metrics["trust_ceiling"] = info.get("trust_ceiling", 1.0)
        tr_metrics["recovery_progress"] = info.get("recovery_progress", 0.0)

    elif env_id in ["SynergySearch-v0"]:
        # Synergy parameters
        tr_metrics["true_gamma"] = info.get("true_gamma")
        tr_metrics["gamma_type"] = info.get("gamma_type", "unknown")
        tr_metrics["cumulative_rewards"] = info.get("cumulative_rewards", [])
        tr_metrics["reward_variance"] = info.get("reward_variance", 0.0)

    elif env_id in ["CooperativeNegotiation-v0"]:
        # Negotiation outcomes
        tr_metrics["agreement_reached"] = info.get("agreement_reached", False)
        tr_metrics["total_agreements"] = info.get("total_agreements", 0)
        tr_metrics["total_breaches"] = info.get("total_breaches", 0)
        tr_metrics["proposal_convergence"] = info.get("proposal_convergence", 0.0)

    # TR-2: Ecosystem and Validated environments
    elif env_id in ["PlatformEcosystem-v0"]:
        tr_metrics["platform_investment"] = info.get("platform_investment", 0.0)
        tr_metrics["mean_developer_investment"] = info.get("mean_developer_investment", 0.0)
        tr_metrics["developer_trust_in_platform"] = info.get("developer_trust_in_platform", 0.0)
        tr_metrics["developer_exits"] = info.get("developer_exits", 0)

    elif env_id in ["DynamicPartnerSelection-v0"]:
        tr_metrics["reputation_scores"] = info.get("reputation_scores", [])
        tr_metrics["reputation_ranking"] = info.get("reputation_ranking", [])

    elif env_id in ["SLCD-v0"]:
        tr_metrics["samsung_investment"] = info.get("samsung_investment", 0.0)
        tr_metrics["sony_investment"] = info.get("sony_investment", 0.0)

    elif env_id in ["RenaultNissan-v0"]:
        tr_metrics["phase"] = info.get("phase", "unknown")
        tr_metrics["period"] = info.get("period", "unknown")

    elif env_id in ["ReputationMarket-v0"]:
        tr_metrics["public_reputations"] = info.get("public_reputations", [])
        tr_metrics["reputation_ranking"] = info.get("reputation_ranking", [])
        tr_metrics["mean_reputation"] = info.get("mean_reputation", 0.0)
        tr_metrics["reputation_inequality"] = info.get("reputation_inequality", 0.0)
        tr_metrics["agent_tiers"] = info.get("agent_tiers", [])

    # TR-3: Collective Action environments
    elif env_id in ["TeamProduction-v0"]:
        tr_metrics["team_output"] = info.get("team_output", 0.0)
        tr_metrics["nash_equilibrium"] = info.get("nash_equilibrium", 0.0)
        tr_metrics["social_optimum"] = info.get("social_optimum", 0.0)
        tr_metrics["efficiency_ratio"] = info.get("efficiency_ratio", 0.0)
        tr_metrics["mean_loyalty"] = info.get("mean_loyalty", 0.0)
        tr_metrics["free_rider_count"] = info.get("free_rider_count", 0)
        tr_metrics["team_cohesion"] = info.get("team_cohesion", 0.0)

    elif env_id in ["LoyaltyTeam-v0"]:
        tr_metrics["team_output"] = info.get("team_output", 0.0)
        tr_metrics["mean_loyalty"] = info.get("mean_loyalty", 0.0)
        tr_metrics["loyalty_scores"] = info.get("loyalty_scores", [])
        tr_metrics["loyalty_lift"] = info.get("loyalty_lift", 1.0)
        tr_metrics["efficiency_ratio"] = info.get("efficiency_ratio", 0.0)
        tr_metrics["free_rider_count"] = info.get("free_rider_count", 0)

    elif env_id in ["CoalitionFormation-v0"]:
        tr_metrics["coalition_size"] = info.get("coalition_size", 0)
        tr_metrics["coalition_members"] = info.get("coalition_members", [])
        tr_metrics["excluded_agents"] = info.get("excluded_agents", [])
        tr_metrics["coalition_stability"] = info.get("coalition_stability", 0.0)
        tr_metrics["mean_loyalty"] = info.get("mean_loyalty", 0.0)

    elif env_id in ["ApacheProject-v0"]:
        tr_metrics["phase"] = info.get("phase", "unknown")
        tr_metrics["phase_loyalty"] = info.get("phase_loyalty", 1.0)
        tr_metrics["expected_effort"] = info.get("expected_effort", 0.0)
        tr_metrics["effort_deviation"] = info.get("effort_deviation", 0.0)
        tr_metrics["validation_accuracy"] = info.get("validation_accuracy", 0.0)
        tr_metrics["team_output"] = info.get("team_output", 0.0)

    elif env_id in ["PublicGoods-v0"]:
        tr_metrics["total_contribution"] = info.get("total_contribution", 0.0)
        tr_metrics["public_good"] = info.get("public_good", 0.0)
        tr_metrics["contribution_rate"] = info.get("contribution_rate", 0.0)
        tr_metrics["social_efficiency"] = info.get("social_efficiency", 0.0)
        tr_metrics["mean_loyalty"] = info.get("mean_loyalty", 0.0)

    # TR-4: Reciprocity environments
    elif env_id in ["ReciprocalDilemma-v0", "GiftExchange-v0"]:
        tr_metrics["cooperation_signals"] = info.get("cooperation_signals", {})
        tr_metrics["reciprocity_effects"] = info.get("reciprocity_effects", {})
        tr_metrics["memory_averages"] = info.get("memory_averages", {})
        tr_metrics["tr4_memory_window"] = info.get("tr4_memory_window", 5)

    elif env_id in ["IndirectReciprocity-v0"]:
        tr_metrics["cooperation_signals"] = info.get("cooperation_signals", {})
        tr_metrics["reciprocity_effects"] = info.get("reciprocity_effects", {})
        tr_metrics["memory_averages"] = info.get("memory_averages", {})
        tr_metrics["tr4_memory_window"] = info.get("tr4_memory_window", 7)

    elif env_id in ["GraduatedSanction-v0"]:
        tr_metrics["cooperation_signals"] = info.get("cooperation_signals", {})
        tr_metrics["reciprocity_effects"] = info.get("reciprocity_effects", {})
        tr_metrics["memory_averages"] = info.get("memory_averages", {})
        tr_metrics["tr4_memory_window"] = info.get("tr4_memory_window", 10)

    elif env_id in ["AppleAppStore-v0"]:
        tr_metrics["cooperation_signals"] = info.get("cooperation_signals", {})
        tr_metrics["reciprocity_effects"] = info.get("reciprocity_effects", {})
        tr_metrics["memory_averages"] = info.get("memory_averages", {})
        tr_metrics["tr4_memory_window"] = info.get("tr4_memory_window", 4)

    return tr_metrics


def evaluate_agent(agent, env_id: str, n_episodes: int, base_seed: int) -> Dict[str, Any]:
    """
    Evaluate a trained agent with TR-specific metrics capture.

    Captures both universal metrics (return, trust, cooperation) and
    environment-specific metrics based on TR category.
    """
    _setup_path()
    import coopetition_gym

    episode_returns = []
    episode_lengths = []
    final_trusts = []
    cooperation_rates = []
    per_episode_data = []

    # TR-specific metric accumulators (will aggregate across episodes)
    tr_metrics_accum = {}

    for ep in range(n_episodes):
        seed = base_seed * 1000 + ep
        env = coopetition_gym.make(env_id)
        obs, info = env.reset(seed=seed)

        episode_return = 0.0
        steps = 0
        action_sum = 0.0
        terminated, truncated = False, False

        while not (terminated or truncated):
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += np.sum(reward) if isinstance(reward, np.ndarray) else reward
            action_sum += np.mean(action) if isinstance(action, np.ndarray) else action
            steps += 1

        episode_returns.append(episode_return)
        episode_lengths.append(steps)
        final_trusts.append(info.get('mean_trust', 0.0))
        cooperation_rates.append(action_sum / max(steps, 1))

        # Extract TR-specific metrics from final info
        tr_ep_metrics = _extract_tr_specific_metrics(info, env_id)

        # Build per-episode record
        ep_data = {
            "seed": seed,
            "return": float(episode_return),
            "final_trust": float(info.get('mean_trust', 0.0)),
            "steps": steps,
            "cooperation_rate": float(action_sum / max(steps, 1)),
        }
        # Add TR-specific metrics to per-episode data
        ep_data.update(tr_ep_metrics)
        per_episode_data.append(ep_data)

        # Accumulate numeric TR metrics for aggregation
        for key, value in tr_ep_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key not in tr_metrics_accum:
                    tr_metrics_accum[key] = []
                tr_metrics_accum[key].append(value)

        env.close()

    # Build result with universal metrics
    result = {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_final_trust": float(np.mean(final_trusts)),
        "std_final_trust": float(np.std(final_trusts)),
        "mean_cooperation_rate": float(np.mean(cooperation_rates)),
        "std_cooperation_rate": float(np.std(cooperation_rates)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "episodes_evaluated": len(episode_returns),
        "per_episode": per_episode_data,
    }

    # Add aggregated TR-specific metrics (mean and std for numeric values)
    tr_aggregated = {}
    for key, values in tr_metrics_accum.items():
        if len(values) > 0:
            tr_aggregated[f"mean_{key}"] = float(np.mean(values))
            tr_aggregated[f"std_{key}"] = float(np.std(values))
    result["tr_metrics"] = tr_aggregated

    return result


# ============================================================================
# LOGGING
# ============================================================================

class Logger:
    """Logger with file and console output."""

    def __init__(self, log_dir: Path, name: str):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"{name}_{self.timestamp}.log"
        self._setup()

    def _setup(self):
        self.logger = logging.getLogger("orchestrator")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"))
        self.logger.addHandler(ch)

        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
        self.logger.addHandler(fh)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class UnifiedOrchestrator:
    """
    Unified orchestrator with dynamic resource management.

    Resource Management Techniques:
    1. GPU Distribution: Dynamic allocation with memory tracking
    2. CPU Parallelization: Separate pool for heuristics
    3. Memory Optimization: Per-GPU memory tracking and bin-packing
    4. Load Balancing: Work-stealing pattern with least-loaded GPU preference
    5. Queue Packing: Sort by memory requirements for efficient packing
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        mode_str = "_".join(config.modes)
        self.logger = Logger(self.logs_dir, f"orchestrator_{mode_str}")

        # Hardware detection
        self.hardware = detect_hardware()

        # Compute worker limits with explicit overrides
        self.worker_limits = compute_safe_worker_limits(
            self.hardware,
            max_workers=config.max_workers,
            max_gpu_workers=config.max_gpu_workers,
            max_cpu_workers=config.max_cpu_workers,
        )

        # Log if max_workers was applied
        if config.max_workers:
            self.logger.info(
                f"--max-workers {config.max_workers} applied: "
                f"GPU={self.worker_limits['gpu_workers']}, "
                f"CPU={self.worker_limits['cpu_workers']}"
            )

        # GPU memory manager (allocation tracking)
        if self.hardware["num_gpus"] > 0:
            self.gpu_manager = GPUMemoryManager(
                self.hardware["num_gpus"],
                self.hardware["usable_gpu_memory_gb"]
            )
        else:
            self.gpu_manager = None

        # GPU memory monitor (runtime nvidia-smi tracking with thermal monitoring)
        self.gpu_monitor: Optional[GPUMemoryMonitor] = None
        if self.hardware["num_gpus"] > 0 and config.enable_memory_monitoring:
            self.gpu_monitor = GPUMemoryMonitor(
                num_gpus=self.hardware["num_gpus"],
                poll_interval=config.memory_poll_interval,
                memory_threshold_pct=config.memory_threshold_pct,
                critical_threshold_pct=config.critical_threshold_pct,
                thermal_throttle_c=config.thermal_throttle_c,
                thermal_critical_c=config.thermal_critical_c,
                enable_thermal=config.enable_thermal_monitoring,
                logger=self.logger.logger,
            )

        # System resource monitor (RAM/disk monitoring)
        self.system_monitor: Optional[SystemResourceMonitor] = None
        if config.enable_ram_monitoring:
            self.system_monitor = SystemResourceMonitor(
                output_dir=self.output_dir,
                poll_interval=config.memory_poll_interval * 2,  # Less frequent than GPU
                ram_threshold_pct=config.ram_threshold_pct,
                ram_critical_pct=config.ram_critical_pct,
                disk_min_gb=50.0,  # Warn if less than 50GB free
                logger=self.logger.logger,
            )

        # NVLink topology detection
        self.nvlink_topology: Optional[NVLinkTopology] = None
        if self.hardware["num_gpus"] > 0 and config.enable_nvlink_scheduling:
            self.nvlink_topology = NVLinkTopology(
                num_gpus=self.hardware["num_gpus"],
                logger=self.logger.logger,
            )

        # Checkpoint directory
        self.checkpoint_dir: Optional[Path] = None
        if config.enable_checkpoints:
            self.checkpoint_dir = config.checkpoint_dir or (self.output_dir / "checkpoints")
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Checkpoints enabled: {self.checkpoint_dir}")

        # Progress/pulsecheck directory for observability
        self.progress_dir: Path = self.output_dir / "progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Progress telemetry enabled: {self.progress_dir}")

        # Adaptive buffer state
        self.current_buffer_level = 0  # 0=normal, 1=moderate, 2=aggressive

        # Backpressure controller
        self.backpressure: Optional[BackpressureController] = None
        if config.enable_backpressure and self.hardware["num_gpus"] > 0:
            self.backpressure = BackpressureController(
                initial_gpu_workers=self.worker_limits["gpu_workers"],
                initial_cpu_workers=self.worker_limits["cpu_workers"],
                min_gpu_workers=max(2, self.worker_limits["gpu_workers"] // 4),
                min_cpu_workers=max(2, self.worker_limits["cpu_workers"] // 4),
                reduction_factor=BACKPRESSURE_CONFIG["reduction_factor"],
                cooldown_seconds=config.backpressure_cooldown,
                logger=self.logger.logger,
            )

            # Connect monitor callbacks to backpressure controller
            if self.gpu_monitor:
                self.gpu_monitor.set_backpressure_callback(
                    lambda util: self._handle_memory_backpressure(util)
                )
                self.gpu_monitor.set_critical_callback(
                    lambda util: self._handle_critical_memory(util)
                )
                # Connect thermal callback for thermal throttling
                self.gpu_monitor.set_thermal_callback(
                    lambda temp: self._handle_thermal_throttle(temp)
                )
                # Connect thermal CRITICAL callback (separate from memory critical!)
                self.gpu_monitor.set_thermal_critical_callback(
                    lambda temp: self._handle_thermal_critical(temp)
                )

            # Connect system monitor callbacks
            if self.system_monitor:
                self.system_monitor.set_ram_backpressure_callback(
                    lambda pct: self._handle_ram_backpressure(pct)
                )
                self.system_monitor.set_ram_critical_callback(
                    lambda pct: self._handle_critical_ram(pct)
                )

        # State for resume
        self.state_file = self.logs_dir / "state.json"
        self.completed_keys: Set[str] = set()

        # ALWAYS auto-detect completed experiments from existing result files.
        # This prevents re-running experiments after a restart, regardless of
        # whether --resume was passed. Result files are the source of truth.
        self._scan_completed_results()

        # Also load from state file if --resume and it has additional entries
        if config.resume and self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
                state_keys = set(state.get("completed", []))
                self.completed_keys.update(state_keys)

        # Build experiment queues (separate for CPU and GPU)
        self.gpu_experiments, self.cpu_experiments = self._build_experiments()

        # Graceful shutdown
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _handle_memory_backpressure(self, utilization: float):
        """Handle GPU memory backpressure by reducing workers and increasing buffer level."""
        self.logger.warning(f"GPU MEMORY BACKPRESSURE: {utilization:.1f}%")
        if self.backpressure:
            self.backpressure.reduce_workers(f"GPU memory at {utilization:.1f}%")

        # Increase adaptive buffer level if enabled
        if self.config.enable_adaptive_buffers and self.current_buffer_level < 2:
            self.current_buffer_level = min(2, self.current_buffer_level + 1)
            self.logger.info(f"Adaptive buffers: increased to level {self.current_buffer_level}")

    def _handle_critical_memory(self, utilization: float):
        """Handle critical memory state by pausing submissions and reducing workers."""
        self.logger.error(f"CRITICAL MEMORY: {utilization:.1f}% - pausing new submissions")
        if self.backpressure:
            self.backpressure.pause_submissions()
            self.backpressure.handle_oom_error()
        if self.gpu_monitor:
            self.gpu_monitor.record_oom_error()

        # Maximum adaptive buffer reduction
        if self.config.enable_adaptive_buffers:
            self.current_buffer_level = 2
            self.logger.warning("Adaptive buffers: set to maximum reduction (level 2)")

    def _handle_thermal_throttle(self, temperature: float):
        """Handle thermal throttling by reducing workers."""
        self.logger.warning(f"THERMAL THROTTLE: GPU at {temperature:.1f}°C")
        if self.backpressure:
            self.backpressure.reduce_workers(f"thermal at {temperature:.1f}°C")

    def _handle_thermal_critical(self, temperature: float):
        """Handle thermal critical state by pausing submissions and reducing workers."""
        self.logger.error(f"THERMAL CRITICAL: {temperature:.1f}°C - pausing new submissions for cooling")
        if self.backpressure:
            self.backpressure.pause_submissions()
            self.backpressure.reduce_workers(f"thermal critical at {temperature:.1f}°C")

    def _handle_ram_backpressure(self, utilization_pct: float):
        """Handle RAM backpressure by reducing CPU workers."""
        self.logger.warning(f"RAM BACKPRESSURE: {utilization_pct:.1f}%")
        if self.backpressure:
            # For RAM pressure, primarily reduce CPU workers
            self.backpressure.reduce_workers(f"RAM at {utilization_pct:.1f}%")

    def _handle_critical_ram(self, utilization_pct: float):
        """Handle critical RAM state."""
        self.logger.error(f"CRITICAL RAM: {utilization_pct:.1f}% - pausing submissions")
        if self.backpressure:
            self.backpressure.pause_submissions()

    def _signal_handler(self, signum, frame):
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

    def _get_environments(self) -> List[Dict]:
        """Get environments based on selected modes."""
        envs = []
        for mode in self.config.modes:
            if mode in ENVIRONMENTS_BY_MODE:
                envs.extend(ENVIRONMENTS_BY_MODE[mode])

        # Remove duplicates while preserving order
        seen = set()
        unique_envs = []
        for env in envs:
            if env["id"] not in seen:
                seen.add(env["id"])
                unique_envs.append(env)

        # Filter if specific environments requested
        if self.config.environments:
            unique_envs = [e for e in unique_envs if e["id"] in self.config.environments]

        return unique_envs

    def _get_algorithms(self) -> List[Dict]:
        """Get algorithms based on configuration."""
        algos = ALL_ALGORITHMS

        if self.config.algorithms:
            algos = [a for a in algos if a["name"] in self.config.algorithms]

        return algos

    def _build_experiments(self) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Build experiment queues with memory-aware and priority-aware sorting.

        Returns two queues:
        - GPU experiments (training algorithms)
        - CPU experiments (heuristics)

        Queue Packing Strategy:
        - If priority queue enabled: sort by estimated time (configurable order)
        - Otherwise: sort by memory (descending) for efficient bin-packing
        - Then shuffle within bands to distribute across GPUs
        """
        gpu_experiments = []
        cpu_experiments = []
        prioritized_gpu = []  # For priority queue

        envs = self._get_environments()
        algos = self._get_algorithms()

        for algo in algos:
            for env in envs:
                # Check category applicability
                applicable_categories = algo.get("applicable_categories")
                if applicable_categories and env.get("category") not in applicable_categories:
                    continue

                # Check TR applicability (Oracle algorithms are TR-specific)
                applicable_trs = algo.get("applicable_trs")
                if applicable_trs and env.get("tr") not in applicable_trs:
                    continue

                for seed in self.config.seeds:
                    key = f"{algo['name']}_{env['id']}_{seed}"
                    if key not in self.completed_keys:
                        exp = (algo, env, seed)

                        if algo.get("cpu_only", False):
                            cpu_experiments.append(exp)
                        else:
                            # Create prioritized experiment if priority queue enabled
                            if self.config.enable_priority_queue:
                                priority, est_minutes = compute_experiment_priority(
                                    algo, env, self.config.prioritize_fast_algorithms
                                )
                                mem_gb = algo.get("gpu_memory_gb", 4.0)
                                prioritized_gpu.append(PrioritizedExperiment(
                                    algo_config=algo,
                                    env_config=env,
                                    seed=seed,
                                    priority=priority,
                                    estimated_minutes=est_minutes,
                                    memory_gb=mem_gb,
                                ))
                            else:
                                gpu_experiments.append(exp)

        # Priority queue sorting
        if self.config.enable_priority_queue and prioritized_gpu:
            # Sort by priority (higher = first)
            prioritized_gpu.sort(key=lambda x: x.priority, reverse=True)

            # Log priority queue info
            total_est_time = sum(p.estimated_minutes for p in prioritized_gpu)
            self.logger.info(f"Priority queue: {len(prioritized_gpu)} experiments")
            self.logger.info(f"Estimated total time: {total_est_time/60:.1f} hours")
            if self.config.prioritize_fast_algorithms:
                self.logger.info("Strategy: Fast algorithms first")
            else:
                self.logger.info("Strategy: Slow algorithms first (for early completion)")

            # Convert back to tuple format
            gpu_experiments = [p.to_tuple() for p in prioritized_gpu]

        else:
            # Legacy: Sort GPU experiments by memory requirement (descending)
            # This enables better bin-packing
            gpu_experiments.sort(key=lambda x: x[0].get("gpu_memory_gb", 4.0), reverse=True)

            # Shuffle within memory bands to distribute across GPUs
            # Group by memory, shuffle each group, interleave
            memory_bands = defaultdict(list)
            for exp in gpu_experiments:
                mem = exp[0].get("gpu_memory_gb", 4.0)
                band = int(mem)  # Group by integer GB
                memory_bands[band].append(exp)

            random.seed(self.config.shuffle_seed)
            shuffled_gpu = []
            for band in sorted(memory_bands.keys(), reverse=True):
                random.shuffle(memory_bands[band])
                shuffled_gpu.extend(memory_bands[band])
            gpu_experiments = shuffled_gpu

        # Shuffle CPU experiments (all same priority)
        random.shuffle(cpu_experiments)

        return gpu_experiments, cpu_experiments

    def _scan_completed_results(self):
        """Scan raw results directory for already-completed experiments.

        This ensures experiments are never re-run after a restart, even if
        --resume was not passed. The result files on disk are the source of
        truth for what has already been completed.

        Filename convention: {algorithm}_{environment}_{seed}.json
        """
        if not self.raw_dir.exists():
            return

        scanned = 0
        for filepath in self.raw_dir.glob("*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                # Only count successful experiments as completed
                if data.get("status") == "success":
                    key = f"{data['algorithm']}_{data['environment']}_{data['training_seed']}"
                    self.completed_keys.add(key)
                    scanned += 1
            except (json.JSONDecodeError, KeyError, OSError):
                continue

        if scanned > 0:
            self.logger.info(f"Auto-detected {scanned} completed experiments from result files")

    def _save_result(self, result: ExperimentResult):
        """Save result to file."""
        filename = f"{result.algorithm}_{result.environment}_{result.training_seed}.json"
        filepath = self.raw_dir / filename
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, separators=(',', ':'), cls=NumpyEncoder)

    def _save_state(self):
        """Save current state for resume."""
        state = {
            "completed": list(self.completed_keys),
            "timestamp": datetime.now().isoformat(),
            "modes": self.config.modes,
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _print_resource_summary(self):
        """Print resource allocation summary."""
        self.logger.info("=" * 70)
        self.logger.info("RESOURCE ALLOCATION SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"Hardware: {self.hardware['num_gpus']} GPUs, {self.hardware['num_vcpus']} vCPUs")

        if self.hardware['num_gpus'] > 0:
            self.logger.info(f"GPU Memory: {self.hardware['total_gpu_memory_gb']:.1f} GB total")
            self.logger.info(f"Usable per GPU: {self.hardware['usable_gpu_memory_gb'][0]:.1f} GB (after {GPU_MEMORY_SAFETY_MARGIN_GB}GB safety margin)")

        self.logger.info(f"Worker Limits (CPU & GPU pools run IN PARALLEL):")
        self.logger.info(f"  - GPU workers: {self.worker_limits['gpu_workers']} ({self.worker_limits['max_per_gpu']} per GPU) [replay-buffer algos]")
        self.logger.info(f"  - CPU workers: {self.worker_limits['cpu_workers']} [MLP algos + heuristics]")
        self.logger.info(f"  - Total concurrent: {self.worker_limits['total_workers']}")

        # Show if explicit limits were applied
        if self.config.max_workers:
            self.logger.info(f"  - MAX WORKERS OVERRIDE: {self.config.max_workers}")
        if self.config.max_gpu_workers:
            self.logger.info(f"  - MAX GPU WORKERS OVERRIDE: {self.config.max_gpu_workers}")
        if self.config.max_cpu_workers:
            self.logger.info(f"  - MAX CPU WORKERS OVERRIDE: {self.config.max_cpu_workers}")

        self.logger.info("")
        self.logger.info("Monitoring & Backpressure:")
        self.logger.info(f"  - GPU memory monitoring: {'ENABLED' if self.gpu_monitor else 'DISABLED'}")
        self.logger.info(f"  - Thermal monitoring: {'ENABLED' if self.config.enable_thermal_monitoring else 'DISABLED'}")
        self.logger.info(f"  - RAM monitoring: {'ENABLED' if self.system_monitor else 'DISABLED'}")
        self.logger.info(f"  - Backpressure: {'ENABLED' if self.backpressure else 'DISABLED'}")
        if self.gpu_monitor:
            self.logger.info(f"  - Poll interval: {self.config.memory_poll_interval}s")
            self.logger.info(f"  - Memory backpressure: {self.config.memory_threshold_pct}%")
            self.logger.info(f"  - Memory critical: {self.config.critical_threshold_pct}%")
            if self.config.enable_thermal_monitoring:
                self.logger.info(f"  - Thermal throttle: {self.config.thermal_throttle_c}°C")
                self.logger.info(f"  - Thermal critical: {self.config.thermal_critical_c}°C")
        if self.system_monitor:
            self.logger.info(f"  - RAM backpressure: {self.config.ram_threshold_pct}%")
            self.logger.info(f"  - RAM critical: {self.config.ram_critical_pct}%")

        self.logger.info("")
        self.logger.info("Execution Strategy:")
        self.logger.info(f"  - PARALLEL POOLS: CPU and GPU experiments run CONCURRENTLY")
        self.logger.info(f"  - No resource waste: GPUs active while CPU pool runs MLP algorithms")
        self.logger.info("")
        self.logger.info("Advanced Features:")
        self.logger.info(f"  - GPU isolation (CUDA_VISIBLE_DEVICES): {'ENABLED' if self.config.enable_gpu_isolation else 'DISABLED'}")
        self.logger.info(f"  - Adaptive buffers: {'ENABLED' if self.config.enable_adaptive_buffers else 'DISABLED'}")
        self.logger.info(f"  - Priority queue: {'ENABLED' if self.config.enable_priority_queue else 'DISABLED'}")
        if self.config.enable_priority_queue:
            self.logger.info(f"    - Strategy: {'Fast first' if self.config.prioritize_fast_algorithms else 'Slow first'}")
        self.logger.info(f"  - NVLink scheduling: {'ENABLED' if self.nvlink_topology else 'DISABLED'}")
        if self.nvlink_topology and self.nvlink_topology.has_nvlink():
            self.logger.info(f"    - NVLink pairs: {self.nvlink_topology.nvlink_pairs}")
        self.logger.info(f"  - Checkpoints: {'ENABLED' if self.checkpoint_dir else 'DISABLED'}")
        if self.checkpoint_dir:
            self.logger.info(f"    - Directory: {self.checkpoint_dir}")
            self.logger.info(f"    - Interval: {self.config.checkpoint_interval} steps")

        self.logger.info("")
        self.logger.info("Lambda Cloud 8x A100 Safe Configuration:")
        self.logger.info("  - 40GB per GPU - 4GB safety = 36GB usable")
        self.logger.info("  - Heaviest algorithm (MAPPO) = 6GB")
        self.logger.info("  - Safe per GPU = 36/6 = 6, conservative = 3")
        self.logger.info("  - Recommended: --max-workers 48 (24 GPU + 24 CPU)")
        self.logger.info("=" * 70)

    def run(self):
        """Execute all experiments with dynamic resource management."""
        mode_str = ",".join(self.config.modes)

        self.logger.info("=" * 70)
        self.logger.info("UNIFIED COOPETITION-GYM ORCHESTRATOR V2")
        self.logger.info("=" * 70)
        self.logger.info(f"Modes: {mode_str}")
        self.logger.info(f"GPU experiments: {len(self.gpu_experiments)} (replay-buffer algorithms)")
        self.logger.info(f"CPU experiments: {len(self.cpu_experiments)} (MLP algorithms + heuristics)")
        self.logger.info(f"Total: {len(self.gpu_experiments) + len(self.cpu_experiments)}")
        self.logger.info(f"Already completed: {len(self.completed_keys)}")
        self.logger.info(f"Output: {self.output_dir}")

        self._print_resource_summary()

        if self.config.dry_run:
            self.logger.info("\nDRY RUN - Experiments that would run:")
            self.logger.info("\n*** PARALLEL EXECUTION: CPU and GPU pools run CONCURRENTLY ***")
            self.logger.info("\nGPU Experiments - replay-buffer algorithms (first 10):")
            for algo, env, seed in self.gpu_experiments[:10]:
                mem = algo.get("gpu_memory_gb", 0)
                self.logger.info(f"  [{mem}GB] {algo['name']} on {env['id']} (seed={seed})")
            if len(self.gpu_experiments) > 10:
                self.logger.info(f"  ... and {len(self.gpu_experiments) - 10} more GPU experiments")

            self.logger.info("\nCPU Experiments - MLP algorithms + heuristics (first 10):")
            for algo, env, seed in self.cpu_experiments[:10]:
                self.logger.info(f"  [CPU] {algo['name']} on {env['id']} (seed={seed})")
            if len(self.cpu_experiments) > 10:
                self.logger.info(f"  ... and {len(self.cpu_experiments) - 10} more CPU experiments")
            return

        total_experiments = len(self.gpu_experiments) + len(self.cpu_experiments)
        if total_experiments == 0:
            self.logger.info("No experiments to run")
            return

        completed = 0
        failed = 0
        start_time = time.time()

        # Start monitoring threads
        if self.gpu_monitor:
            self.gpu_monitor.start()
            self.logger.info("GPU memory monitor STARTED")
            # Log initial GPU status
            self.gpu_monitor.log_status("[INITIAL] ")

        if self.system_monitor:
            self.system_monitor.start()
            self.logger.info("System resource monitor STARTED")
            self.system_monitor.log_status("[INITIAL] ")

        try:
            # Use spawn context for CUDA compatibility
            spawn_ctx = mp.get_context('spawn')

            # PARALLEL EXECUTION: Run CPU and GPU experiments CONCURRENTLY
            # This prevents GPUs from sitting idle while CPU experiments run
            from concurrent.futures import ThreadPoolExecutor, as_completed

            cpu_future = None
            gpu_future = None

            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="pool_launcher") as launcher:
                # Launch both pools in parallel
                if self.cpu_experiments:
                    self.logger.info(f"\n--- Launching {len(self.cpu_experiments)} CPU experiments (MLP algorithms + heuristics) ---")
                    cpu_future = launcher.submit(self._run_cpu_experiments, spawn_ctx)

                if self.gpu_experiments:
                    self.logger.info(f"\n--- Launching {len(self.gpu_experiments)} GPU experiments (replay-buffer algorithms) ---")
                    gpu_future = launcher.submit(self._run_gpu_experiments, spawn_ctx)

                # Wait for both pools to complete and collect results
                if cpu_future:
                    try:
                        completed_cpu, failed_cpu = cpu_future.result()
                        completed += completed_cpu
                        failed += failed_cpu
                        self.logger.info(f"CPU pool finished: {completed_cpu} completed, {failed_cpu} failed")
                    except Exception as e:
                        self.logger.error(f"CPU experiment pool failed: {e}")

                if gpu_future:
                    try:
                        completed_gpu, failed_gpu = gpu_future.result()
                        completed += completed_gpu
                        failed += failed_gpu
                        self.logger.info(f"GPU pool finished: {completed_gpu} completed, {failed_gpu} failed")
                    except Exception as e:
                        self.logger.error(f"GPU experiment pool failed: {e}")

        finally:
            # Stop monitoring threads
            if self.gpu_monitor:
                self.gpu_monitor.log_status("[FINAL] ")
                self.gpu_monitor.stop()
                self.logger.info("GPU memory monitor STOPPED")

            if self.system_monitor:
                self.system_monitor.log_status("[FINAL] ")
                self.system_monitor.stop()
                self.logger.info("System resource monitor STOPPED")

        # Final summary
        elapsed = time.time() - start_time
        self._save_state()

        self.logger.info("=" * 70)
        self.logger.info("ORCHESTRATION COMPLETE")
        self.logger.info(f"Completed: {completed}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Total time: {elapsed/3600:.2f} hours")
        self.logger.info(f"Results: {self.raw_dir}")

        # Backpressure summary
        if self.backpressure:
            bp_status = self.backpressure.get_status()
            self.logger.info(f"Backpressure events: {bp_status['reduction_count']} reductions, {bp_status['recovery_count']} recoveries")

        # OOM summary
        if self.gpu_monitor:
            monitor_status = self.gpu_monitor.get_current_status()
            if monitor_status['oom_count'] > 0:
                self.logger.warning(f"OOM errors encountered: {monitor_status['oom_count']}")

        self.logger.info("=" * 70)

    def _run_cpu_experiments(self, spawn_ctx) -> Tuple[int, int]:
        """Run CPU-only experiments (MLP training algorithms + heuristics) with high parallelism.

        This runs IN PARALLEL with _run_gpu_experiments for optimal resource utilization.
        """
        completed = 0
        failed = 0

        num_workers = self.worker_limits["cpu_workers"]

        # PATCHED: Right-size the pool to match actual work
        actual_pool_size = min(num_workers, len(self.cpu_experiments))
        self.logger.info(
            f"Creating CPU pool with {actual_pool_size} workers "
            f"for {len(self.cpu_experiments)} CPU experiments"
        )

        with ProcessPoolExecutor(max_workers=actual_pool_size, mp_context=spawn_ctx) as executor:
            futures = {}
            experiment_iter = iter(self.cpu_experiments)

            # Submit initial batch
            for _ in range(actual_pool_size):
                try:
                    algo, env, seed = next(experiment_iter)
                    future = executor.submit(
                        run_single_experiment,
                        algo, env, seed,
                        self.config.n_eval_episodes, -1,  # CPU only
                        True, 0, None, 100000,
                        str(self.logger.log_file),
                        self.progress_dir,
                    )
                    futures[future] = (algo, env, seed)
                except StopIteration:
                    break

            while futures and not self.shutdown_requested:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    algo, env, seed = futures.pop(future)
                    key = f"{algo['name']}_{env['id']}_{seed}"

                    try:
                        result = future.result()
                        self._save_result(result)
                        self.completed_keys.add(key)

                        if result.status == "success":
                            completed += 1
                            mean_ret = result.metrics.get('mean_return', 0) if result.metrics else 0
                            self.logger.info(f"[CPU {completed + failed}/{len(self.cpu_experiments)}] "
                                           f"{algo['name']} on {env['id']}: return={mean_ret:.2f}")
                        else:
                            failed += 1
                            self.logger.error(f"[FAIL] {algo['name']} on {env['id']}")
                    except Exception as e:
                        failed += 1
                        self.logger.error(f"[ERROR] {algo['name']} on {env['id']}: {e}")

                    # Submit next
                    try:
                        next_algo, next_env, next_seed = next(experiment_iter)
                        future = executor.submit(
                            run_single_experiment,
                            next_algo, next_env, next_seed,
                            self.config.n_eval_episodes, -1,
                            True, 0, None, 100000,
                            str(self.logger.log_file),
                            self.progress_dir,
                        )
                        futures[future] = (next_algo, next_env, next_seed)
                    except StopIteration:
                        pass

                if (completed + failed) % 20 == 0:
                    self._save_state()

        return completed, failed

    def _run_gpu_experiments(self, spawn_ctx) -> Tuple[int, int]:
        """Run GPU experiments with memory-aware scheduling and backpressure.

        PATCHED: Added periodic recovery attempts to restore workers after pressure subsides.
        """
        completed = 0
        failed = 0
        oom_errors = 0
        last_status_log = time.time()
        last_recovery_attempt = time.time()  # PATCHED: Track recovery attempts
        status_log_interval = 60  # Log GPU status every 60 seconds
        recovery_check_interval = 45  # Check recovery every 45 seconds for faster GPU utilization

        # Safety check: Ensure GPU resources are available
        if self.gpu_manager is None:
            self.logger.error("GPU manager not initialized - no GPUs available")
            self.logger.error("GPU experiments cannot run without GPUs")
            # Mark all GPU experiments as failed
            for algo, env, seed in self.gpu_experiments:
                result = ExperimentResult(
                    algorithm=algo['name'],
                    environment=env['id'],
                    training_seed=seed,
                    status="failed",
                    error_message="No GPUs available for GPU experiment",
                    timestamp=datetime.now().isoformat(),
                    gpu_id=-1,
                    tr_mode=env.get("tr", "unknown"),
                )
                self._save_result(result)
                failed += 1
            return completed, failed

        # Get initial worker count (may be adjusted by backpressure)
        if self.backpressure:
            num_workers, _ = self.backpressure.get_current_workers()
        else:
            num_workers = self.worker_limits["gpu_workers"]

        # PATCHED: Right-size the pool to match actual work
        # Don't spawn 192 workers for 72 experiments - waste of resources
        actual_pool_size = min(num_workers, len(self.gpu_experiments))
        self.logger.info(
            f"Creating pool with {actual_pool_size} workers "
            f"for {len(self.gpu_experiments)} GPU experiments "
            f"(max configured: {num_workers})"
        )

        with ProcessPoolExecutor(max_workers=actual_pool_size, mp_context=spawn_ctx) as executor:
            futures = {}
            experiment_iter = iter(self.gpu_experiments)
            pending_experiments = []  # Buffer for experiments waiting for backpressure relief

            # Submit initial batch with memory-aware GPU allocation
            for _ in range(num_workers):
                if self.backpressure and self.backpressure.is_paused():
                    self.logger.warning("Initial submission paused due to backpressure")
                    break

                try:
                    algo, env, seed = next(experiment_iter)
                    mem_required = algo.get("gpu_memory_gb", 4.0)

                    # Allocate GPU using memory manager
                    gpu_id = self.gpu_manager.allocate(algo['name'], mem_required)
                    if gpu_id < 0:
                        # Fall back to round-robin if no space
                        gpu_id = random.randint(0, self.hardware['num_gpus'] - 1)

                    future = executor.submit(
                        run_single_experiment,
                        algo, env, seed,
                        self.config.n_eval_episodes, gpu_id,
                        self.config.enable_gpu_isolation,
                        self.current_buffer_level,
                        self.checkpoint_dir,
                        self.config.checkpoint_interval,
                        str(self.logger.log_file),
                        self.progress_dir,
                    )
                    futures[future] = (algo, env, seed, gpu_id, mem_required)
                except StopIteration:
                    break

            while (futures or pending_experiments) and not self.shutdown_requested:
                # Wait for completion with timeout to allow periodic checks
                if futures:
                    done, _ = wait(futures.keys(), timeout=5.0, return_when=FIRST_COMPLETED)
                else:
                    done = set()
                    time.sleep(1.0)  # Wait if only pending experiments

                # Check if we should resume from backpressure pause
                if self.backpressure and self.backpressure.is_paused():
                    if self.gpu_monitor:
                        status = self.gpu_monitor.get_current_status()
                        memory_ok = status['max_utilization_pct'] < self.config.memory_threshold_pct - 5
                        # BUG FIX: Also check thermal conditions for resume
                        max_temp = status.get('max_temperature_c', 0)
                        thermal_ok = max_temp < self.config.thermal_critical_c - 5
                        if memory_ok and thermal_ok:
                            self.backpressure.resume_submissions()
                            self.logger.info(
                                f"Resuming submissions (memory: {status['max_utilization_pct']:.1f}%, "
                                f"temp: {max_temp:.1f}°C)"
                            )

                for future in done:
                    algo, env, seed, gpu_id, mem_required = futures.pop(future)
                    key = f"{algo['name']}_{env['id']}_{seed}"

                    # Release GPU memory
                    self.gpu_manager.release(gpu_id, mem_required)

                    try:
                        result = future.result()
                        self._save_result(result)
                        self.completed_keys.add(key)

                        if result.status == "success":
                            completed += 1
                            mean_ret = result.metrics.get('mean_return', 0) if result.metrics else 0
                            self.logger.info(f"[GPU{gpu_id} {completed + failed}/{len(self.gpu_experiments)}] "
                                           f"{algo['name']} on {env['id']}: return={mean_ret:.2f}")
                        else:
                            failed += 1
                            # Check for OOM in error message
                            error_msg = result.error_message or ""
                            if "out of memory" in error_msg.lower() or "CUDA" in error_msg:
                                oom_errors += 1
                                self.logger.error(f"[OOM GPU{gpu_id}] {algo['name']} on {env['id']}")
                                if self.gpu_monitor:
                                    self.gpu_monitor.record_oom_error()
                                if self.backpressure:
                                    self.backpressure.handle_oom_error()
                            else:
                                self.logger.error(f"[FAIL GPU{gpu_id}] {algo['name']} on {env['id']}")

                    except Exception as e:
                        failed += 1
                        error_str = str(e).lower()
                        if "out of memory" in error_str or "cuda" in error_str:
                            oom_errors += 1
                            self.logger.error(f"[OOM ERROR] {algo['name']} on {env['id']}: {e}")
                            if self.gpu_monitor:
                                self.gpu_monitor.record_oom_error()
                            if self.backpressure:
                                self.backpressure.handle_oom_error()
                        else:
                            self.logger.error(f"[ERROR] {algo['name']} on {env['id']}: {e}")

                    # Get current allowed workers from backpressure controller
                    if self.backpressure:
                        current_gpu_workers, _ = self.backpressure.get_current_workers()
                    else:
                        current_gpu_workers = num_workers

                    # Submit next if under worker limit and not paused
                    if len(futures) < current_gpu_workers:
                        if not (self.backpressure and self.backpressure.is_paused()):
                            # Try pending experiments first
                            if pending_experiments:
                                next_algo, next_env, next_seed = pending_experiments.pop(0)
                            else:
                                try:
                                    next_algo, next_env, next_seed = next(experiment_iter)
                                except StopIteration:
                                    continue

                            next_mem = next_algo.get("gpu_memory_gb", 4.0)
                            next_gpu = self.gpu_manager.allocate(next_algo['name'], next_mem)
                            if next_gpu < 0:
                                next_gpu = self.gpu_manager.get_least_loaded_gpu()

                            future = executor.submit(
                                run_single_experiment,
                                next_algo, next_env, next_seed,
                                self.config.n_eval_episodes, next_gpu,
                                self.config.enable_gpu_isolation,
                                self.current_buffer_level,
                                self.checkpoint_dir,
                                self.config.checkpoint_interval,
                                str(self.logger.log_file),
                                self.progress_dir,
                            )
                            futures[future] = (next_algo, next_env, next_seed, next_gpu, next_mem)

                # Buffer remaining experiments if paused
                if self.backpressure and self.backpressure.is_paused():
                    try:
                        while True:
                            exp = next(experiment_iter)
                            pending_experiments.append(exp)
                            if len(pending_experiments) > 100:  # Limit buffer size
                                break
                    except StopIteration:
                        pass

                # Periodic state save and GPU status logging
                now = time.time()
                if (completed + failed) % 20 == 0:
                    self._save_state()

                # Log GPU status periodically
                if now - last_status_log > status_log_interval:
                    if self.gpu_monitor:
                        self.gpu_monitor.log_status(f"[Progress: {completed + failed}/{len(self.gpu_experiments)}] ")
                        # Reconcile GPU memory tracker with actual nvidia-smi readings
                        if self.gpu_manager:
                            monitor_status = self.gpu_monitor.get_current_status()
                            actual_used_gb = [
                                mb / 1024.0 for mb in monitor_status.get('used_mb', [])
                            ]
                            if actual_used_gb:
                                self.gpu_manager.reconcile_with_actual(
                                    actual_used_gb, logger=self.logger
                                )
                    if self.gpu_manager:
                        status = self.gpu_manager.get_status()
                        self.logger.debug(f"GPU allocation: {status['utilization_pct']}")
                    if self.backpressure:
                        bp_status = self.backpressure.get_status()
                        self.logger.debug(f"Backpressure: workers={bp_status['gpu_workers']}, reductions={bp_status['reduction_count']}")
                    last_status_log = now

                # PATCHED: Periodic recovery attempt
                if (self.backpressure and
                    not self.backpressure.is_paused() and
                    now - last_recovery_attempt > recovery_check_interval):

                    if self.gpu_monitor:
                        status = self.gpu_monitor.get_current_status()
                        # If memory below 70%, try to recover workers
                        if status.get('max_utilization_pct', 100) < 70:
                            old_workers, _ = self.backpressure.get_current_workers()
                            new_workers, _ = self.backpressure.try_recover_workers()
                            if new_workers > old_workers:
                                self.logger.info(
                                    f"Worker recovery triggered: {old_workers} → {new_workers}"
                                )
                    last_recovery_attempt = now

        # Final OOM summary
        if oom_errors > 0:
            self.logger.warning(f"GPU experiments completed with {oom_errors} OOM errors")

        return completed, failed


# ============================================================================
# CLI
# ============================================================================

def _orchestrator_main():
    """Core orchestrator entry point preserved from the original campaign.

    Called by :func:`main` after the subcommand layer has set the
    ``COOPETITION_REWARD_TYPE`` environment variable and injected safety
    defaults into ``sys.argv``. This function retains the full CLI of the
    original ``orchestrator.py`` for users who need direct access.
    """
    parser = argparse.ArgumentParser(
        description="Unified Coopetition-Gym MARL Baseline Orchestrator V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes (can be combined with comma):
    tr1     - Interdependence & Complementarity (5 environments)
    tr2     - Trust & Reputation Dynamics (5 environments)
    tr3     - Collective Action & Loyalty (5 environments)

Examples:
    # Single mode
    python orchestrator.py --mode tr1 --output results/tr1
    python orchestrator.py --mode tr2 --output results/tr2
    python orchestrator.py --mode tr3 --output results/tr3

    # Combined modes
    python orchestrator.py --mode tr1,tr2 --output results/tr12
    python orchestrator.py --mode tr1,tr3 --output results/tr13
    python orchestrator.py --mode tr2,tr3 --output results/tr23
    python orchestrator.py --mode tr1,tr2,tr3 --output results/all

    # Filter to specific algorithms
    python orchestrator.py --mode tr3 --algorithms IPPO,MAPPO,MADDPG

    # Resume interrupted run
    python orchestrator.py --mode tr1,tr2 --output results/tr12 --resume

    # Dry run to see what would execute
    python orchestrator.py --mode tr3 --dry-run

    # Lambda Cloud 8x A100 (RECOMMENDED)
    python orchestrator.py --mode tr1,tr2,tr3 --max-workers 48 --output results/all

Resource Management:
    The orchestrator implements dynamic resource management:
    - GPU memory tracking with bin-packing allocation
    - Runtime GPU monitoring via nvidia-smi
    - Dynamic backpressure to reduce workers on memory pressure
    - Automatic OOM recovery with worker reduction
    - Separate CPU pool for heuristics (no GPU waste)
    - Memory-aware queue sorting

Lambda Cloud 8x A100 Safe Configuration:
    --max-workers 48 provides safe operation:
    - 24 GPU workers (3 per A100, 6GB headroom each)
    - 24 CPU workers (for heuristics)
    - Backpressure threshold at 85%% memory
    - Critical threshold at 95%% memory
        """
    )

    # Mode and output
    parser.add_argument("--mode", type=str, default="tr1,tr2,tr3,tr4",
                        help="Modes to run: tr1, tr2, tr3, tr4, or combinations like tr1,tr2")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--algorithms", type=str, default=None,
                        help="Comma-separated list of algorithms to run")
    parser.add_argument("--environments", type=str, default=None,
                        help="Comma-separated list of environments to run")
    parser.add_argument("--seeds", type=str, default="100,101,102,103,104",
                        help="Comma-separated list of training seeds")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    parser.add_argument("--shuffle-seed", type=int, default=42,
                        help="Seed for experiment queue shuffling")

    # Worker limits (CRITICAL for Lambda Cloud cost control)
    worker_group = parser.add_argument_group('Worker Limits',
        'Control parallel workers to prevent OOM and optimize cost')
    worker_group.add_argument("--max-workers", type=int, default=None,
                        metavar="N",
                        help="Total worker cap (GPU + CPU). Recommended: 48 for Lambda Cloud 8x A100")
    worker_group.add_argument("--max-gpu-workers", type=int, default=None,
                        metavar="N",
                        help="GPU worker cap only. Recommended: 24 for 8x A100 (3 per GPU)")
    worker_group.add_argument("--max-cpu-workers", type=int, default=None,
                        metavar="N",
                        help="CPU worker cap only (for heuristics). Recommended: 24")

    # Monitoring options
    monitor_group = parser.add_argument_group('GPU Monitoring',
        'Runtime memory monitoring via nvidia-smi')
    monitor_group.add_argument("--no-monitoring", action="store_true",
                        help="Disable GPU memory monitoring")
    monitor_group.add_argument("--memory-poll-interval", type=float, default=5.0,
                        metavar="SECS",
                        help="GPU memory poll interval in seconds (default: 5.0)")
    monitor_group.add_argument("--memory-threshold", type=float, default=85.0,
                        metavar="PCT",
                        help="Backpressure threshold percentage (default: 85.0)")
    monitor_group.add_argument("--critical-threshold", type=float, default=95.0,
                        metavar="PCT",
                        help="Critical memory threshold percentage (default: 95.0)")

    # Backpressure options
    bp_group = parser.add_argument_group('Dynamic Backpressure',
        'Automatic worker reduction on memory pressure')
    bp_group.add_argument("--no-backpressure", action="store_true",
                        help="Disable dynamic backpressure")
    bp_group.add_argument("--backpressure-cooldown", type=float, default=30.0,
                        metavar="SECS",
                        help="Cooldown between backpressure reductions (default: 30.0)")

    # Thermal monitoring options
    thermal_group = parser.add_argument_group('Thermal Monitoring',
        'GPU temperature monitoring and throttling')
    thermal_group.add_argument("--no-thermal-monitoring", action="store_true",
                        help="Disable GPU thermal monitoring")
    thermal_group.add_argument("--thermal-throttle", type=float, default=80.0,
                        metavar="CELSIUS",
                        help="Temperature to trigger backpressure (default: 80°C)")
    thermal_group.add_argument("--thermal-critical", type=float, default=85.0,
                        metavar="CELSIUS",
                        help="Temperature to pause submissions (default: 85°C)")

    # RAM monitoring options
    ram_group = parser.add_argument_group('System RAM Monitoring',
        'Monitor system RAM usage (important for 1800GB Lambda Cloud)')
    ram_group.add_argument("--no-ram-monitoring", action="store_true",
                        help="Disable system RAM monitoring")
    ram_group.add_argument("--ram-threshold", type=float, default=80.0,
                        metavar="PCT",
                        help="RAM usage percentage to trigger backpressure (default: 80%%)")
    ram_group.add_argument("--ram-critical", type=float, default=90.0,
                        metavar="PCT",
                        help="RAM usage percentage to pause submissions (default: 90%%)")

    # Advanced features
    advanced_group = parser.add_argument_group('Advanced Features',
        'GPU isolation, adaptive buffers, priority scheduling, checkpoints')
    advanced_group.add_argument("--no-gpu-isolation", action="store_true",
                        help="Disable CUDA_VISIBLE_DEVICES isolation per process")
    advanced_group.add_argument("--no-adaptive-buffers", action="store_true",
                        help="Disable adaptive buffer reduction under memory pressure")
    advanced_group.add_argument("--no-priority-queue", action="store_true",
                        help="Disable priority-based experiment scheduling")
    advanced_group.add_argument("--prioritize-fast", action="store_true",
                        help="Run fast algorithms first (default: slow algorithms first)")
    advanced_group.add_argument("--no-nvlink-scheduling", action="store_true",
                        help="Disable NVLink-aware GPU scheduling")
    advanced_group.add_argument("--no-checkpoints", action="store_true",
                        help="Disable checkpoint saving. Default: enabled every 100k steps.")
    # Legacy flag kept for backward compatibility (the new default is "enabled").
    advanced_group.add_argument("--enable-checkpoints", action="store_true",
                        help=argparse.SUPPRESS)
    advanced_group.add_argument("--timesteps-override", type=int, default=None,
                        metavar="N",
                        help="Override TIMESTEPS_BY_CATEGORY for every training algorithm "
                             "to N. Propagated to worker subprocesses via the "
                             "COOPETITION_TIMESTEPS_OVERRIDE environment variable. "
                             "Primarily useful for smoke tests and reduced-budget "
                             "reproductions; not recommended for paper-scale runs.")
    advanced_group.add_argument("--checkpoint-interval", type=int, default=100000,
                        metavar="STEPS",
                        help="Steps between checkpoints (default: 100000)")
    advanced_group.add_argument("--checkpoint-dir", type=str, default=None,
                        metavar="DIR",
                        help="Directory for checkpoints (default: <output>/checkpoints)")

    args = parser.parse_args()

    # Validate worker limits
    if args.max_workers is not None and args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if args.max_gpu_workers is not None and args.max_gpu_workers < 0:
        parser.error("--max-gpu-workers must be non-negative")
    if args.max_cpu_workers is not None and args.max_cpu_workers < 1:
        parser.error("--max-cpu-workers must be at least 1")
    if args.memory_threshold >= args.critical_threshold:
        parser.error("--memory-threshold must be less than --critical-threshold")

    # Propagate --timesteps-override through the environment so multiprocessing
    # workers (which re-import this module fresh via spawn) see it. The worker
    # consults ``COOPETITION_TIMESTEPS_OVERRIDE`` inside ``run_single_experiment``
    # just before training to override the category-default budget.
    if args.timesteps_override is not None:
        if args.timesteps_override < 1:
            parser.error("--timesteps-override must be a positive integer")
        os.environ["COOPETITION_TIMESTEPS_OVERRIDE"] = str(args.timesteps_override)
        print(f"[campaign] COOPETITION_TIMESTEPS_OVERRIDE = {args.timesteps_override}")

    # Parse mode argument
    modes = [m.strip().lower() for m in args.mode.split(",")]
    valid_modes = {"tr1", "tr2", "tr3", "tr4"}
    for mode in modes:
        if mode not in valid_modes:
            print(f"Error: Invalid mode '{mode}'. Valid modes: {valid_modes}")
            sys.exit(1)

    # Parse other list arguments
    algorithms = args.algorithms.split(",") if args.algorithms else None
    environments = args.environments.split(",") if args.environments else None
    seeds = [int(s) for s in args.seeds.split(",")]

    # Log worker limit configuration
    if args.max_workers:
        print(f"[CONFIG] --max-workers {args.max_workers} set")
    if args.max_gpu_workers:
        print(f"[CONFIG] --max-gpu-workers {args.max_gpu_workers} set")
    if args.max_cpu_workers:
        print(f"[CONFIG] --max-cpu-workers {args.max_cpu_workers} set")

    config = OrchestratorConfig(
        modes=modes,
        output_dir=Path(args.output),
        algorithms=algorithms,
        environments=environments,
        seeds=seeds,
        n_eval_episodes=args.eval_episodes,
        resume=args.resume,
        dry_run=args.dry_run,
        shuffle_seed=args.shuffle_seed,

        # Worker limits
        max_workers=args.max_workers,
        max_gpu_workers=args.max_gpu_workers,
        max_cpu_workers=args.max_cpu_workers,

        # GPU memory monitoring options
        enable_memory_monitoring=not args.no_monitoring,
        memory_poll_interval=args.memory_poll_interval,
        memory_threshold_pct=args.memory_threshold,
        critical_threshold_pct=args.critical_threshold,

        # Backpressure options
        enable_backpressure=not args.no_backpressure,
        backpressure_cooldown=args.backpressure_cooldown,

        # Thermal monitoring options
        enable_thermal_monitoring=not args.no_thermal_monitoring,
        thermal_throttle_c=args.thermal_throttle,
        thermal_critical_c=args.thermal_critical,

        # RAM monitoring options
        enable_ram_monitoring=not args.no_ram_monitoring,
        ram_threshold_pct=args.ram_threshold,
        ram_critical_pct=args.ram_critical,

        # Advanced features
        enable_gpu_isolation=not args.no_gpu_isolation,
        enable_adaptive_buffers=not args.no_adaptive_buffers,
        enable_priority_queue=not args.no_priority_queue,
        prioritize_fast_algorithms=args.prioritize_fast,
        enable_nvlink_scheduling=not args.no_nvlink_scheduling,
        # Checkpoints are ON by default (safety). Explicit --no-checkpoints disables.
        # The legacy --enable-checkpoints flag is a no-op because the default is now ON.
        enable_checkpoints=not args.no_checkpoints,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
    )

    orchestrator = UnifiedOrchestrator(config)
    orchestrator.run()


# =============================================================================
# Subcommand layer — the public campaign CLI
# =============================================================================

def _verify_reward_patcher(expected_reward_type: str) -> None:
    """Verify that the reward-type patcher is installed in site-packages.

    The patcher (``reward_type_patcher.py`` + ``reward_type_patch.pth``)
    reads ``COOPETITION_REWARD_TYPE`` at Python startup and applies the
    reward type to every environment constructed via ``coopetition_gym.make``.
    It must be installed for non-``integrated`` campaigns to produce correct
    reward mutuality.

    Prints a warning and exits with code 1 if the patcher is missing and the
    expected reward type is not ``integrated``.
    """
    try:
        from coopetition_gym.envs import make as _test_make
        env = _test_make("TrustDilemma-v0")
        actual = getattr(env, "reward_type", "unknown")
        env.close()
        if actual == expected_reward_type:
            print(f"[campaign] reward patcher verified: env.reward_type = {actual}")
            return
        print(
            f"[campaign] WARNING: reward patcher may not be installed.\n"
            f"    Expected: {expected_reward_type}\n"
            f"    Actual:   {actual}\n"
            f"    Install ``reward_type_patcher.py`` + ``reward_type_patch.pth``\n"
            f"    into the venv's site-packages directory. See REPRODUCE.md."
        )
        if expected_reward_type != "integrated":
            print(f"[campaign] ABORTING — non-integrated campaign would produce wrong data.")
            sys.exit(1)
    except Exception as exc:
        print(f"[campaign] WARNING: could not verify reward patcher: {exc}")


def _run_reward_campaign(reward_type: str, forwarded_argv: list) -> None:
    """Run a reward-configured campaign: baseline, private, or cooperative.

    Sets ``COOPETITION_REWARD_TYPE`` before any further environment creation,
    verifies the patcher, then delegates to :func:`_orchestrator_main` with
    the forwarded argument vector.
    """
    os.environ["COOPETITION_REWARD_TYPE"] = reward_type
    print(f"[campaign] COOPETITION_REWARD_TYPE = {reward_type}")

    _verify_reward_patcher(reward_type)

    # Replace sys.argv so the downstream argparse sees the forwarded arguments.
    sys.argv = [sys.argv[0]] + forwarded_argv
    _orchestrator_main()


def main(argv: Optional[List[str]] = None) -> int:
    """Top-level CLI with campaign-type subcommands.

    Parses the subcommand (``baseline``, ``private``, ``cooperative``, or
    ``sensitivity``) and dispatches to the appropriate underlying
    orchestrator. All other arguments are forwarded to the underlying
    orchestrator unchanged.
    """
    if argv is None:
        argv = sys.argv[1:]

    # Valid campaign subcommands.
    subcommands = {"baseline", "private", "cooperative", "sensitivity"}

    if not argv or argv[0] in ("-h", "--help"):
        print(
            "Usage: python -m experiments.campaign SUBCOMMAND [OPTIONS]\n\n"
            "Subcommands:\n"
            "  baseline      Main experimental campaign (integrated reward)\n"
            "  private       Private-reward ablation (D_ij = 0)\n"
            "  cooperative   Cooperative-reward ablation (fully shared reward)\n"
            "  sensitivity   Network capacity sensitivity analysis\n\n"
            "Run 'python -m experiments.campaign SUBCOMMAND --help' for options.\n\n"
            "Safety defaults (all on): checkpoints, GPU monitoring, backpressure,\n"
            "thermal monitoring. See module docstring for details."
        )
        return 0

    subcommand = argv[0]
    if subcommand not in subcommands:
        # No subcommand recognized: fall back to the original orchestrator CLI
        # for backward compatibility with direct users of orchestrator.py.
        return _orchestrator_main() or 0

    forwarded = argv[1:]

    if subcommand == "baseline":
        _run_reward_campaign("integrated", forwarded)
    elif subcommand == "private":
        _run_reward_campaign("private", forwarded)
    elif subcommand == "cooperative":
        _run_reward_campaign("cooperative", forwarded)
    elif subcommand == "sensitivity":
        # Delegate to the dedicated sensitivity module.
        from experiments import sensitivity
        sys.argv = [sys.argv[0]] + forwarded
        sensitivity.main()
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main() or 0)