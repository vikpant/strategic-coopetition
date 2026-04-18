"""Single source of truth for experiment defaults.

This module defines every default value used by the experiment orchestration,
evaluation, behavioral audit, analysis, and validation modules. Every other
file in the ``experiments/`` package imports from here and does not redefine
these values.

The defaults in this file are the exact values used to produce the 25,708-file
training dataset and the 1,116-file behavioral audit dataset released with the
NeurIPS 2026 paper. Modifying a default does not change the released datasets.
Reproducing a paper result requires the default shown here.

Grouping:

* **Version** — package version string
* **Paths** — directory layout for results, checkpoints, logs
* **Seeds** — the seven training seeds and the three audit seeds
* **Reward types** — the three reward configurations for the ablation
* **Environments** — the 20 environments grouped by technical report
* **Algorithms** — 18 training + 7 oracle + 101 constant + 2 heuristic = 128
* **Oracles** — oracle-to-environment reference mapping for Gap% computation
* **Timesteps** — per-category training budgets
* **Audit** — cooperation sweep and temporal deviation parameters
* **Safety** — checkpoint, monitoring, and disk-pressure defaults
* **Sensitivity** — default network capacities for the sensitivity analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Version
# =============================================================================

VERSION = "1.0.0"
DATASET_VERSION = "v1"
NEURIPS_SUBMISSION_YEAR = 2026


# =============================================================================
# Paths
# =============================================================================

#: Repository root, computed relative to this file's location.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Default output directory for training result files.
DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "training"

#: Default output directory for behavioral audit JSON files.
DEFAULT_AUDIT_DIR = REPO_ROOT / "data" / "audit"

#: Default location for policy checkpoints during training.
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "data" / "checkpoints"

#: Default location for log files.
DEFAULT_LOG_DIR = REPO_ROOT / "data" / "logs"

#: Default location for analysis outputs (plots, CSV summaries).
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"


# =============================================================================
# Seeds
# =============================================================================

#: Seven seeds used for the full training campaign.
#: Every paper result uses these seeds. Changing them produces different data.
TRAINING_SEEDS: Tuple[int, ...] = (99, 100, 101, 102, 103, 104, 105)

#: Three seeds used for the behavioral audit.
#: The audit is algorithm-independent so three seeds are sufficient to establish
#: determinism; extending to seven adds no information.
AUDIT_SEEDS: Tuple[int, ...] = (99, 100, 101)


# =============================================================================
# Reward types
# =============================================================================

#: The three reward configurations ablated in the paper.
#:
#: * ``private`` — :math:`U_i = \pi_i` — agent rewarded only for own payoff.
#: * ``integrated`` — :math:`U_i = \pi_i + \sum_{j \ne i} D_{ij} \pi_j` —
#:   agent rewarded for own payoff plus weighted share of partner payoffs.
#: * ``cooperative`` — :math:`U_i = \sum_j \pi_j` — all agents receive the
#:   sum of payoffs (fully shared reward).
REWARD_TYPES: Tuple[str, ...] = ("private", "integrated", "cooperative")


# =============================================================================
# Environments
# =============================================================================

@dataclass(frozen=True)
class EnvironmentSpec:
    """Static specification of a coopetition-gym environment.

    Attributes:
        id: Gymnasium environment identifier ending in ``-v0``.
        horizon: Episode length in steps.
        category: Semantic category for timestep budget lookup.
        n_agents: Number of agents in the environment.
        tr: Source technical report (``tr1``, ``tr2``, ``tr3``, ``tr4``).
    """

    id: str
    horizon: int
    category: str
    n_agents: int
    tr: str


#: TR-1 environments — Interdependence and Complementarity (arXiv:2510.18802).
TR1_ENVIRONMENTS: Tuple[EnvironmentSpec, ...] = (
    EnvironmentSpec("PartnerHoldUp-v0",           100, "dyadic",    2, "tr1"),
    EnvironmentSpec("PlatformEcosystem-v0",       100, "ecosystem", 5, "tr1"),
    EnvironmentSpec("DynamicPartnerSelection-v0", 100, "ecosystem", 4, "tr1"),
    EnvironmentSpec("SynergySearch-v0",           100, "benchmark", 2, "tr1"),
    EnvironmentSpec("RenaultNissan-v0",            60, "validated", 2, "tr1"),
)

#: TR-2 environments — Trust Dynamics (arXiv:2510.24909).
TR2_ENVIRONMENTS: Tuple[EnvironmentSpec, ...] = (
    EnvironmentSpec("TrustDilemma-v0",          100, "dyadic",    2, "tr2"),
    EnvironmentSpec("RecoveryRace-v0",          150, "benchmark", 2, "tr2"),
    EnvironmentSpec("SLCD-v0",                   40, "validated", 2, "tr2"),
    EnvironmentSpec("CooperativeNegotiation-v0",100, "extended",  2, "tr2"),
    EnvironmentSpec("ReputationMarket-v0",      100, "extended",  2, "tr2"),
)

#: TR-3 environments — Collective Action and Loyalty (arXiv:2601.16237).
TR3_ENVIRONMENTS: Tuple[EnvironmentSpec, ...] = (
    EnvironmentSpec("TeamProduction-v0",     100, "collective_action", 4, "tr3"),
    EnvironmentSpec("LoyaltyTeam-v0",        100, "collective_action", 4, "tr3"),
    EnvironmentSpec("CoalitionFormation-v0", 150, "collective_action", 6, "tr3"),
    EnvironmentSpec("ApacheProject-v0",       60, "collective_action", 5, "tr3"),
    EnvironmentSpec("PublicGoods-v0",        100, "collective_action", 4, "tr3"),
)

#: TR-4 environments — Sequential Interaction and Reciprocity (arXiv:2604.01240).
TR4_ENVIRONMENTS: Tuple[EnvironmentSpec, ...] = (
    EnvironmentSpec("ReciprocalDilemma-v0",   100, "dyadic",      2, "tr4"),
    EnvironmentSpec("GiftExchange-v0",        100, "dyadic",      2, "tr4"),
    EnvironmentSpec("IndirectReciprocity-v0", 150, "reciprocity", 4, "tr4"),
    EnvironmentSpec("GraduatedSanction-v0",   200, "reciprocity", 6, "tr4"),
    EnvironmentSpec("AppleAppStore-v0",        66, "reciprocity", 3, "tr4"),
)

#: All 20 environments combined.
ALL_ENVIRONMENTS: Tuple[EnvironmentSpec, ...] = (
    TR1_ENVIRONMENTS + TR2_ENVIRONMENTS + TR3_ENVIRONMENTS + TR4_ENVIRONMENTS
)

#: Environment lookup by technical report tier.
ENVIRONMENTS_BY_TR: Dict[str, Tuple[EnvironmentSpec, ...]] = {
    "tr1": TR1_ENVIRONMENTS,
    "tr2": TR2_ENVIRONMENTS,
    "tr3": TR3_ENVIRONMENTS,
    "tr4": TR4_ENVIRONMENTS,
}

#: Environment lookup by ID.
ENVIRONMENT_BY_ID: Dict[str, EnvironmentSpec] = {
    env.id: env for env in ALL_ENVIRONMENTS
}


# =============================================================================
# Timesteps
# =============================================================================

#: Training timesteps per environment category.
#:
#: All categories normalized to approximately 250,000 steps per agent.
#: Dyadic (2 agents) uses 500K total, ecosystem (4-5 agents) uses 1M, etc.
TIMESTEPS_BY_CATEGORY: Dict[str, int] = {
    "dyadic":            500_000,
    "ecosystem":       1_000_000,
    "benchmark":         500_000,
    "validated":         500_000,
    "extended":          500_000,
    "collective_action": 1_000_000,
    "reciprocity":      1_000_000,
}


# =============================================================================
# Algorithms
# =============================================================================

@dataclass(frozen=True)
class AlgorithmSpec:
    """Static specification of a training, oracle, or heuristic algorithm.

    Attributes:
        name: Unique algorithm identifier used in output filenames.
        class_name: Python class name in the algorithms module.
        requires_training: Whether the algorithm needs a training phase before evaluation.
        gpu_memory_gb: Estimated peak GPU memory in GB (0.0 for CPU-only).
        cpu_only: True if the algorithm never uses a GPU.
        speed: Rough wall-clock category (``fast``, ``medium``, ``slow``).
        params: Hyperparameters passed to the algorithm constructor.
        applicable_trs: Optional restriction to specific TR tiers.
            When None, the algorithm applies to all tiers.
        applicable_categories: Optional restriction to specific environment categories.
            When None, the algorithm applies to all categories.
    """

    name: str
    class_name: str
    requires_training: bool
    gpu_memory_gb: float
    cpu_only: bool
    speed: str
    params: Dict[str, object] = field(default_factory=dict)
    applicable_trs: Optional[Tuple[str, ...]] = None
    applicable_categories: Optional[Tuple[str, ...]] = None


# -- Training algorithms (18) -------------------------------------------------

#: Independent learners (no centralized critic during training).
#: FCP is classified here (not CTDE) because each agent independently trains
#: against a population of opponent checkpoints without a shared central critic.
INDEPENDENT_LEARNING_ALGORITHMS: Tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        name="IPPO", class_name="IndependentPPO",
        requires_training=True, gpu_memory_gb=0.0, cpu_only=True, speed="fast",
        params={"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,
                "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95,
                "clip_range": 0.2, "ent_coef": 0.01, "vf_coef": 0.5,
                "max_grad_norm": 0.5, "net_arch": [128, 128]},
    ),
    AlgorithmSpec(
        name="IA2C", class_name="IndependentA2C",
        requires_training=True, gpu_memory_gb=0.0, cpu_only=True, speed="fast",
        params={"learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99,
                "gae_lambda": 1.0, "ent_coef": 0.01, "vf_coef": 0.5,
                "max_grad_norm": 0.5, "net_arch": [128, 128]},
    ),
    AlgorithmSpec(
        name="IndependentREINFORCE", class_name="IndependentREINFORCE",
        requires_training=True, gpu_memory_gb=0.0, cpu_only=True, speed="fast",
        params={"learning_rate": 1e-3, "gamma": 0.99, "net_arch": [128, 128]},
    ),
    AlgorithmSpec(
        name="ISAC", class_name="IndependentSAC",
        requires_training=True, gpu_memory_gb=3.0, cpu_only=False, speed="medium",
        params={"learning_rate": 3e-4, "buffer_size": 100_000, "batch_size": 256,
                "tau": 0.005, "gamma": 0.99, "net_arch": [128, 128]},
    ),
    AlgorithmSpec(
        name="LOLA", class_name="LOLA",
        requires_training=True, gpu_memory_gb=0.0, cpu_only=True, speed="medium",
        params={"learning_rate": 1e-3, "opponent_lr": 1e-3, "n_lookahead": 1,
                "gamma": 0.99, "net_arch": [128, 128]},
    ),
    AlgorithmSpec(
        name="SelfPlay_PPO", class_name="SelfPlayPPO",
        requires_training=True, gpu_memory_gb=0.0, cpu_only=True, speed="slow",
        params={"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,
                "gamma": 0.99, "opponent_update_freq": 10_000,
                "net_arch": [128, 128]},
    ),
    AlgorithmSpec(
        name="FCP", class_name="FictitiousCoPlay",
        requires_training=True, gpu_memory_gb=0.0, cpu_only=True, speed="slow",
        params={"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,
                "gamma": 0.99, "checkpoint_freq": 50_000,
                "sample_recent_prob": 0.5, "net_arch": [128, 128]},
    ),
)

#: CTDE learners (centralized training with decentralized execution).
CTDE_ALGORITHMS: Tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        name="MAPPO", class_name="MAPPO",
        requires_training=True, gpu_memory_gb=0.0, cpu_only=True, speed="medium",
        params={"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,
                "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95,
                "clip_range": 0.2, "ent_coef": 0.01, "share_critic": True,
                "net_arch": [128, 128]},
    ),
    AlgorithmSpec(
        name="MADDPG", class_name="MADDPG",
        requires_training=True, gpu_memory_gb=4.0, cpu_only=False, speed="slow",
        params={"learning_rate_actor": 1e-4, "learning_rate_critic": 1e-3,
                "buffer_size": 100_000, "batch_size": 256, "tau": 0.005,
                "gamma": 0.99, "net_arch": [128, 128]},
    ),
    AlgorithmSpec(
        name="MATD3", class_name="MATD3",
        requires_training=True, gpu_memory_gb=4.0, cpu_only=False, speed="slow",
        params={"learning_rate_actor": 1e-4, "learning_rate_critic": 1e-3,
                "buffer_size": 100_000, "batch_size": 256, "tau": 0.005,
                "gamma": 0.99, "policy_noise": 0.2, "noise_clip": 0.5,
                "policy_delay": 2, "net_arch": [128, 128]},
    ),
    AlgorithmSpec(
        name="MASAC", class_name="MASAC",
        requires_training=True, gpu_memory_gb=4.0, cpu_only=False, speed="slow",
        params={"learning_rate": 3e-4, "buffer_size": 100_000, "batch_size": 256,
                "tau": 0.005, "gamma": 0.99, "net_arch": [128, 128]},
    ),
    AlgorithmSpec(
        name="M3DDPG", class_name="M3DDPG",
        requires_training=True, gpu_memory_gb=4.0, cpu_only=False, speed="slow",
        params={"learning_rate_actor": 1e-4, "learning_rate_critic": 1e-3,
                "buffer_size": 100_000, "batch_size": 256, "gamma": 0.99,
                "minimax_weight": 0.5, "net_arch": [128, 128]},
    ),
    AlgorithmSpec(
        name="QMIX", class_name="QMIX",
        requires_training=True, gpu_memory_gb=2.5, cpu_only=False, speed="medium",
        params={"learning_rate": 5e-4, "buffer_size": 5_000, "batch_size": 32,
                "gamma": 0.99, "action_bins": 11},
    ),
    AlgorithmSpec(
        name="VDN", class_name="VDN",
        requires_training=True, gpu_memory_gb=2.0, cpu_only=False, speed="fast",
        params={"learning_rate": 5e-4, "buffer_size": 5_000, "batch_size": 32,
                "gamma": 0.99, "action_bins": 11},
    ),
    AlgorithmSpec(
        name="COMA", class_name="COMA",
        requires_training=True, gpu_memory_gb=1.5, cpu_only=False, speed="fast",
        params={"learning_rate": 5e-4, "gamma": 0.99},
    ),
    AlgorithmSpec(
        # MeanFieldAC is restricted to N>=3 environments.
        # Yang et al. (2018) mean-field approximation replaces joint opponent
        # effect with population average, which is degenerate for N=2.
        name="MeanFieldAC", class_name="MeanFieldActorCritic",
        requires_training=True, gpu_memory_gb=0.0, cpu_only=True, speed="slow",
        applicable_categories=("ecosystem", "collective_action", "reciprocity"),
        params={"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048,
                "net_arch": [128, 128]},
    ),
)

TRAINING_ALGORITHMS: Tuple[AlgorithmSpec, ...] = (
    INDEPENDENT_LEARNING_ALGORITHMS + CTDE_ALGORITHMS
)

# -- Oracle algorithms (7) ----------------------------------------------------

ORACLE_ALGORITHMS: Tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        name="Oracle_Equilibrium", class_name="CoopetitiveEquilibriumOracle",
        requires_training=False, gpu_memory_gb=0.0, cpu_only=True, speed="fast",
        applicable_trs=("tr1",),
    ),
    AlgorithmSpec(
        name="Oracle_TrustAware", class_name="TrustAwareEquilibriumOracle",
        requires_training=False, gpu_memory_gb=0.0, cpu_only=True, speed="fast",
        applicable_trs=("tr2",),
    ),
    AlgorithmSpec(
        name="Oracle_Nash", class_name="NashEquilibriumOracle",
        requires_training=False, gpu_memory_gb=0.0, cpu_only=True, speed="fast",
        applicable_trs=("tr3",),
    ),
    AlgorithmSpec(
        name="Oracle_SocialOptimum", class_name="SocialOptimumOracle",
        requires_training=False, gpu_memory_gb=0.0, cpu_only=True, speed="fast",
        applicable_trs=("tr3",),
    ),
    AlgorithmSpec(
        name="Oracle_Loyalty", class_name="LoyaltyAugmentedOracle",
        requires_training=False, gpu_memory_gb=0.0, cpu_only=True, speed="fast",
        applicable_trs=("tr3",),
    ),
    AlgorithmSpec(
        name="Oracle_ReciprocityEquilibrium", class_name="ReciprocityEquilibriumOracle",
        requires_training=False, gpu_memory_gb=0.0, cpu_only=True, speed="fast",
        applicable_trs=("tr4",),
    ),
    AlgorithmSpec(
        name="Oracle_BoundedReciprocity", class_name="BoundedReciprocityOracle",
        requires_training=False, gpu_memory_gb=0.0, cpu_only=True, speed="fast",
        applicable_trs=("tr4",),
    ),
)

# -- Heuristic algorithms -----------------------------------------------------

HEURISTIC_ALGORITHMS: Tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        name="Random", class_name="RandomPolicy",
        requires_training=False, gpu_memory_gb=0.0, cpu_only=True, speed="fast",
    ),
    AlgorithmSpec(
        name="TitForTat", class_name="TitForTatPolicy",
        requires_training=False, gpu_memory_gb=0.0, cpu_only=True, speed="fast",
    ),
)

# -- Constant cooperation policies (101) --------------------------------------

def _constant_policy(level: float) -> AlgorithmSpec:
    """Build a constant-action policy spec that contributes ``level`` fraction
    of endowment at every step."""
    return AlgorithmSpec(
        name=f"Constant_{int(round(level * 100)):02d}",
        class_name="ConstantPolicy",
        requires_training=False, gpu_memory_gb=0.0, cpu_only=True, speed="fast",
        params={"level": round(level, 2)},
    )

#: 101 constant-cooperation policies from 0% to 100% in 1% increments.
#: Used for fine-grained monotonicity analysis of the cooperation-return relationship.
CONSTANT_ALGORITHMS: Tuple[AlgorithmSpec, ...] = tuple(
    _constant_policy(i / 100.0) for i in range(101)
)

# -- All algorithms -----------------------------------------------------------

ALL_ALGORITHMS: Tuple[AlgorithmSpec, ...] = (
    TRAINING_ALGORITHMS + ORACLE_ALGORITHMS + HEURISTIC_ALGORITHMS + CONSTANT_ALGORITHMS
)

#: Algorithm lookup by name.
ALGORITHM_BY_NAME: Dict[str, AlgorithmSpec] = {
    algo.name: algo for algo in ALL_ALGORITHMS
}


# =============================================================================
# Oracle references for Gap% computation
# =============================================================================

#: Reference oracle per environment. Gap% is computed against this oracle.
#:
#: Formula: ``Gap = (Algorithm_return - Oracle_reference_return) / abs(Oracle_reference_return) * 100``.
#: Positive values indicate the algorithm exceeds the oracle reference.
#: TR-3 uses Oracle_Loyalty (upper bound). TR-4 uses Oracle_BoundedReciprocity (upper bound).
ENV_ORACLE_REF: Dict[str, str] = {
    "TrustDilemma-v0":            "Oracle_TrustAware",
    "PartnerHoldUp-v0":           "Oracle_Equilibrium",
    "PlatformEcosystem-v0":       "Oracle_Equilibrium",
    "DynamicPartnerSelection-v0": "Oracle_Equilibrium",
    "SynergySearch-v0":           "Oracle_Equilibrium",
    "RecoveryRace-v0":            "Oracle_TrustAware",
    "CooperativeNegotiation-v0":  "Oracle_TrustAware",
    "ReputationMarket-v0":        "Oracle_TrustAware",
    "SLCD-v0":                    "Oracle_TrustAware",
    "RenaultNissan-v0":           "Oracle_Equilibrium",
    "ApacheProject-v0":           "Oracle_Loyalty",
    "CoalitionFormation-v0":      "Oracle_Loyalty",
    "LoyaltyTeam-v0":             "Oracle_Loyalty",
    "PublicGoods-v0":             "Oracle_Loyalty",
    "TeamProduction-v0":          "Oracle_Loyalty",
    "ReciprocalDilemma-v0":       "Oracle_BoundedReciprocity",
    "GiftExchange-v0":            "Oracle_BoundedReciprocity",
    "IndirectReciprocity-v0":     "Oracle_BoundedReciprocity",
    "GraduatedSanction-v0":       "Oracle_BoundedReciprocity",
    "AppleAppStore-v0":           "Oracle_BoundedReciprocity",
}

#: Per-tier oracle groupings for comparison tables.
#: First entry is the primary reference; subsequent entries are alternative bounds.
TIER_ORACLES: Dict[str, Tuple[str, ...]] = {
    "tr1": ("Oracle_Equilibrium", "Oracle_TrustAware"),
    "tr2": ("Oracle_TrustAware", "Oracle_Equilibrium"),
    "tr3": ("Oracle_Nash", "Oracle_Loyalty", "Oracle_SocialOptimum"),
    "tr4": ("Oracle_ReciprocityEquilibrium", "Oracle_BoundedReciprocity"),
}


# =============================================================================
# Behavioral audit defaults
# =============================================================================

@dataclass(frozen=True)
class StaticAuditConfig:
    """Defaults for the static response-surface audit.

    The audit sweeps uniform cooperation from 0% to 100% in 5% increments
    (21 levels) and tests unilateral deviation at four cooperation levels
    (20%, 40%, 60%, 80%) by reducing agent 0's contribution by 50%.
    """

    #: Number of cooperation levels in the sweep (21 yields 0%, 5%, ..., 100%).
    n_cooperation_levels: int = 21

    #: Number of episodes per cooperation level.
    episodes_per_level: int = 5

    #: Cooperation levels at which exploitation is tested.
    exploitation_test_levels: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)

    #: Agent 0's contribution is reduced to (1 - deviation_fraction) * baseline.
    deviation_fraction: float = 0.5


@dataclass(frozen=True)
class TemporalAuditConfig:
    """Defaults for the temporal deviation audit.

    Tests five temporal strategies per (environment, seed):

    1. Full defection throughout
    2. Binary late defection at 9 switchpoints
    3. Early defection for 10-30% of the episode
    4. Gradual ramp-down over the final 20%
    5. Last-step-only defection (implicit in #2 with switchpoint = T-1)
    """

    #: Baseline cooperation level as a fraction of endowment.
    baseline_coop_fraction: float = 0.5

    #: Defection action (fraction of endowment).
    defect_action_fraction: float = 0.0

    #: Late-defection switchpoints as fractions of episode length.
    switchpoint_fractions: Tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

    #: Additional terminal switchpoints expressed as offsets from episode end.
    terminal_offsets: Tuple[int, ...] = (5, 3, 1)

    #: Early-defection durations as fractions of episode length.
    early_defect_fractions: Tuple[float, ...] = (0.10, 0.20, 0.30)

    #: Fraction of episode over which gradual ramp-down occurs.
    gradual_rampdown_fraction: float = 0.20

    #: Episodes per strategy per seed.
    episodes_per_strategy: int = 10


STATIC_AUDIT = StaticAuditConfig()
TEMPORAL_AUDIT = TemporalAuditConfig()


# =============================================================================
# Safety defaults
# =============================================================================

@dataclass(frozen=True)
class SafetyConfig:
    """Safety defaults for long-running campaigns.

    All defaults are opt-out only: disabling them requires an explicit flag.
    These values reflect lessons learned from the NeurIPS campaign where
    lack of checkpoints caused 30-40 GPU-hours of lost work and disk
    pressure filled instances three times.
    """

    #: Whether to save policy checkpoints during training.
    enable_checkpoints: bool = True

    #: Save a checkpoint every N environment steps.
    checkpoint_interval: int = 100_000

    #: Keep only the latest checkpoint per experiment (delete older ones).
    checkpoint_rotation: bool = True

    #: Alert and clean old checkpoints when disk usage exceeds this fraction.
    disk_pressure_threshold: float = 0.80

    #: Whether algorithms must emit progress at least every N seconds.
    require_progress_reporting: bool = True

    #: Interval between progress updates (seconds).
    progress_interval_seconds: int = 60

    #: Whether to verify hyperparameters match the canonical config before launch.
    verify_params_before_launch: bool = True

    #: Whether to verify disk capacity is sufficient before launch.
    verify_disk_before_launch: bool = True

    #: Whether to run a smoke test (1,000 steps, 1 experiment) before full launch.
    smoke_test_before_launch: bool = True


SAFETY = SafetyConfig()


# =============================================================================
# Network sensitivity analysis defaults
# =============================================================================

#: Network capacity configurations for the sensitivity analysis.
#: Each entry is a list of hidden layer sizes passed as ``net_arch``.
#: The default ``[128, 128]`` used in the main campaign is included.
SENSITIVITY_NET_SIZES: Tuple[Tuple[int, ...], ...] = (
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
)


# =============================================================================
# Campaign types
# =============================================================================

@dataclass(frozen=True)
class CampaignType:
    """Describes a campaign type that ``campaign.py`` can execute."""

    name: str
    description: str
    default_reward_types: Tuple[str, ...]
    default_environments: Tuple[str, ...]
    default_algorithms: Tuple[str, ...]


CAMPAIGN_BASELINE = CampaignType(
    name="baseline",
    description="Main experimental campaign with integrated reward.",
    default_reward_types=("integrated",),
    default_environments=tuple(env.id for env in ALL_ENVIRONMENTS),
    default_algorithms=tuple(algo.name for algo in TRAINING_ALGORITHMS),
)

CAMPAIGN_PRIVATE = CampaignType(
    name="private",
    description="Private-reward ablation (D_ij = 0).",
    default_reward_types=("private",),
    default_environments=tuple(env.id for env in ALL_ENVIRONMENTS),
    default_algorithms=tuple(algo.name for algo in TRAINING_ALGORITHMS),
)

CAMPAIGN_COOPERATIVE = CampaignType(
    name="cooperative",
    description="Cooperative-reward ablation (fully shared reward).",
    default_reward_types=("cooperative",),
    default_environments=tuple(env.id for env in ALL_ENVIRONMENTS),
    default_algorithms=tuple(algo.name for algo in TRAINING_ALGORITHMS),
)

CAMPAIGN_SENSITIVITY = CampaignType(
    name="sensitivity",
    description="Network capacity sensitivity analysis.",
    default_reward_types=("integrated", "private"),
    default_environments=tuple(env.id for env in ALL_ENVIRONMENTS),
    default_algorithms=("ISAC", "COMA", "QMIX", "MADDPG"),
)

ALL_CAMPAIGN_TYPES: Tuple[CampaignType, ...] = (
    CAMPAIGN_BASELINE,
    CAMPAIGN_PRIVATE,
    CAMPAIGN_COOPERATIVE,
    CAMPAIGN_SENSITIVITY,
)

CAMPAIGN_BY_NAME: Dict[str, CampaignType] = {
    campaign.name: campaign for campaign in ALL_CAMPAIGN_TYPES
}


# =============================================================================
# Dataset release
# =============================================================================

#: HuggingFace Hub repository identifiers for the released datasets.
#: These are stable identifiers used in REPRODUCE.md and paper supplementary.
HUGGINGFACE_TRAINING_DATASET = "vikpant/coopetition-gym-v1"
HUGGINGFACE_AUDIT_DATASET = "vikpant/coopetition-gym-audit"

#: Expected file counts in the released datasets.
#: Deviations from these counts during reproduction indicate missing data.
EXPECTED_TRAINING_FILES = 25_708
EXPECTED_AUDIT_STATIC_FILES = 1_056
EXPECTED_AUDIT_TEMPORAL_FILES = 60
EXPECTED_AUDIT_FILES = EXPECTED_AUDIT_STATIC_FILES + EXPECTED_AUDIT_TEMPORAL_FILES


# =============================================================================
# Convenience helpers
# =============================================================================

def algorithms_for_environment(env_id: str) -> List[AlgorithmSpec]:
    """Return the algorithms applicable to the given environment.

    Filters ``ALL_ALGORITHMS`` by ``applicable_trs`` and ``applicable_categories``
    restrictions. For example, ``MeanFieldAC`` is excluded from 2-agent
    environments and TR-specific oracles are excluded from non-matching tiers.
    """
    env = ENVIRONMENT_BY_ID.get(env_id)
    if env is None:
        raise KeyError(f"Unknown environment: {env_id}")

    applicable = []
    for algo in ALL_ALGORITHMS:
        if algo.applicable_trs is not None and env.tr not in algo.applicable_trs:
            continue
        if algo.applicable_categories is not None and env.category not in algo.applicable_categories:
            continue
        applicable.append(algo)
    return applicable


def environments_for_algorithm(algorithm_name: str) -> List[EnvironmentSpec]:
    """Return the environments on which the given algorithm is evaluated.

    The inverse of :func:`algorithms_for_environment`.
    """
    algo = ALGORITHM_BY_NAME.get(algorithm_name)
    if algo is None:
        raise KeyError(f"Unknown algorithm: {algorithm_name}")

    applicable = []
    for env in ALL_ENVIRONMENTS:
        if algo.applicable_trs is not None and env.tr not in algo.applicable_trs:
            continue
        if algo.applicable_categories is not None and env.category not in algo.applicable_categories:
            continue
        applicable.append(env)
    return applicable


def timesteps_for_environment(env_id: str) -> int:
    """Return the canonical training budget for an environment in steps."""
    env = ENVIRONMENT_BY_ID.get(env_id)
    if env is None:
        raise KeyError(f"Unknown environment: {env_id}")
    return TIMESTEPS_BY_CATEGORY[env.category]


def oracle_reference(env_id: str) -> str:
    """Return the reference oracle name used for Gap% computation."""
    if env_id not in ENV_ORACLE_REF:
        raise KeyError(f"No oracle reference defined for: {env_id}")
    return ENV_ORACLE_REF[env_id]
