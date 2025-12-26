"""
================================================================================
COOPETITION-GYM: Interdependence Module
================================================================================

This module implements the interdependence coefficient calculations from TR-1
(arXiv:2510.18802). Interdependence captures how much one agent's welfare 
depends on another's actions, derived from structural dependency analysis.

Mathematical Foundation:
------------------------
From Pant & Yu (2025), interdependence is computed as:

Equation 1: D_ij = Σ_d (w_d · Dep(i,j,d) · crit(i,j,d)) / Σ_d w_d

Where:
- w_d = importance weight for dependum d
- Dep(i,j,d) ∈ {0,1} = whether actor i depends on actor j for d
- crit(i,j,d) ∈ [0,1] = criticality of j for providing d

The interdependence matrix D captures the asymmetric nature of coopetitive
relationships - Sony may depend heavily on Samsung's manufacturing while
Samsung depends moderately on Sony's capital.

Key Validated Values (TR-1 §8 S-LCD Case):
-----------------------------------------
- D_Sony,Samsung = 0.86 (Sony highly dependent on Samsung's manufacturing)
- D_Samsung,Sony = 0.64 (Samsung moderately dependent on Sony's capital)

Authors: Vik Pant, Eric Yu
         Faculty of Information, University of Toronto
License: MIT
================================================================================
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """
    Types of dependencies in the i* framework.
    
    The i* (i-star) framework distinguishes between different types of
    dependencies based on what is being depended upon:
    - GOAL: Desired state of affairs in the world
    - TASK: Particular way of doing something  
    - RESOURCE: Physical or informational entity
    - SOFTGOAL: Quality or non-functional requirement
    """
    GOAL = "goal"
    TASK = "task"
    RESOURCE = "resource"
    SOFTGOAL = "softgoal"


@dataclass
class Dependency:
    """
    Represents a single dependency relationship between actors.
    
    In the i* framework, a dependency is a relationship where a depender
    relies on a dependee to provide some dependum (the thing being 
    depended upon).
    
    Attributes:
        depender: Name/ID of the actor who depends
        dependee: Name/ID of the actor being depended upon
        dependum: Description of what is being depended upon
        dep_type: Type of the dependency (goal, task, resource, softgoal)
        importance: Weight w_d indicating how important this dependency is [0,1]
        criticality: How critical the dependee is for providing the dependum [0,1]
    
    Example:
        >>> dep = Dependency(
        ...     depender="Sony",
        ...     dependee="Samsung", 
        ...     dependum="LCD Panel Manufacturing",
        ...     dep_type=DependencyType.RESOURCE,
        ...     importance=0.9,
        ...     criticality=0.95
        ... )
    """
    depender: str
    dependee: str
    dependum: str
    dep_type: DependencyType = DependencyType.RESOURCE
    importance: float = 1.0
    criticality: float = 1.0
    
    def __post_init__(self):
        """Validate attribute bounds."""
        if not 0 <= self.importance <= 1:
            raise ValueError(f"Importance must be in [0,1], got {self.importance}")
        if not 0 <= self.criticality <= 1:
            raise ValueError(f"Criticality must be in [0,1], got {self.criticality}")


@dataclass
class InterdependenceMatrix:
    """
    Encapsulates the interdependence matrix and related computations.
    
    The interdependence matrix D is an NxN matrix where D[i,j] represents
    how much agent i's utility depends on agent j's welfare. This captures
    the structural dependencies between coopetitive actors.
    
    Attributes:
        matrix: NxN numpy array of interdependence coefficients
        agent_names: Optional list of agent names for lookup
        metadata: Optional dictionary of additional information
    
    Properties:
        - D[i,i] = 0 (self-dependency is undefined)
        - D[i,j] ∈ [0,1] (normalized dependency strength)
        - D[i,j] ≠ D[j,i] in general (asymmetric)
    """
    matrix: NDArray[np.floating]
    agent_names: Optional[List[str]] = None
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate matrix properties."""
        if self.matrix.ndim != 2:
            raise ValueError("Interdependence matrix must be 2D")
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Interdependence matrix must be square")
        
        # Ensure diagonal is zero (no self-dependency)
        np.fill_diagonal(self.matrix, 0.0)
        
        # Set agent names if not provided
        if self.agent_names is None:
            self.agent_names = [f"Agent_{i}" for i in range(self.n_agents)]
    
    @property
    def n_agents(self) -> int:
        """Number of agents in the system."""
        return self.matrix.shape[0]
    
    def get_dependency(self, i: Union[int, str], j: Union[int, str]) -> float:
        """
        Get the interdependence coefficient D[i,j].
        
        Args:
            i: Depender (row index or name)
            j: Dependee (column index or name)
        
        Returns:
            Interdependence coefficient D[i,j]
        """
        i_idx = self._resolve_index(i)
        j_idx = self._resolve_index(j)
        return float(self.matrix[i_idx, j_idx])
    
    def _resolve_index(self, idx: Union[int, str]) -> int:
        """Convert name to index if necessary."""
        if isinstance(idx, str):
            if self.agent_names is None:
                raise ValueError("Agent names not set")
            return self.agent_names.index(idx)
        return idx
    
    def total_dependency(self, agent: Union[int, str]) -> float:
        """
        Compute total dependency of an agent on all others.
        
        This represents how much agent i depends on the collective,
        computed as Σ_j D[i,j].
        
        Args:
            agent: Agent index or name
        
        Returns:
            Sum of outgoing dependencies
        """
        i = self._resolve_index(agent)
        return float(np.sum(self.matrix[i, :]))
    
    def total_dependability(self, agent: Union[int, str]) -> float:
        """
        Compute how much others depend on a specific agent.
        
        This represents agent j's importance to the collective,
        computed as Σ_i D[i,j].
        
        Args:
            agent: Agent index or name
        
        Returns:
            Sum of incoming dependencies
        """
        j = self._resolve_index(agent)
        return float(np.sum(self.matrix[:, j]))
    
    def asymmetry(self, i: Union[int, str], j: Union[int, str]) -> float:
        """
        Compute the dependency asymmetry between two agents.
        
        Positive values indicate i depends more on j than vice versa.
        This asymmetry is a key driver of bargaining dynamics.
        
        Args:
            i: First agent
            j: Second agent
        
        Returns:
            D[i,j] - D[j,i]
        """
        i_idx = self._resolve_index(i)
        j_idx = self._resolve_index(j)
        return float(self.matrix[i_idx, j_idx] - self.matrix[j_idx, i_idx])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "matrix": self.matrix.tolist(),
            "agent_names": self.agent_names,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "InterdependenceMatrix":
        """Create from dictionary representation."""
        return cls(
            matrix=np.array(data["matrix"]),
            agent_names=data.get("agent_names"),
            metadata=data.get("metadata", {})
        )


def compute_interdependence_coefficient(
    dependencies: List[Dependency],
    depender: str,
    dependee: str
) -> float:
    """
    Compute the interdependence coefficient D_ij from a list of dependencies.
    
    This implements TR-1 Equation 1:
    D_ij = Σ_d (w_d · Dep(i,j,d) · crit(i,j,d)) / Σ_d w_d
    
    The coefficient aggregates all dependencies from depender i to dependee j,
    weighting each by its importance and criticality.
    
    Args:
        dependencies: List of all dependency relationships
        depender: Name of the depending actor (i)
        dependee: Name of the actor being depended upon (j)
    
    Returns:
        Normalized interdependence coefficient D_ij ∈ [0, 1]
    
    Example:
        >>> deps = [
        ...     Dependency("Sony", "Samsung", "Panels", importance=0.9, criticality=0.95),
        ...     Dependency("Sony", "Samsung", "Tech", importance=0.7, criticality=0.80),
        ... ]
        >>> compute_interdependence_coefficient(deps, "Sony", "Samsung")
        0.88...
    """
    # Filter dependencies from depender to dependee
    relevant_deps = [
        d for d in dependencies 
        if d.depender == depender and d.dependee == dependee
    ]
    
    if not relevant_deps:
        return 0.0
    
    # Compute weighted sum: Σ (w_d · crit_d)
    weighted_sum = sum(d.importance * d.criticality for d in relevant_deps)
    
    # Compute total weight: Σ w_d
    total_weight = sum(d.importance for d in relevant_deps)
    
    if total_weight == 0:
        return 0.0
    
    # Normalize by total weight
    coefficient = weighted_sum / total_weight
    
    logger.debug(
        f"D[{depender},{dependee}] = {coefficient:.4f} "
        f"(from {len(relevant_deps)} dependencies)"
    )
    
    return coefficient


def build_interdependence_matrix(
    dependencies: List[Dependency],
    agent_names: List[str]
) -> InterdependenceMatrix:
    """
    Build the complete interdependence matrix from dependency specifications.
    
    This constructs the NxN matrix D where each entry D[i,j] is computed
    using the interdependence coefficient formula from TR-1.
    
    Args:
        dependencies: Complete list of dependency relationships
        agent_names: Ordered list of agent names
    
    Returns:
        InterdependenceMatrix with computed coefficients
    
    Example:
        >>> deps = [
        ...     Dependency("Sony", "Samsung", "Panels", criticality=0.95),
        ...     Dependency("Samsung", "Sony", "Capital", criticality=0.70),
        ... ]
        >>> matrix = build_interdependence_matrix(deps, ["Samsung", "Sony"])
        >>> matrix.get_dependency("Sony", "Samsung")
        0.95
    """
    n_agents = len(agent_names)
    matrix = np.zeros((n_agents, n_agents))
    
    for i, depender in enumerate(agent_names):
        for j, dependee in enumerate(agent_names):
            if i != j:  # No self-dependency
                matrix[i, j] = compute_interdependence_coefficient(
                    dependencies, depender, dependee
                )
    
    return InterdependenceMatrix(
        matrix=matrix,
        agent_names=agent_names,
        metadata={"n_dependencies": len(dependencies)}
    )


# =============================================================================
# Predefined Interdependence Matrices for Case Studies
# =============================================================================

def create_slcd_interdependence() -> InterdependenceMatrix:
    """
    Create the interdependence matrix for Samsung-Sony S-LCD joint venture.
    
    From TR-1 Section 8.2, the S-LCD case involves:
    - Samsung: Contributes manufacturing capabilities, holds 50%+1 equity
    - Sony: Contributes capital investment and guaranteed offtake
    
    Validated interdependence coefficients:
    - D[Sony, Samsung] = 0.86 (Sony highly dependent on Samsung's manufacturing)
    - D[Samsung, Sony] = 0.64 (Samsung moderately dependent on Sony)
    
    Returns:
        InterdependenceMatrix for S-LCD case study
    """
    # Agent order: [Samsung, Sony]
    # Matrix[i,j] = how much agent i depends on agent j
    matrix = np.array([
        [0.00, 0.64],  # Samsung: depends on Sony
        [0.86, 0.00],  # Sony: depends on Samsung
    ])
    
    return InterdependenceMatrix(
        matrix=matrix,
        agent_names=["Samsung", "Sony"],
        metadata={
            "case_study": "S-LCD Joint Venture",
            "period": "2004-2011",
            "source": "TR-1 arXiv:2510.18802 Section 8"
        }
    )


def create_renault_nissan_interdependence(phase: str = "mature") -> InterdependenceMatrix:
    """
    Create the interdependence matrix for Renault-Nissan Alliance.
    
    From TR-2 Section 9, the alliance evolved through multiple phases with
    changing interdependence structures.
    
    Args:
        phase: Alliance phase - "formation", "mature", "crisis", or "strained"
    
    Returns:
        InterdependenceMatrix for the specified phase
    
    Phase Dependencies:
        Formation (1999-2002): Nissan heavily dependent (D=0.78) due to near-bankruptcy
        Mature (2002-2018): Balanced dependencies after Nissan's recovery
        Crisis (2018-2020): Ghosn arrest, power struggle
        Strained (2020+): Restructured relationship
    """
    phase_configs = {
        "formation": {
            "matrix": np.array([
                [0.00, 0.78],  # Nissan highly dependent
                [0.66, 0.00],  # Renault dependent on Asian access
            ]),
            "period": "1999-2002"
        },
        "mature": {
            "matrix": np.array([
                [0.00, 0.51],  # Nissan reduced dependency post-recovery
                [0.66, 0.00],  # Renault stable dependency
            ]),
            "period": "2002-2018"
        },
        "crisis": {
            "matrix": np.array([
                [0.00, 0.35],  # Nissan seeking independence
                [0.72, 0.00],  # Renault increased dependency
            ]),
            "period": "2018-2020"
        },
        "strained": {
            "matrix": np.array([
                [0.00, 0.28],  # Nissan further reduced
                [0.65, 0.00],  # Renault adjusted
            ]),
            "period": "2020-2025"
        }
    }
    
    if phase not in phase_configs:
        raise ValueError(f"Unknown phase: {phase}. Use: {list(phase_configs.keys())}")
    
    config = phase_configs[phase]
    
    return InterdependenceMatrix(
        matrix=config["matrix"],
        agent_names=["Nissan", "Renault"],
        metadata={
            "case_study": "Renault-Nissan Alliance",
            "phase": phase,
            "period": config["period"],
            "source": "TR-2 arXiv:2510.24909 Section 9"
        }
    )


def create_symmetric_interdependence(
    n_agents: int,
    dependency_strength: float = 0.5
) -> InterdependenceMatrix:
    """
    Create a symmetric interdependence matrix for N agents.
    
    In symmetric coopetition, all agents depend equally on each other.
    This is useful for baseline experiments and theoretical analysis.
    
    Args:
        n_agents: Number of agents
        dependency_strength: Uniform dependency coefficient D[i,j] for i≠j
    
    Returns:
        InterdependenceMatrix with uniform off-diagonal entries
    """
    matrix = np.full((n_agents, n_agents), dependency_strength)
    np.fill_diagonal(matrix, 0.0)
    
    return InterdependenceMatrix(
        matrix=matrix,
        metadata={"type": "symmetric", "strength": dependency_strength}
    )


def create_asymmetric_interdependence(
    strong_dependency: float = 0.85,
    weak_dependency: float = 0.35
) -> InterdependenceMatrix:
    """
    Create an asymmetric dyadic interdependence matrix.
    
    Models a relationship where one agent (the "weak" agent) depends heavily
    on the other (the "strong" agent), while the strong agent has moderate
    dependency on the weak. This captures hold-up situations.
    
    Args:
        strong_dependency: D[weak, strong] - how much weak depends on strong
        weak_dependency: D[strong, weak] - how much strong depends on weak
    
    Returns:
        InterdependenceMatrix with asymmetric dependencies
    """
    # Agent 0 is "strong", Agent 1 is "weak"
    matrix = np.array([
        [0.00, weak_dependency],   # Strong agent's dependencies
        [strong_dependency, 0.00], # Weak agent's dependencies
    ])
    
    return InterdependenceMatrix(
        matrix=matrix,
        agent_names=["Strong", "Weak"],
        metadata={
            "type": "asymmetric",
            "asymmetry": strong_dependency - weak_dependency
        }
    )


def create_hub_spoke_interdependence(
    n_spokes: int,
    hub_dependency: float = 0.3,
    spoke_dependency: float = 0.8
) -> InterdependenceMatrix:
    """
    Create a hub-and-spoke interdependence structure.
    
    Models a platform ecosystem where one central agent (hub) has many
    peripheral agents (spokes) dependent on it. The hub depends moderately
    on each spoke, while spokes depend heavily on the hub.
    
    Args:
        n_spokes: Number of peripheral agents
        hub_dependency: How much the hub depends on each spoke
        spoke_dependency: How much each spoke depends on the hub
    
    Returns:
        InterdependenceMatrix with hub-spoke structure
    
    Structure:
        - Agent 0 is the hub
        - Agents 1 to n_spokes are spokes
        - Spokes don't depend on each other (independence among periphery)
    """
    n_agents = 1 + n_spokes
    matrix = np.zeros((n_agents, n_agents))
    
    # Hub depends moderately on each spoke
    for j in range(1, n_agents):
        matrix[0, j] = hub_dependency
    
    # Each spoke depends heavily on hub
    for i in range(1, n_agents):
        matrix[i, 0] = spoke_dependency
    
    agent_names = ["Hub"] + [f"Spoke_{i}" for i in range(n_spokes)]
    
    return InterdependenceMatrix(
        matrix=matrix,
        agent_names=agent_names,
        metadata={
            "type": "hub_spoke",
            "n_spokes": n_spokes
        }
    )


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_power_index(D: InterdependenceMatrix) -> NDArray[np.floating]:
    """
    Compute a power index for each agent based on interdependence structure.
    
    Agents with high dependability (others depend on them) and low dependency
    (they don't depend on others) have more structural power.
    
    Power_i = (Σ_j D[j,i]) / (1 + Σ_j D[i,j])
    
    Args:
        D: Interdependence matrix
    
    Returns:
        Array of power indices, one per agent
    """
    n = D.n_agents
    power = np.zeros(n)
    
    for i in range(n):
        dependability = D.total_dependability(i)  # Others depend on me
        dependency = D.total_dependency(i)        # I depend on others
        power[i] = dependability / (1 + dependency)
    
    return power


def compute_vulnerability_index(D: InterdependenceMatrix) -> NDArray[np.floating]:
    """
    Compute vulnerability index for each agent.
    
    Agents who depend heavily on others while being less important to them
    are structurally vulnerable.
    
    Vulnerability_i = (Σ_j D[i,j]) / (1 + Σ_j D[j,i])
    
    Args:
        D: Interdependence matrix
    
    Returns:
        Array of vulnerability indices, one per agent
    """
    n = D.n_agents
    vulnerability = np.zeros(n)
    
    for i in range(n):
        dependency = D.total_dependency(i)
        dependability = D.total_dependability(i)
        vulnerability[i] = dependency / (1 + dependability)
    
    return vulnerability
