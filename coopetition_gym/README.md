# 🎮 Coopetition-Gym

**Multi-Agent Reinforcement Learning for Strategic Coopetition**

**Unreleased source candidate: 1.0.8.** The optional SLCD extension candidate is 0.1.1. These changes do not publish a distribution or create a release tag. The [installation guide](docs/installation.md) describes installing the candidate from this checkout.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-compatible-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Coopetition-Gym provides Gymnasium-compatible environments for studying **coopetitive dynamics** in multi-agent systems. Coopetition refers to the simultaneous presence of cooperation and competition between agents, a phenomenon ubiquitous in business alliances, platform ecosystems, and strategic partnerships.

## 🔬 Research Foundation

This library implements computational frameworks described in the following research technical reports:

| Paper | Topic | Key Contribution |
|-------|-------|------------------|
| **TR-1** ([arXiv:2510.18802](https://arxiv.org/abs/2510.18802)) | Interdependence & Complementarity | Value functions, synergy, coopetitive equilibrium |
| **TR-2** ([arXiv:2510.24909](https://arxiv.org/abs/2510.24909)) | Trust Dynamics | Asymmetric updating, negativity bias, hysteresis |
| **TR-3** ([arXiv:2601.16237](https://arxiv.org/abs/2601.16237)) | Collective Action & Loyalty | Team production, loyalty mechanisms, coalition dynamics |
| **TR-4** ([arXiv:2604.01240](https://arxiv.org/abs/2604.01240)) | Sequential Interaction & Reciprocity | Memory-bounded reciprocity, graduated sanctions, platform dynamics |

**Validated Case Studies:**
- S-LCD (Samsung-Sony): **58/60** validation score, 96.7% (TR-1 §8)
- Renault-Nissan Alliance: **49/60** validation score, 81.7% (TR-2 §9)
- Apache HTTP Server: [recorded score discrepancies and provenance](docs/benchmarks/score_provenance.md) (TR-3).
- Apple iOS App Store: [recorded score discrepancies and provenance](docs/benchmarks/score_provenance.md) (TR-4).

**Research artifacts:** The repository includes training and behavioral-audit datasets. See [REPRODUCE.md](../REPRODUCE.md) for provenance and the distinction between records and experimental cells. Candidate runtime checks do not revalidate historical benchmark claims.

## 🚀 Quick Start

### Installation

```bash
# From the repository root after cloning:
git clone https://github.com/vikpant/strategic-coopetition.git
cd strategic-coopetition
python -m pip install -e './coopetition_gym[dev,experiments]'
cd coopetition_gym
```

Use a suitable Python environment. PyTorch and Stable-Baselines3 are core dependencies; the `experiments` extra adds analysis packages. Published PyPI releases are separate from this unreleased source candidate.

### Basic Usage

```python
import coopetition_gym

# Create an environment
env = coopetition_gym.make("TrustDilemma-v0")

# Standard Gymnasium interface
obs, info = env.reset(seed=42)

# Agents choose cooperation levels [0, 100]
actions = [60.0, 55.0]  # Agent 0 cooperates at 60%, Agent 1 at 55%

obs, rewards, terminated, truncated, info = env.step(actions)

print(f"Rewards: {rewards}")
print(f"Current Trust: {info['mean_trust']:.3f}")
```

The joint-action API returns a NumPy reward vector, one entry per agent. Explicit Gymnasium registration is also available as `gymnasium.make("coopetition_gym:TrustDilemma-v0")`; it preserves that vector. Ordinary scalar-reward learners require an explicit adapter, described below.

### List Available Environments

```python
import coopetition_gym

print(coopetition_gym.list_environments())
# ['TrustDilemma-v0', 'PartnerHoldUp-v0', 'PlatformEcosystem-v0', ...]
```

## 🎯 Environments

Coopetition-Gym includes **20 environments** across seven categories. See [default agent counts, horizons and observation shapes](docs/api/quick_reference.md#available-environments). Apache defaults to 40 agents in its maturity phase, while Apple defaults to 3 agents and 66 steps.

### Category 1: Dyadic (Micro)
Fundamental 2-agent mechanics for understanding core dynamics.

| Environment | Description | Challenge |
|-------------|-------------|-----------|
| `TrustDilemma-v0` | Continuous iterated Prisoner's Dilemma with trust dynamics | Learn long-horizon impulse control |
| `PartnerHoldUp-v0` | Asymmetric vertical relationship | Defensive strategies vs. exploitation |

### Category 2: Ecosystem (Macro)
N-agent systems testing emergent behavior.

| Environment | Description | Challenge |
|-------------|-------------|-----------|
| `PlatformEcosystem-v0` | Platform + N developers | Mechanism design, ecosystem health |
| `DynamicPartnerSelection-v0` | Reputation-based partner matching | Social learning, reputation maintenance |

### Category 3: Research Benchmarks
Diagnostic environments isolating specific dynamics.

| Environment | Description | Challenge |
|-------------|-------------|-----------|
| `RecoveryRace-v0` | Post-crisis trust recovery | Optimal recovery sequences under ceiling constraints |
| `SynergySearch-v0` | Hidden complementarity parameter | Exploration vs. exploitation |

### Category 4: Validated Case Studies
Real-world calibrated benchmarks.

| Environment | Description | Validation |
|-------------|-------------|------------|
| `SLCD-v0` | Samsung-Sony LCD JV (2004-2011) | 58/60 score (TR-1 §8) |
| `RenaultNissan-v0` | Renault-Nissan Alliance (multi-phase) | TR-2 §9 validated |

### Category 5: Extended Environments
Advanced mechanics for specialized research.

| Environment | Description | Challenge |
|-------------|-------------|-----------|
| `CooperativeNegotiation-v0` | Multi-round negotiation with commitment | Agreement formation, breach consequences |
| `ReputationMarket-v0` | Market with public reputation scores | Reputation as strategic asset |

### Category 6: Collective Action (TR-3)
Team production with loyalty dynamics and coalition formation.

| Environment | Description | Challenge |
|-------------|-------------|-----------|
| `TeamProduction-v0` | N-agent team production game | Free-rider dynamics at Nash equilibrium |
| `LoyaltyTeam-v0` | Team production with loyalty mechanisms | Sustaining above-Nash cooperation |
| `CoalitionFormation-v0` | Dynamic coalition with entry/exit | Coalition stability under exclusion threat |
| `ApacheProject-v0` | Apache HTTP Server ([score provenance](docs/benchmarks/score_provenance.md)) | Phase-dependent contributor dynamics |
| `PublicGoods-v0` | Classic public goods game | Contribution with optional punishment |

### Category 7: Reciprocity (TR-4)
Sequential interaction with memory-bounded reciprocity dynamics.

| Environment | Description | Challenge |
|-------------|-------------|-----------|
| `ReciprocalDilemma-v0` | Continuous PD with direct reciprocity | Conditional cooperation via bounded memory |
| `GiftExchange-v0` | Asymmetric employer-worker exchange | Asymmetric reciprocity sensitivity |
| `IndirectReciprocity-v0` | 4-agent reputation-mediated cooperation | Indirect reciprocity via image scoring |
| `GraduatedSanction-v0` | 6-agent commons with graduated sanctions | Proportional punishment and escalation |
| `AppleAppStore-v0` | Apple iOS App Store ([score provenance](docs/benchmarks/score_provenance.md)) | Platform power asymmetry and reciprocity |

## 📐 Mathematical Framework

### Value Creation (TR-1)

**Individual Value Function** (Equation 6):
```
f_i(a_i) = θ · ln(1 + a_i)     where θ = 20.0
```

**Synergy Function** (Equation 7):
```
g(a) = (∏ a_i)^(1/N)          Geometric mean
```

**Total Value** (Equation 8):
```
V(a|γ) = Σ f_i(a_i) + γ · g(a)   where γ = 0.65 for S-LCD
```

### Trust Dynamics (TR-2)

**Cooperation Signal** (Equation 6):
```
s_ij = tanh(κ · (a_j - baseline))   where κ = 1.0
```

**Trust Building** (Equation 7):
```
ΔT = λ⁺ · s · (1-T) · ceiling      where λ⁺ = 0.10
```

**Trust Erosion** (Equation 8):
```
ΔT = λ⁻ · s · T · (1 + ξ·D_ij)     where λ⁻ = 0.30, ξ = 0.50
```

**Key Property: Negativity Bias**
```
λ⁻/λ⁺ = 3×    Trust erodes 3× faster than it builds
```

### Coopetitive Equilibrium (TR-1)

**Integrated Utility** (Equation 13):
```
U_i(a) = π_i(a) + Σ_{j≠i} D_ij · π_j(a)
```

Agents maximize integrated utility, which includes weighted concern for partners' payoffs based on interdependence coefficients D_ij.

### Reciprocity Dynamics (TR-4)

**Cooperation Signal** (Equation 19):
```
s_ij = a_j - ā_j    deviation from memory average
```

**Bounded Response** (Equation 21):
```
φ(x) = tanh(κ · x)    where κ = 0.8-1.0
```

**Reciprocity Sensitivity** (Equation 23):
```
ρ_ij = ρ_0 · D_ij^η    dependency scales reciprocity
```

**Reciprocity Modifier** (Equation 44):
```
U_recip = λ_R Σ T_ij · (1 + ω·D_ij) · ρ_ij · φ(s_ij)
```

Agents condition behavior on observed partner actions over a bounded memory window (k = 3-10 steps). Higher dependency creates stronger reciprocal responses.

## 🎯 Game-Theoretic Oracles

Seven oracle policies provide non-learning reference points for algorithmic comparison. Each oracle applies to a specific mechanism class.

| Oracle | Role | Covers |
|---|---|---|
| `Oracle_Equilibrium` | TR-1 interdependence equilibrium (Nash reference) | DynamicPartnerSelection, PartnerHoldUp, PlatformEcosystem, SynergySearch, RenaultNissan |
| `Oracle_TrustAware` | TR-2 trust-aware equilibrium | CooperativeNegotiation, RecoveryRace, ReputationMarket, SLCD, TrustDilemma |
| `Oracle_Nash` | TR-3 Nash equilibrium reference | ApacheProject, CoalitionFormation, LoyaltyTeam, PublicGoods, TeamProduction |
| `Oracle_Loyalty` | TR-3 cooperative reference | All 5 TR-3 environments |
| `Oracle_SocialOptimum` | TR-3 social optimum (equivalent to Oracle_Loyalty) | All 5 TR-3 environments |
| `Oracle_ReciprocityEquilibrium` | TR-4 Nash-style equilibrium reference | ReciprocalDilemma, GiftExchange, IndirectReciprocity, GraduatedSanction, AppleAppStore |
| `Oracle_BoundedReciprocity` | TR-4 cooperative reference | All 5 TR-4 environments |

These policies provide mechanism-specific reference points. Their names do not establish universal lower or upper bounds on learned-policy returns; comparisons must use matched objectives and evaluation settings.

## 🧪 Training with RL Algorithms

### Explicit scalar aggregation for Stable-Baselines3

A joint-controller example requires a scalar-reward adapter. This adapter sums agent rewards; report that objective when interpreting results. Construction and prediction below do not train the model.

```python
import coopetition_gym as cg
from experiments.algorithms import MultiAgentToSingleAgentWrapper
from stable_baselines3 import PPO

base_env = cg.make("TrustDilemma-v0", reward_type="private", max_steps=20)
env = MultiAgentToSingleAgentWrapper(base_env)
model = PPO("MlpPolicy", env, device="cpu", n_steps=64, batch_size=32)
# Training is a separate, intentional action: model.learn(total_timesteps=...)
obs, info = env.reset(seed=42)
for _ in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

The historical `IndependentPPO`, `IndependentSAC` and `IndependentA2C` implementations also use joint controllers with summed rewards. Their identifiers alone do not establish decentralized independent learning. See the [wrapper reference](docs/api/wrappers.md#scalar-reward-adapter).

### Custom Policy Example

```python
import numpy as np
import coopetition_gym

def cooperative_policy(obs, trust_threshold=0.5):
    """Policy that adjusts cooperation based on observed trust."""
    # Extract trust from observation (simplified)
    n_agents = 2
    trust_start = n_agents
    trust_matrix = obs[trust_start:trust_start + 4].reshape(2, 2)
    mean_trust = (trust_matrix[0, 1] + trust_matrix[1, 0]) / 2
    
    # Higher trust -> higher cooperation
    base = 40.0
    sensitivity = 40.0
    cooperation = base + sensitivity * mean_trust
    
    return np.array([cooperation, cooperation])

# Run episode
env = coopetition_gym.make("TrustDilemma-v0")
obs, _ = env.reset(seed=42)

for step in range(100):
    action = cooperative_policy(obs)
    obs, rewards, done, truncated, info = env.step(action)
    if done or truncated: break

print(f"Episode ended at step {step+1}")
print(f"Final trust: {info['mean_trust']:.3f}")
```

## 📊 Analysis and Evaluation

```python
import coopetition_gym
from coopetition_gym.utils import run_episode, aggregate_results, make_constant_policy

# Define policies to compare
policies = {
    "cooperative": make_constant_policy(70.0),
    "moderate": make_constant_policy(50.0),
    "defecting": make_constant_policy(25.0),
}

# Run experiments
env = coopetition_gym.make("TrustDilemma-v0", max_steps=100)

for name, policy in policies.items():
    results = [run_episode(env, policy, seed=i) for i in range(10)]
    stats = aggregate_results(results)
    
    print(f"\n{name.upper()} Policy:")
    print(f"  Mean Total Reward: {stats['mean_total_reward']}")
    print(f"  Mean Final Trust: {stats['mean_final_trust']:.3f}")
    print(f"  Mean Cooperation Rate: {stats['mean_cooperation_rate']:.1%}")
```

## 🔧 Customization

### Custom Environment Configuration

```python
import numpy as np
from coopetition_gym import EnvironmentConfig, CoopetitionEnv
from coopetition_gym.core import (
    ValueFunctionParameters, 
    TrustParameters,
    create_symmetric_interdependence
)

# Custom parameters
config = EnvironmentConfig(
    n_agents=3,
    max_steps=200,
    endowments=np.array([100.0, 80.0, 120.0]),
    alpha=np.array([0.4, 0.3, 0.3]),
    interdependence_matrix=create_symmetric_interdependence(3, 0.45).matrix,
    value_params=ValueFunctionParameters(gamma=0.70),
    trust_params=TrustParameters(lambda_plus=0.12, lambda_minus=0.36),
    reward_type="integrated",
)

env = CoopetitionEnv(config=config)
```

### Accessing Core Components

```python
from coopetition_gym.core import (
    # Value functions
    individual_value,
    synergy_function,
    total_value,
    
    # Interdependence
    create_slcd_interdependence,
    create_renault_nissan_interdependence,
    
    # Trust dynamics
    TrustDynamicsModel,
    TrustParameters,
    
    # Equilibrium
    solve_equilibrium,
    compute_rewards,
)

# Compute equilibrium for S-LCD case
from coopetition_gym.core import create_slcd_payoff_params

params = create_slcd_payoff_params()
equilibrium = solve_equilibrium(params, equilibrium_type="coopetitive")

print(f"Equilibrium actions: {equilibrium.actions}")
print(f"Total welfare: {equilibrium.total_welfare:.2f}")
```

## 📁 Project Structure

```
coopetition_gym/
├── __init__.py               # Package entry point
├── core/                     # Mathematical foundations
│   ├── value_functions.py        # TR-1 value calculations
│   ├── interdependence.py        # Dependency matrices
│   ├── trust_dynamics.py         # TR-2 trust evolution
│   ├── collective_action.py      # TR-3 loyalty mechanisms
│   └── equilibrium.py            # Payoffs and equilibrium
├── envs/                     # Gymnasium environments
│   ├── base.py                   # CoopetitionEnv base class
│   ├── dyadic_envs.py            # TrustDilemma, PartnerHoldUp
│   ├── ecosystem_envs.py         # Platform, PartnerSelection
│   ├── benchmark_envs.py         # RecoveryRace, SynergySearch
│   ├── case_study_envs.py        # SLCD, RenaultNissan
│   ├── extended_envs.py          # Negotiation, ReputationMarket
│   ├── collective_action_envs.py # TR-3: TeamProduction, Loyalty, Coalition, Apache
│   └── reciprocity_envs.py      # TR-4: ReciprocalDilemma, GiftExchange, IndirectReciprocity, GraduatedSanction, AppleAppStore
├── utils/                    # Utilities and helpers
└── tests/                    # Test suite
```

## 🧪 Running Tests

From the `coopetition_gym/` package directory:

```bash
# Core, experiment, and extension checks without the IPPO training test
python -B -m pytest -p no:cacheprovider -k 'not test_ippo_trains_on_2d'

# Core coverage
python -B -m pytest -p no:cacheprovider coopetition_gym/tests --cov=coopetition_gym
```

Omitting the exclusion from the full suite includes a short extension training test. The optional [SLCD extension](extensions/slcd_2d/README.md) has seven learner adapters plus an oracle, with its own installed `slcd_2d` namespace.

## 📚 Citation

If you use this library in your research, please cite:

```bibtex
@software{coopetition_gym,
  author = {Pant, Vik and Yu, Eric},
  title = {Coopetition-Gym: Multi-Agent RL for Strategic Coopetition},
  year = {2025},
  url = {https://github.com/vikpant/strategic-coopetition}
}

@article{pant2025interdependence,
  title = {Interdependence and Complementarity in Coopetitive Relationships},
  author = {Pant, Vik and Yu, Eric},
  journal = {arXiv preprint arXiv:2510.18802},
  year = {2025}
}

@article{pant2025trust,
  title = {Trust Dynamics in Coopetitive Relationships},
  author = {Pant, Vik and Yu, Eric},
  journal = {arXiv preprint arXiv:2510.24909},
  year = {2025}
}

@article{pant2026collective,
  title = {Collective Action and Loyalty in Coopetitive Relationships},
  author = {Pant, Vik and Yu, Eric},
  journal = {arXiv preprint arXiv:2601.16237},
  year = {2026}
}

@article{pant2026reciprocity,
  title = {Computational Foundations for Strategic Coopetition: Formalizing Sequential Interaction and Reciprocity},
  author = {Pant, Vik and Yu, Eric},
  journal = {arXiv preprint arXiv:2604.01240},
  year = {2026}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👥 Authors

- **Vik Pant, PhD** - Faculty of Information, University of Toronto
- **Eric Yu, PhD** - Faculty of Information and Department of Computer Science, University of Toronto

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](docs/contributing.md) for details.

---

*Coopetition-Gym: Where cooperation meets competition, and game theory meets reinforcement learning.* 🎮
