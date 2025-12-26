# 🎮 Coopetition-Gym

**Multi-Agent Reinforcement Learning for Strategic Coopetition**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-compatible-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Coopetition-Gym provides Gymnasium-compatible environments for studying **coopetitive dynamics** in multi-agent systems. Coopetition refers to the simultaneous presence of cooperation and competition between agents—a phenomenon ubiquitous in business alliances, platform ecosystems, and strategic partnerships.

## 🔬 Research Foundation

This library implements computational frameworks from peer-reviewed game-theoretic research:

| Paper | Topic | Key Contribution |
|-------|-------|------------------|
| **TR-1** ([arXiv:2510.18802](https://arxiv.org/abs/2510.18802)) | Interdependence & Complementarity | Value functions, synergy, coopetitive equilibrium |
| **TR-2** ([arXiv:2510.24909](https://arxiv.org/abs/2510.24909)) | Trust Dynamics | Asymmetric updating, negativity bias, hysteresis |

The S-LCD environment achieves a **58/60 validation score** against historical data from the Samsung-Sony LCD joint venture.

## 🚀 Quick Start

### Installation

```bash
pip install coopetition-gym

# Or from source
git clone https://github.com/vikpant/coopetition-gym.git
cd coopetition-gym
pip install -e .
```

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

### List Available Environments

```python
import coopetition_gym

print(coopetition_gym.list_environments())
# ['TrustDilemma-v0', 'PartnerHoldUp-v0', 'PlatformEcosystem-v0', ...]
```

## 🎯 Environments

Coopetition-Gym includes **10 environments** across five categories:

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

## 🧪 Training with RL Algorithms

### With Stable-Baselines3

```python
import coopetition_gym
from stable_baselines3 import PPO

# Create environment
env = coopetition_gym.make("TrustDilemma-v0")

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Evaluate
obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
print(f"Final trust: {info['mean_trust']:.3f}")
```

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
    if done or truncated:
        break

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
├── __init__.py           # Package entry point
├── core/                 # Mathematical foundations
│   ├── value_functions.py    # TR-1 value calculations
│   ├── interdependence.py    # Dependency matrices
│   ├── trust_dynamics.py     # TR-2 trust evolution
│   └── equilibrium.py        # Payoffs and equilibrium
├── envs/                 # Gymnasium environments
│   ├── base.py               # CoopetitionEnv base class
│   ├── dyadic_envs.py        # TrustDilemma, PartnerHoldUp
│   ├── ecosystem_envs.py     # Platform, PartnerSelection
│   ├── benchmark_envs.py     # RecoveryRace, SynergySearch
│   ├── case_study_envs.py    # SLCD, RenaultNissan
│   └── extended_envs.py      # Negotiation, ReputationMarket
├── utils/                # Utilities and helpers
└── tests/                # Test suite
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=coopetition_gym --cov-report=html
```

## 📚 Citation

If you use this library in your research, please cite:

```bibtex
@software{coopetition_gym,
  author = {Pant, Vik and Yu, Eric},
  title = {Coopetition-Gym: Multi-Agent RL for Strategic Coopetition},
  year = {2025},
  url = {https://github.com/vikpant/coopetition-gym}
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
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👥 Authors

- **Vik Pant, PhD** - Faculty of Information, University of Toronto
- **Eric Yu, PhD** - Faculty of Information, University of Toronto

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

---

*Coopetition-Gym: Where cooperation meets competition, and game theory meets reinforcement learning.* 🎮
