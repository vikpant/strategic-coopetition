# Coopetition-Gym Documentation

**Multi-Agent Reinforcement Learning Environments for Strategic Coopetition**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-Compatible-green.svg)](https://pettingzoo.farama.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Compatible-green.svg)](https://gymnasium.farama.org/)

---

## Compatibility and Requirements

### Framework Compatibility

| Framework | Version | Status | Notes |
|-----------|---------|--------|-------|
| **Python** | 3.9, 3.10, 3.11 | Tested | 3.9+ required |
| **Gymnasium** | 0.29+ | Compatible | Farama Foundation standard |
| **PettingZoo** | 1.24+ | Compatible | Parallel and AEC APIs |
| **NumPy** | 1.21+ | Required | Core dependency |
| **SciPy** | 1.7+ | Required | Mathematical functions |

### MARL Framework Integration

| Framework | Integration | Notes |
|-----------|-------------|-------|
| **Stable-Baselines3** | Direct | Use Gymnasium API with VecEnv |
| **RLlib** | Direct | Use PettingZoo API with MultiAgentEnv |
| **TorchRL** | Compatible | Use Gymnasium API |
| **CleanRL** | Compatible | Single-file implementations |

### Verification

```python
import coopetition_gym
import gymnasium
import pettingzoo

# Verify installation
print(f"Coopetition-Gym environments: {len(coopetition_gym.list_environments())}")
print(f"Gymnasium version: {gymnasium.__version__}")
print(f"PettingZoo version: {pettingzoo.__version__}")

# Quick environment test
env = coopetition_gym.make("TrustDilemma-v0")
obs, info = env.reset(seed=42)
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")
```

---

## Overview

**Coopetition-Gym** is a Python research library providing multi-agent reinforcement learning environments for studying *coopetitive dynamics*—scenarios where agents must simultaneously cooperate and compete. The library implements mathematical frameworks from published research:

- **TR-1**: [Computational Foundations for Strategic Coopetition: Formalizing Interdependence and Complementarity](https://arxiv.org/abs/2510.18802)
- **TR-2**: [Computational Foundations for Strategic Coopetition: Formalizing Trust and Reputation Dynamics](https://arxiv.org/abs/2510.24909)

### Key Features

- **10 Specialized Environments** spanning dyadic relationships to multi-agent ecosystems
- **Validated Case Studies** based on real business partnerships (Samsung-Sony, Renault-Nissan)
- **Trust Dynamics** with asymmetric updating and reputation hysteresis
- **Multiple APIs**: Gymnasium (single-agent), PettingZoo Parallel, and PettingZoo AEC
- **Configurable Parameters** for research flexibility

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/strategic-coopetition.git
cd strategic-coopetition/coopetition_gym

# Install in development mode
pip install -e .

# Install with all dependencies
pip install -e ".[dev,viz,rl]"
```

### Basic Usage

```python
import coopetition_gym
import numpy as np

# Create environment
env = coopetition_gym.make("TrustDilemma-v0")

# Reset and run episode
obs, info = env.reset(seed=42)
done = False

while not done:
    # Agents choose cooperation levels
    actions = np.array([50.0, 50.0])  # 50% cooperation each
    obs, rewards, terminated, truncated, info = env.step(actions)
    done = terminated or truncated

print(f"Final trust: {info['mean_trust']:.2f}")
```

### PettingZoo APIs

```python
# Parallel API (simultaneous moves)
env = coopetition_gym.make_parallel("PlatformEcosystem-v0")
observations, infos = env.reset()
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
observations, rewards, terminations, truncations, infos = env.step(actions)

# AEC API (sequential moves)
env = coopetition_gym.make_aec("TrustDilemma-v0")
env.reset()
for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    action = policy(obs) if not term else None
    env.step(action)
```

---

## Environment Categories

Coopetition-Gym provides 10 environments organized into 5 categories:

### Dyadic Environments (2-Agent)

Micro-level scenarios modeling direct partnerships between two agents.

| Environment | Description | Key Challenge |
|-------------|-------------|---------------|
| [TrustDilemma-v0](environments/trust_dilemma.md) | Continuous Prisoner's Dilemma with trust dynamics | Long-horizon impulse control |
| [PartnerHoldUp-v0](environments/partner_holdup.md) | Asymmetric power relationship | Power dynamics and exploitation |

### Ecosystem Environments (N-Agent)

Macro-level scenarios with multiple interacting agents.

| Environment | Description | Key Challenge |
|-------------|-------------|---------------|
| [PlatformEcosystem-v0](environments/platform_ecosystem.md) | Platform with N developers | Ecosystem health management |
| [DynamicPartnerSelection-v0](environments/dynamic_partner_selection.md) | Reputation-based partner matching | Social learning and signaling |

### Benchmark Environments

Research-focused environments for algorithm evaluation.

| Environment | Description | Key Challenge |
|-------------|-------------|---------------|
| [RecoveryRace-v0](environments/recovery_race.md) | Post-crisis trust recovery | Planning under trust constraints |
| [SynergySearch-v0](environments/synergy_search.md) | Hidden complementarity discovery | Exploration vs. exploitation |

### Validated Case Studies

Environments with parameters validated against real business data.

| Environment | Description | Validation |
|-------------|-------------|------------|
| [SLCD-v0](environments/slcd.md) | Samsung-Sony S-LCD Joint Venture | 58/60 accuracy |
| [RenaultNissan-v0](environments/renault_nissan.md) | Renault-Nissan Alliance phases | Multi-phase dynamics |

### Extended Environments

Advanced scenarios with additional mechanics.

| Environment | Description | Key Mechanics |
|-------------|-------------|---------------|
| [CooperativeNegotiation-v0](environments/cooperative_negotiation.md) | Multi-round negotiation | Commitment and breach penalties |
| [ReputationMarket-v0](environments/reputation_market.md) | Market with reputation tiers | Reputation as strategic asset |

---

## Core Concepts

### Coopetitive Dynamics

Coopetition occurs when entities simultaneously cooperate (to create value) and compete (to capture value). Examples include:

- **Technology Standards**: Competitors collaborate on standards while competing in products
- **Joint Ventures**: Partners invest jointly but negotiate surplus division
- **Platform Ecosystems**: Developers depend on platforms that also compete with them

### Integrated Utility (TR-1)

Agents maximize *integrated utility* that accounts for partner outcomes:

```
U_i = (e_i - a_i) + f(a_i) + α_i × Synergy + Σ(D_ij × payoff_j)
```

Where:
- `e_i - a_i`: Retained resources (endowment minus investment)
- `f(a_i)`: Value created from investment
- `α_i × Synergy`: Share of collaborative surplus
- `D_ij × payoff_j`: Value from partners' success (interdependence)

### Trust Dynamics (TR-2)

Trust evolves asymmetrically with a **negativity bias**:

```
τ(t+1) = τ(t) + λ⁺ × max(0, signal) - λ⁻ × max(0, -signal)
```

Key properties:
- **Slow to build**: `λ⁺` ≈ 0.08-0.15
- **Fast to erode**: `λ⁻` ≈ 0.25-0.45 (3:1 ratio)
- **Reputation hysteresis**: Trust ceiling `Θ = 1 - R` creates permanent limits
- **Interdependence amplification**: Higher dependency magnifies trust effects

### Value Functions

Two specifications for value creation:

**Logarithmic** (default):
```
V = θ × ln(Σ actions) × (1 + γ × complementarity)
```

**Power**:
```
V = (Σ actions)^β × (1 + γ × complementarity)
```

---

## Observation and Action Spaces

### Observation Space

All environments provide observations containing:

| Component | Shape | Description |
|-----------|-------|-------------|
| Actions | `(N,)` | All agents' cooperation levels |
| Trust Matrix | `(N, N)` | Pairwise trust levels |
| Reputation Matrix | `(N, N)` | Pairwise reputation damage |
| Interdependence | `(N, N)` | Structural dependencies |
| Step Count | `(1,)` | Normalized timestep |

### Action Space

Continuous actions representing cooperation level:

```python
Box(low=0.0, high=endowment_i, shape=(1,), dtype=float32)
```

Higher actions = more cooperation/investment.

---

## Common Parameters

### Trust Parameters

| Parameter | Symbol | Typical Range | Description |
|-----------|--------|---------------|-------------|
| Trust Building Rate | λ⁺ | 0.08 - 0.15 | Speed of trust increase |
| Trust Erosion Rate | λ⁻ | 0.25 - 0.45 | Speed of trust decrease |
| Reputation Damage | μ_R | 0.45 - 0.70 | Damage from violations |
| Reputation Decay | δ_R | 0.01 - 0.03 | Forgetting rate |
| Interdependence Amp. | ξ | 0.40 - 0.70 | Dependency amplification |
| Signal Sensitivity | κ | 1.0 - 1.5 | Action sensitivity |

### Value Function Parameters

| Parameter | Symbol | Typical Range | Description |
|-----------|--------|---------------|-------------|
| Logarithmic Scale | θ | 18 - 25 | Value magnitude |
| Complementarity | γ | 0.50 - 0.75 | Synergy from cooperation |
| Power Exponent | β | 0.70 - 0.80 | Diminishing returns |

---

## API Reference

### Factory Functions

```python
coopetition_gym.make(env_id, **kwargs)
# Returns: Gymnasium-compatible environment

coopetition_gym.make_parallel(env_id, **kwargs)
# Returns: PettingZoo ParallelEnv

coopetition_gym.make_aec(env_id, **kwargs)
# Returns: PettingZoo AECEnv

coopetition_gym.list_environments()
# Returns: List of available environment IDs
```

### Common Methods

```python
env.reset(seed=None, options=None)
# Returns: (observation, info)

env.step(action)
# Returns: (observation, reward, terminated, truncated, info)

env.render()
# Returns: Rendered output (if render_mode set)

env.close()
# Cleanup resources
```

---

## Research Applications

Coopetition-Gym supports research in:

- **Multi-Agent Reinforcement Learning**: Test MARL algorithms on strategic interaction problems
- **Game Theory**: Study equilibria in repeated games with trust dynamics
- **Mechanism Design**: Evaluate incentive structures for cooperation
- **Organizational Behavior**: Model partnership dynamics and alliance management
- **AI Safety**: Understand cooperation emergence and breakdown

---

## Citation

If you use Coopetition-Gym in your research, please cite:

```bibtex
@software{coopetition_gym,
  title = {Coopetition-Gym: Multi-Agent RL Environments for Strategic Coopetition},
  author = {Pant, Vik and Yu, Eric},
  year = {2025},
  institution = {Faculty of Information, University of Toronto},
  url = {https://github.com/your-org/strategic-coopetition}
}

@article{pant2025tr1,
  title = {Computational Foundations for Strategic Coopetition: Formalizing Interdependence and Complementarity},
  author = {Pant, Vik and Yu, Eric},
  journal = {arXiv preprint arXiv:2510.18802},
  year = {2025}
}

@article{pant2025tr2,
  title = {Computational Foundations for Strategic Coopetition: Formalizing Trust and Reputation Dynamics},
  author = {Pant, Vik and Yu, Eric},
  journal = {arXiv preprint arXiv:2510.24909},
  year = {2025}
}
```

---

## License

Coopetition-Gym is released under the [MIT License](../LICENSE).

---

## Navigation

- [Installation Guide](installation.md)
- [Tutorials](tutorials/index.md)
- [Environment Reference](environments/index.md)
- [API Documentation](api/index.md)
- [Evaluation Protocol](evaluation_protocol.md)
- [Troubleshooting](troubleshooting.md)
- [Contributing](contributing.md)
