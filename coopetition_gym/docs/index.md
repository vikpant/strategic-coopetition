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

![Coopetition Dynamics Animation](assets/images/manim/coopetition_dynamics.gif)

*Two-agent coopetition: agents balance cooperation (blue) and competition over time, with trust (purple) evolving based on their strategic choices. The animation shows action trajectories and real-time trust evolution.*

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

> **For Researchers**: Full mathematical derivations, proofs, and validation methodology are available in the [Theoretical Foundations](theory/index.md) documentation and the published technical reports.
>
> **For Practitioners**: The summaries below provide the essential intuition needed to use the environments effectively.

### Coopetitive Dynamics

Coopetition occurs when entities simultaneously cooperate (to create value) and compete (to capture value). As Brandenburger and Nalebuff articulated: actors *"cooperate to grow the pie and compete to split it up."*

**Real-World Examples**:
- **Technology Standards**: Competitors collaborate on standards while competing in products (e.g., Bluetooth SIG members)
- **Joint Ventures**: Partners invest jointly but negotiate surplus division (e.g., Samsung-Sony S-LCD)
- **Platform Ecosystems**: Developers depend on platforms that also compete with them (e.g., iOS App Store)
- **Supply Chains**: Suppliers share information for efficiency while competing for contracts

**The Coopetition Paradox**: The same relationship exhibits both cooperative and competitive dynamics simultaneously—not sequentially or in separate domains. This creates strategic tension that standard game theory struggles to capture.

### Interdependence & Structural Coupling (TR-1)

Interdependence captures why actors must consider partner outcomes even while competing. When Actor A depends on Actor B for critical resources, A's success structurally requires B's success—creating *instrumental* concern for B's welfare distinct from altruism.

**The Interdependence Matrix** quantifies structural dependencies:

$$\Large D_{ij} = \frac{\sum_{d \in \mathcal{D}_i} w_d \cdot \text{Dep}(i,j,d) \cdot \text{crit}(i,j,d)}{\sum_{d \in \mathcal{D}_i} w_d}$$

| Component | Meaning | Example |
|-----------|---------|---------|
| $w_d$ | Importance weight of goal d | Revenue goal: 0.8, Brand goal: 0.2 |
| $\text{Dep}(i,j,d)$ | Does i depend on j for d? | Developer depends on platform for distribution |
| $\text{crit}(i,j,d)$ | Criticality (1 = sole provider) | API provider with no alternatives: 1.0 |

**Key Insight**: $D_{ij} \neq D_{ji}$ in general. Asymmetric dependencies create power imbalances—a startup may critically depend on a platform ($D_{\text{startup,platform}} \approx 0.8$) while the platform barely notices any single startup ($D_{\text{platform,startup}} \approx 0.01$).

### Integrated Utility Function (TR-1)

Agents maximize *integrated utility* that accounts for partner outcomes through structural coupling:

$$\Large U_i(\mathbf{a}) = \pi_i(\mathbf{a}) + \sum_{j \neq i} D_{ij} \cdot \pi_j(\mathbf{a})$$

**Components Explained**:

| Term | Formula | Intuition |
|------|---------|-----------|
| Private Payoff | $\pi_i = e_i - a_i + f(a_i) + \alpha_i \cdot \text{Synergy}$ | What I keep + what I create + my share of joint value |
| Interdependence Term | $\sum_{j} D_{ij} \cdot \pi_j$ | Partner success weighted by my dependency on them |

**Why This Matters**: Classical Nash Equilibrium assumes purely self-interested payoffs. The *Coopetitive Equilibrium* extends Nash by incorporating dependency-weighted concern for partner outcomes—capturing why dependent actors rationally care about partner success.

### Value Creation & Complementarity (TR-1)

Complementarity creates the cooperative incentive: joint action produces superadditive value exceeding independent contributions.

$$\Large V(\mathbf{a} \mid \gamma) = \sum_{i=1}^{N} f_i(a_i) + \gamma \cdot g(a_1, \ldots, a_N)$$

**Two Validated Specifications**:

| Specification | Individual Value $f(a)$ | Synergy $g(a)$ | Best For |
|---------------|----------------------|--------------|----------|
| **Logarithmic** (default) | $\theta \cdot \ln(1 + a_i)$, $\theta=20$ | Geometric mean | Manufacturing JVs (58/60 validation) |
| **Power** | $a_i^{\beta}$, $\beta=0.75$ | Geometric mean | General scenarios (46/60 validation) |

**Key Parameters** (validated across 22,000+ trials):
- **θ = 20.0**: Logarithmic scale producing realistic cooperation magnitudes
- **β = 0.75**: Diminishing returns reflecting investment economics
- **γ = 0.65**: Complementarity strength balancing individual and joint value

### Trust Dynamics (TR-2)

![Cooperation-Trust Phase Space](assets/images/manim/phase_space.gif)

*Phase space visualization: The green region represents sustainable partnerships with high cooperation and trust. The red region shows defection spirals. Trajectories demonstrate how initial conditions determine the equilibrium reached—cooperation begets cooperation, while early defection leads to relationship collapse.*

Trust evolves through a **two-layer architecture** capturing both immediate behavioral responses and long-term memory:

| Layer | Symbol | Updates | Captures |
|-------|--------|---------|----------|
| Immediate Trust | $T_{ij} \in [0,1]$ | Every interaction | Current confidence in partner |
| Reputation Damage | $R_{ij} \in [0,1]$ | On violations | Historical memory of betrayals |

**Asymmetric Evolution with Negativity Bias**:

$$
\Delta T =
\begin{cases}
\lambda^+ \cdot s \cdot (\Theta - T) & \text{if } s > 0 \; [\lambda^+ = 0.10] \\
-\lambda^- \cdot |s| \cdot T \cdot (1 + \xi D) & \text{if } s \leq 0 \; [\lambda^- = 0.30]
\end{cases}
$$

**The 3:1 Ratio**: Trust erodes approximately 3× faster than it builds ($\lambda^-/\lambda^+ \approx 3.0$). This negativity bias, validated against behavioral economics research, explains why:
- A single major violation can destroy months of trust-building
- Consistent cooperation is essential for sustainable partnerships
- Recovery from betrayal requires sustained effort over extended periods

**Trust Ceiling Mechanism**:

$$\Large \Theta = 1 - R \quad \text{(reputation damage limits maximum achievable trust)}$$

Even with perfect cooperation, damaged reputation prevents trust from fully recovering—creating permanent relationship constraints (hysteresis).

**Interdependence Amplification**: High-dependency relationships experience 27% faster trust erosion for equivalent violations:

$$\Large \text{Erosion factor} = (1 + \xi \cdot D_{ij}) \quad \text{where } \xi = 0.50$$

When you depend heavily on a partner, their betrayal hurts more.

### Empirical Validation

The mathematical framework has been validated against real business partnerships:

| Case Study | Validation Score | Key Dynamics Captured |
|------------|------------------|----------------------|
| **Samsung-Sony S-LCD** (2004-2011) | 58/60 (96.7%) | Interdependence, complementarity, cooperation levels |
| **Renault-Nissan Alliance** (1999-2025) | 49/60 (81.7%) | Trust evolution, crisis, recovery across 5 phases |

These validations ensure the environments produce realistic coopetitive dynamics rather than artificial constructs.

> **Learn More**: See [Theoretical Foundations](theory/index.md) for complete mathematical derivations, [Parameter Reference](theory/parameters.md) for validated values, and [Benchmark Results](benchmarks/index.md) for algorithm performance analysis.

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
| Trust Building Rate | $\lambda^+$ | 0.08 - 0.15 | Speed of trust increase |
| Trust Erosion Rate | $\lambda^-$ | 0.25 - 0.45 | Speed of trust decrease |
| Reputation Damage | $\mu_R$ | 0.45 - 0.70 | Damage from violations |
| Reputation Decay | $\delta_R$ | 0.01 - 0.03 | Forgetting rate |
| Interdependence Amp. | $\xi$ | 0.40 - 0.70 | Dependency amplification |
| Signal Sensitivity | $\kappa$ | 1.0 - 1.5 | Action sensitivity |

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

### Getting Started
- [Installation Guide](installation.md)
- [Tutorials](tutorials/index.md)
- [Quick Start](tutorials/quickstart.md)

### Reference
- [Environment Reference](environments/index.md)
- [API Documentation](api/index.md)
- [Parameter Reference](theory/parameters.md)

### Theory & Research
- [Theoretical Foundations](theory/index.md) **NEW**
  - [Interdependence Framework](theory/interdependence.md)
  - [Value Creation & Complementarity](theory/value_creation.md)
  - [Trust Dynamics](theory/trust_dynamics.md)
- [Benchmark Results](benchmarks/index.md)
- [Implementation Roadmap](roadmap.md) **NEW**

### Development
- [Evaluation Protocol](evaluation_protocol.md)
- [Contributing](contributing.md)
- [Troubleshooting](troubleshooting.md)

---

## Benchmark Highlights

![Algorithm Performance Comparison](assets/images/manim/algorithm_comparison.gif)

*Benchmark results from 760 experiments: Simple heuristic strategies (Constant_050, Constant_075) surprisingly outperform all 20 learning algorithms. The key finding: predictable cooperation builds trust more effectively than optimized but erratic learned policies.*

We have evaluated **20 MARL algorithms** across all 10 environments with **760 experiments** totaling **76,000 evaluation episodes**. Key findings:

| Finding | Implication |
|---------|-------------|
| Simple heuristics (Constant_050) outperform all learning algorithms | Predictable cooperation builds trust |
| Trust-Return correlation: r = 0.552 | Trust causally drives performance |
| Population methods (Self-Play, FCP) fail catastrophically | Nash equilibria are Pareto-suboptimal |
| CTDE methods cluster together | Centralized critic dominates actor architecture |

See [Benchmark Results](benchmarks/index.md) for comprehensive analysis.
