# Tutorials

Learn to use Coopetition-Gym through hands-on examples, from basic environment interaction to advanced multi-agent training.

---

## Getting Started

Before diving into tutorials, ensure you have:

1. [Installed Coopetition-Gym](../installation.md)
2. Verified your installation works
3. Basic familiarity with Python and NumPy

---

## Tutorial Overview

### Beginner

| Tutorial | Description | Time |
|----------|-------------|------|
| [Quick Start](quickstart.md) | First steps with Coopetition-Gym | 15 min |

### Intermediate

| Tutorial | Description | Time |
|----------|-------------|------|
| Environment Deep Dive | Understanding observations, actions, and rewards | 30 min |
| Training with Stable-Baselines3 | PPO training on TrustDilemma-v0 | 45 min |
| PettingZoo APIs | Using Parallel and AEC interfaces | 30 min |

### Advanced

| Tutorial | Description | Time |
|----------|-------------|------|
| Multi-Agent Training | Independent and centralized training | 60 min |
| Custom Strategies | Implementing game-theoretic policies | 45 min |
| Experiment Design | Reproducible research workflows | 45 min |

---

## Quick Start Guide

The [Quick Start Tutorial](quickstart.md) covers:

1. **Creating environments** - Using the factory functions
2. **Basic interaction** - Reset, step, observe
3. **Understanding observations** - What agents see
4. **Understanding rewards** - How payoffs work
5. **Running episodes** - Complete interaction loops

---

## Learning Path

### For MARL Researchers

1. **Quick Start** → Basic API familiarity
2. **Environment Deep Dive** → Understanding state spaces
3. **Multi-Agent Training** → Algorithm implementation
4. **Experiment Design** → Reproducible benchmarks

### For Game Theorists

1. **Quick Start** → Basic API familiarity
2. **Custom Strategies** → Implementing equilibrium strategies
3. **Case Studies** → SLCD and Renault-Nissan analysis

### For Engineers

1. **Quick Start** → Basic API familiarity
2. **Training with Stable-Baselines3** → Practical training
3. **PettingZoo APIs** → Integration patterns

---

## Prerequisites by Tutorial

| Tutorial | Prerequisites |
|----------|---------------|
| Quick Start | Python basics, NumPy |
| Environment Deep Dive | Quick Start completed |
| Training with SB3 | Quick Start, familiarity with RL concepts |
| PettingZoo APIs | Quick Start |
| Multi-Agent Training | SB3 tutorial, MARL concepts |
| Custom Strategies | Game theory basics |
| Experiment Design | All previous tutorials |

---

## Code Repository

All tutorial code is available in the `examples/` directory:

```
examples/
├── quickstart.py           # Quick start examples
├── sb3_training.py         # Stable-Baselines3 training
├── pettingzoo_demo.py      # PettingZoo API examples
├── multi_agent/            # Multi-agent training scripts
└── strategies/             # Game-theoretic strategies
```

---

## Next Steps

Start with the [Quick Start Tutorial](quickstart.md) to begin working with Coopetition-Gym.

After completing tutorials, explore:

- [Environment Reference](../environments/index.md) - Detailed environment documentation
- [API Documentation](../api/index.md) - Complete API reference
