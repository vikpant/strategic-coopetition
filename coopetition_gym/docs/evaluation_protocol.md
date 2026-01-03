# Evaluation Protocol

Standard methodology for evaluating MARL algorithms on Coopetition-Gym environments.

---

## Overview

This document defines the **standard evaluation protocol** for reproducible benchmarking on Coopetition-Gym environments. Following this protocol ensures comparable results across studies.

---

## Episode Configuration

### Standard Episode Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Episodes per evaluation** | 100 | Sufficient for statistical significance |
| **Seeds** | 0-99 | Deterministic, reproducible evaluation |
| **Horizon** | Environment default | Typically 100-200 steps |
| **Discount factor** | γ = 1.0 | Undiscounted for full episode comparison |

### Seed Protocol

Use consecutive seeds for evaluation:

```python
import numpy as np
import coopetition_gym

def evaluate_policy(policy, env_id, n_episodes=100, seed_offset=0):
    """
    Standard evaluation function.

    Parameters
    ----------
    policy : callable
        Function mapping observation to action
    env_id : str
        Environment identifier
    n_episodes : int
        Number of evaluation episodes
    seed_offset : int
        Starting seed (for parallel evaluation)

    Returns
    -------
    dict
        Evaluation metrics
    """
    env = coopetition_gym.make(env_id)

    episode_returns = []
    episode_lengths = []
    final_trusts = []
    cooperation_rates = []

    for seed in range(seed_offset, seed_offset + n_episodes):
        obs, info = env.reset(seed=seed)
        episode_return = 0.0
        steps = 0

        terminated, truncated = False, False
        while not (terminated or truncated):
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += np.sum(reward)  # Sum over agents
            steps += 1

        episode_returns.append(episode_return)
        episode_lengths.append(steps)
        final_trusts.append(info.get('mean_trust', 0.0))
        cooperation_rates.append(info.get('cooperation_rate', 0.0))

    env.close()

    return {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'mean_final_trust': np.mean(final_trusts),
        'mean_cooperation_rate': np.mean(cooperation_rates),
        'episodes': n_episodes,
        'seeds': list(range(seed_offset, seed_offset + n_episodes)),
    }
```

---

## Primary Metrics

### Core Performance Metrics

| Metric | Description | Higher is Better |
|--------|-------------|------------------|
| **Mean Episode Return** | Sum of rewards across all agents, averaged over episodes | Yes |
| **Mean Final Trust** | Trust level at episode end | Yes |
| **Cooperation Rate** | Mean action / mean endowment | Context-dependent |
| **Episode Length** | Steps until termination/truncation | Environment-specific |

### Coopetition-Specific Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Price of Anarchy** | Welfare at NE / Welfare at Pareto Optimal | Lower is better (1.0 = efficient) |
| **Trust Stability** | Variance in trust over episode | Lower = more stable |
| **Exploitation Rate** | Instances of unilateral defection | Lower is better |
| **Recovery Events** | Trust recovery after breakdown | Higher = robust cooperation |

### Per-Agent Metrics

```python
def compute_per_agent_metrics(episode_data):
    """Compute agent-specific metrics."""
    return {
        'agent_returns': [np.mean(agent_rewards) for agent_rewards in episode_data['rewards']],
        'agent_cooperation': [np.mean(agent_actions) for agent_actions in episode_data['actions']],
        'fairness_index': compute_jain_fairness(episode_data['rewards']),
        'exploitation_asymmetry': compute_exploitation_asymmetry(episode_data),
    }

def compute_jain_fairness(rewards):
    """Jain's Fairness Index: 1.0 = perfectly fair."""
    n = len(rewards)
    sum_r = sum(rewards)
    sum_r2 = sum(r**2 for r in rewards)
    return (sum_r ** 2) / (n * sum_r2) if sum_r2 > 0 else 1.0
```

---

## Baseline Algorithms

### Required Baselines

Every evaluation should compare against these baselines:

#### 1. Random Policy

```python
def random_policy(obs, env):
    """Uniformly random actions."""
    return env.action_space.sample()
```

Expected behavior: Provides lower bound; trust should decay due to inconsistent behavior.

#### 2. Constant Cooperation

```python
def constant_cooperation_policy(obs, env, level=0.5):
    """Fixed cooperation level (proportion of endowment)."""
    return level * env.action_space.high
```

Variants:
- `level=0.35`: Baseline threshold (expected minimum)
- `level=0.50`: Moderate cooperation
- `level=0.75`: High cooperation

#### 3. Tit-for-Tat (TFT)

```python
class TitForTatPolicy:
    """Classic TFT adapted for continuous actions."""

    def __init__(self, initial_action=0.5):
        self.last_partner_action = initial_action

    def __call__(self, obs, env, agent_idx=0):
        # Extract partner's last action from observation
        partner_idx = 1 - agent_idx
        partner_action = obs[partner_idx]  # Depends on obs structure

        # Match partner's cooperation level
        action = partner_action
        self.last_partner_action = partner_action

        return np.clip(action, env.action_space.low, env.action_space.high)
```

#### 4. Independent PPO (IPPO)

Standard PPO applied independently to each agent:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def train_ippo(env_id, total_timesteps=100_000):
    """Train independent PPO agents."""
    env = DummyVecEnv([lambda: coopetition_gym.make(env_id)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)
    return model
```

### Optional Baselines

| Algorithm | Use Case | Reference |
|-----------|----------|-----------|
| **MAPPO** | Centralized training | Yu et al., 2022 |
| **QMIX** | Value decomposition | Rashid et al., 2018 |
| **MADDPG** | Continuous multi-agent | Lowe et al., 2017 |
| **SAC** | Maximum entropy | Haarnoja et al., 2018 |

---

## Training Protocol

### Hyperparameter Reporting

Report all hyperparameters used:

```yaml
# hyperparameters.yaml
algorithm: PPO
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5
network:
  hidden_layers: [128, 128]
  activation: ReLU
training:
  total_timesteps: 1_000_000
  eval_freq: 10_000
  n_eval_episodes: 10
```

### Training Seeds

Use separate seeds for training and evaluation:

| Seed Range | Purpose |
|------------|---------|
| 0-99 | Evaluation (fixed) |
| 100-104 | Training runs (5 seeds for variance) |

```python
training_seeds = [100, 101, 102, 103, 104]

for seed in training_seeds:
    model = train_algorithm(env_id, seed=seed)
    results = evaluate_policy(model.predict, env_id, n_episodes=100)
    save_results(results, seed)
```

### Learning Curves

Record metrics during training:

```python
from stable_baselines3.common.callbacks import EvalCallback

eval_env = coopetition_gym.make(env_id)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/",
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
)

model.learn(total_timesteps=1_000_000, callback=eval_callback)
```

---

## Environment-Specific Protocols

### Dyadic Environments (2-Agent)

**TrustDilemma-v0, PartnerHoldUp-v0**

| Parameter | Value |
|-----------|-------|
| Horizon | 100 steps |
| Evaluation episodes | 100 |
| Training timesteps | 500K - 1M |

Key metrics:
- Mean episode return (both agents combined)
- Final trust level
- Cooperation rate
- Trust trajectory (record trust at each step)

### Ecosystem Environments (N-Agent)

**PlatformEcosystem-v0, DynamicPartnerSelection-v0**

| Parameter | Value |
|-----------|-------|
| Default N | 4 developers (5 total agents) |
| Horizon | 100 steps |
| Evaluation episodes | 50 (higher variance) |
| Training timesteps | 1M - 2M |

Key metrics:
- Mean ecosystem welfare
- Platform-developer trust
- Developer retention rate
- Network effects captured

### Benchmark Environments

**RecoveryRace-v0, SynergySearch-v0**

| Parameter | Value |
|-----------|-------|
| Horizon | 100 steps |
| Evaluation episodes | 100 |
| SynergySearch gamma | Record inferred vs. true |

SynergySearch-specific metrics:
- Inference accuracy (correctly identified high/low synergy)
- Exploration steps (steps before convergence)
- Regret (optimal - achieved return)

### Validated Environments

**SLCD-v0, RenaultNissan-v0**

| Parameter | Value |
|-----------|-------|
| Horizon | 40 (SLCD), 60 (RenaultNissan) |
| Evaluation episodes | 100 |
| Compare against | Historical data points |

Key metrics:
- Trajectory matching (compared to real data)
- Phase transition accuracy
- Parameter sensitivity

---

## Results Reporting

### Standard Results Table

```markdown
| Algorithm | Mean Return | Std | Final Trust | Coop Rate | Training Steps |
|-----------|-------------|-----|-------------|-----------|----------------|
| Random | 85.2 | 12.3 | 0.32 | 0.48 | - |
| Constant(0.5) | 112.4 | 8.7 | 0.58 | 0.50 | - |
| TFT | 124.6 | 10.2 | 0.62 | 0.54 | - |
| IPPO | 142.3 | 9.8 | 0.71 | 0.58 | 500K |
| Your Method | **156.8** | 7.2 | **0.78** | 0.62 | 500K |
```

### Statistical Significance

Report confidence intervals and significance tests:

```python
from scipy import stats

def compute_ci(data, confidence=0.95):
    """Compute confidence interval."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h

def compare_methods(results_a, results_b):
    """Welch's t-test for method comparison."""
    t_stat, p_value = stats.ttest_ind(results_a, results_b, equal_var=False)
    return t_stat, p_value
```

Report:
- Mean ± 95% CI
- p-values for comparisons (α = 0.05)
- Effect sizes (Cohen's d)

### Reproducibility Checklist

Include in publications:

- [ ] Environment version (coopetition_gym.__version__)
- [ ] Random seeds for training and evaluation
- [ ] Complete hyperparameters
- [ ] Hardware specifications (GPU, memory)
- [ ] Training time
- [ ] Code availability (link to repository)

---

## Computational Requirements

### Estimated Training Times

| Environment | 100K Steps | 500K Steps | 1M Steps |
|-------------|------------|------------|----------|
| TrustDilemma-v0 | ~5 min | ~25 min | ~50 min |
| PartnerHoldUp-v0 | ~5 min | ~25 min | ~50 min |
| PlatformEcosystem-v0 (N=4) | ~10 min | ~50 min | ~100 min |
| SLCD-v0 | ~3 min | ~15 min | ~30 min |

*Estimates based on NVIDIA GTX 1650 with PPO, single environment.*

### Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | None (CPU ok) | NVIDIA GTX 1650+ |
| VRAM | - | 4 GB+ |
| RAM | 8 GB | 16 GB |
| Storage | 1 GB | 10 GB (for logs) |

---

## Example: Complete Evaluation Script

```python
#!/usr/bin/env python
"""
Standard evaluation script for Coopetition-Gym.

Usage:
    python evaluate.py --env TrustDilemma-v0 --algorithm ppo --seed 100
"""

import argparse
import json
import numpy as np
import coopetition_gym
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="TrustDilemma-v0")
    parser.add_argument("--algorithm", default="random")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    # Create environment
    env = coopetition_gym.make(args.env)

    # Select policy
    if args.algorithm == "random":
        policy = lambda obs: env.action_space.sample()
    elif args.algorithm == "constant":
        policy = lambda obs: 0.5 * env.action_space.high
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    # Evaluate
    results = evaluate_policy(
        policy,
        args.env,
        n_episodes=args.n_episodes,
        seed_offset=0
    )

    # Add metadata
    results['environment'] = args.env
    results['algorithm'] = args.algorithm
    results['timestamp'] = datetime.now().isoformat()
    results['coopetition_gym_version'] = coopetition_gym.__version__

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")
    print(f"Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"Mean Final Trust: {results['mean_final_trust']:.3f}")

if __name__ == "__main__":
    main()
```

---

## See Also

- [Environment Reference](environments/index.md) - Detailed environment documentation
- [API Documentation](api/index.md) - API reference
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [Contributing](contributing.md) - How to contribute baseline results

---

## References

1. Yu, C., Velu, A., Vinitsky, E., et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games. NeurIPS.
2. Rashid, T., Samvelyan, M., de Witt, C.S., et al. (2018). QMIX: Monotonic Value Function Factorisation. ICML.
3. Lowe, R., Wu, Y., Tamar, A., et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments. NeurIPS.
4. Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL. ICML.
