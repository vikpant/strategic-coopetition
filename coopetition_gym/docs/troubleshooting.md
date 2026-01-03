# Troubleshooting

Common issues and solutions when working with Coopetition-Gym.

---

## Installation Issues

### ModuleNotFoundError: No module named 'coopetition_gym'

**Cause**: Package not installed or installed in different environment.

**Solution**:
```bash
# Verify installation
pip show coopetition_gym

# If not found, install
cd strategic-coopetition/coopetition_gym
pip install -e .

# Verify
python -c "import coopetition_gym; print(coopetition_gym.list_environments())"
```

### Version Conflicts with Gymnasium/PettingZoo

**Cause**: Incompatible package versions.

**Solution**:
```bash
# Check versions
pip show gymnasium pettingzoo

# Upgrade to compatible versions
pip install "gymnasium>=0.29" "pettingzoo>=1.24"

# Reinstall coopetition_gym
pip install -e .
```

**Required versions**:
- Python: 3.9+
- Gymnasium: 0.29+
- PettingZoo: 1.24+
- NumPy: 1.21+

### Import Errors with NumPy

**Error**: `AttributeError: module 'numpy' has no attribute 'bool'`

**Cause**: NumPy 2.0 removed deprecated aliases.

**Solution**:
```bash
# Downgrade NumPy (if needed)
pip install "numpy<2.0"

# Or upgrade coopetition_gym to latest version
pip install -e . --upgrade
```

---

## Environment Creation Issues

### ValueError: Unknown environment ID

**Error**: `ValueError: Environment 'TrustDilema-v0' not found`

**Cause**: Typo in environment name or environment not registered.

**Solution**:
```python
import coopetition_gym

# List all available environments
print(coopetition_gym.list_environments())
# ['TrustDilemma-v0', 'PartnerHoldUp-v0', ...]

# Correct spelling
env = coopetition_gym.make("TrustDilemma-v0")  # Note: 'Dilemma' not 'Dilema'
```

### TypeError: __init__() got an unexpected keyword argument

**Cause**: Using invalid parameter for environment.

**Solution**:
```python
# Check valid parameters in documentation
# Example: TrustDilemma-v0 parameters
env = coopetition_gym.make(
    "TrustDilemma-v0",
    max_steps=100,           # Valid
    lambda_plus=0.10,        # Valid
    lambda_minus=0.30,       # Valid
    # invalid_param=1.0,     # Would cause error
)
```

---

## Training Issues

### NaN in Observations or Rewards

**Cause**: Numerical instability from extreme actions or parameter values.

**Solution**:
```python
import numpy as np

# Clip actions to valid range
action = np.clip(raw_action, env.action_space.low, env.action_space.high)

# Check for NaN before stepping
if np.isnan(action).any():
    action = env.action_space.sample()  # Fallback to random

obs, reward, terminated, truncated, info = env.step(action)

# Verify output
if np.isnan(obs).any() or np.isnan(reward).any():
    print("Warning: NaN detected in environment output")
    obs, info = env.reset()  # Reset if corrupted
```

### Trust Collapses Too Quickly

**Cause**: High erosion rate or inconsistent actions.

**Symptoms**:
- Trust drops to 0 within first 10-20 steps
- Episodes terminate early due to trust collapse

**Solution**:
```python
# Use more conservative trust parameters
env = coopetition_gym.make(
    "TrustDilemma-v0",
    lambda_plus=0.12,   # Faster building
    lambda_minus=0.25,  # Slower erosion (still 2:1 ratio)
    initial_trust=0.60, # Start higher
)

# Ensure consistent cooperation during training
# Avoid large swings in action values
```

### Agent Learns to Always Defect

**Cause**: Short-term reward dominates; agent doesn't discover cooperation benefits.

**Symptoms**:
- Actions converge to minimum (0 or near-0)
- Episode returns plateau at low value
- Trust always near 0

**Solution**:
```python
# 1. Increase exploration
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    env,
    ent_coef=0.05,  # Higher entropy for more exploration
)

# 2. Use longer horizon
env = coopetition_gym.make("TrustDilemma-v0", max_steps=200)

# 3. Reward shaping (add cooperation bonus during training)
# Note: Only for training, remove for evaluation
class CooperationBonus(gym.Wrapper):
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        coop_bonus = 0.1 * np.mean(action)  # Small bonus for cooperating
        return obs, reward + coop_bonus, term, trunc, info
```

### Agents Don't Coordinate

**Cause**: Independent learning without communication mechanism.

**Symptoms**:
- Agents oscillate between cooperation and defection
- Trust never stabilizes
- Returns have high variance

**Solution**:
```python
# 1. Try parameter sharing
# Use same policy network for both agents

# 2. Use centralized training (MAPPO, QMIX)
# Agents share information during training

# 3. Increase training time
# Coordination often emerges later in training
model.learn(total_timesteps=2_000_000)  # More than default
```

---

## Memory and Performance Issues

### CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Cause**: Batch size or network too large for GPU.

**Solution**:
```python
# Reduce batch size
model = PPO("MlpPolicy", env, batch_size=32)  # Down from 64

# Smaller network
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[64, 64])  # Down from [128, 128]
)

# Force CPU training
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
```

### Training Too Slow

**Cause**: Inefficient environment stepping or excessive logging.

**Solution**:
```python
# 1. Use vectorized environments
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(seed):
    def _init():
        env = coopetition_gym.make("TrustDilemma-v0")
        env.reset(seed=seed)
        return env
    return _init

env = SubprocVecEnv([make_env(i) for i in range(4)])  # 4 parallel envs

# 2. Reduce logging frequency
model.learn(total_timesteps=1_000_000, log_interval=100)  # Less frequent

# 3. Disable rendering during training
env = coopetition_gym.make("TrustDilemma-v0", render_mode=None)
```

### High Memory Usage

**Cause**: Large replay buffer or storing unnecessary data.

**Solution**:
```python
# Limit replay buffer (for off-policy algorithms)
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy",
    env,
    buffer_size=50_000,  # Down from default 1M
)

# Clear memory periodically
import gc
gc.collect()

# Monitor memory
import psutil
print(f"Memory: {psutil.Process().memory_info().rss / 1e9:.2f} GB")
```

---

## API Issues

### PettingZoo API Mismatch

**Error**: `AttributeError: 'ParallelEnv' object has no attribute 'observation_spaces'`

**Cause**: Using old PettingZoo API (pre-1.24).

**Solution**:
```python
# Upgrade PettingZoo
pip install "pettingzoo>=1.24"

# New API uses methods, not properties
env = coopetition_gym.make_parallel("TrustDilemma-v0")

# Correct usage
obs_space = env.observation_space("agent_0")  # Method call
act_space = env.action_space("agent_0")       # Method call

# Not: env.observation_spaces["agent_0"]  # Old API
```

### Action Space Mismatch

**Error**: `AssertionError: Action outside of bounds`

**Cause**: Action doesn't match expected shape or range.

**Solution**:
```python
env = coopetition_gym.make("TrustDilemma-v0")

# Check action space
print(f"Shape: {env.action_space.shape}")    # (2,) for dyadic
print(f"Low: {env.action_space.low}")        # [0., 0.]
print(f"High: {env.action_space.high}")      # [100., 100.]

# Ensure correct action format
action = np.array([50.0, 50.0], dtype=np.float32)

# Clip to valid range
action = np.clip(action, env.action_space.low, env.action_space.high)
```

### AEC vs Parallel API Confusion

**Error**: Actions not accepted or wrong observation returned.

**Cause**: Mixing up AEC (sequential) and Parallel (simultaneous) APIs.

**Solution**:
```python
# Parallel API: All agents act simultaneously
env = coopetition_gym.make_parallel("TrustDilemma-v0")
observations, infos = env.reset()
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
observations, rewards, terminations, truncations, infos = env.step(actions)

# AEC API: Agents act sequentially
env = coopetition_gym.make_aec("TrustDilemma-v0")
env.reset()
for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    action = None if term or trunc else env.action_space(agent).sample()
    env.step(action)  # Single action, not dict
```

---

## Reproducibility Issues

### Different Results with Same Seed

**Cause**: Environment or algorithm has additional randomness sources.

**Solution**:
```python
import numpy as np
import random
import torch

def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set seeds before creating environment
set_all_seeds(42)
env = coopetition_gym.make("TrustDilemma-v0")
obs, info = env.reset(seed=42)

# For Stable-Baselines3
from stable_baselines3.common.utils import set_random_seed
set_random_seed(42)
```

### Results Vary Across Runs

**Cause**: GPU non-determinism or multi-threading.

**Solution**:
```python
# Force deterministic operations (slower)
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Or use CPU only for exact reproducibility
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

---

## Debugging Tips

### Visualize Trust Evolution

```python
import matplotlib.pyplot as plt

env = coopetition_gym.make("TrustDilemma-v0")
obs, info = env.reset(seed=42)

trust_history = [info['mean_trust']]
action_history = []

for _ in range(100):
    action = np.array([50.0, 50.0])
    obs, reward, term, trunc, info = env.step(action)
    trust_history.append(info['mean_trust'])
    action_history.append(action.mean())

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(trust_history)
plt.xlabel('Step')
plt.ylabel('Mean Trust')
plt.title('Trust Evolution')

plt.subplot(1, 2, 2)
plt.plot(action_history)
plt.xlabel('Step')
plt.ylabel('Mean Action')
plt.title('Cooperation Level')

plt.tight_layout()
plt.savefig('debug_trust.png')
```

### Inspect Observation Structure

```python
env = coopetition_gym.make("TrustDilemma-v0")
obs, info = env.reset(seed=42)

print(f"Observation shape: {obs.shape}")
print(f"Observation: {obs}")
print(f"Info keys: {info.keys()}")

# Decode observation components
print(f"Actions (0-1): {obs[0:2]}")
print(f"Trust matrix (2-5): {obs[2:6].reshape(2,2)}")
print(f"Reputation matrix (6-9): {obs[6:10].reshape(2,2)}")
```

### Check Reward Scaling

```python
# Run episode and check reward distribution
env = coopetition_gym.make("TrustDilemma-v0")
obs, info = env.reset(seed=42)

rewards = []
for _ in range(100):
    action = np.array([50.0, 50.0])
    obs, reward, term, trunc, info = env.step(action)
    rewards.append(reward)

rewards = np.array(rewards)
print(f"Reward mean: {rewards.mean():.2f}")
print(f"Reward std: {rewards.std():.2f}")
print(f"Reward min: {rewards.min():.2f}")
print(f"Reward max: {rewards.max():.2f}")
```

---

## Getting Help

If issues persist:

1. **Check documentation**: [Environment Reference](environments/index.md), [API Documentation](api/index.md)
2. **Search issues**: [GitHub Issues](https://github.com/your-org/strategic-coopetition/issues)
3. **Open new issue**: Include:
   - Python version
   - Package versions (`pip freeze`)
   - Minimal reproducing code
   - Full error traceback
   - Expected vs. actual behavior

---

## See Also

- [Installation Guide](installation.md) - Setup instructions
- [Evaluation Protocol](evaluation_protocol.md) - Standard benchmarking methodology
- [Contributing](contributing.md) - How to report bugs
