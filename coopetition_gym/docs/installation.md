# Installation Guide

This guide covers installing Coopetition-Gym for research and development use.

---

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.9 | 3.10 or 3.11 |
| **RAM** | 4 GB | 8 GB+ |
| **Disk Space** | 500 MB | 2 GB (with RL frameworks) |

### Required Dependencies

Coopetition-Gym requires:

| Package | Version | Purpose |
|---------|---------|---------|
| **NumPy** | 1.21+ | Array operations |
| **SciPy** | 1.7+ | Mathematical functions |
| **Gymnasium** | 0.29+ | Single-agent API |
| **PettingZoo** | 1.24+ | Multi-agent APIs |

---

## Quick Install

### From PyPI (Recommended)

```bash
pip install coopetition-gym
```

### From Source

```bash
# Clone the repository
git clone https://github.com/your-org/strategic-coopetition.git
cd strategic-coopetition/coopetition_gym

# Install in development mode
pip install -e .
```

---

## Development Install

For contributors and researchers who need the full development environment:

```bash
# Clone the repository
git clone https://github.com/your-org/strategic-coopetition.git
cd strategic-coopetition/coopetition_gym

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install with all development dependencies
pip install -e ".[dev,viz,rl]"
```

### Dependency Groups

| Group | Contents | Use Case |
|-------|----------|----------|
| `dev` | pytest, black, mypy, sphinx | Development and testing |
| `viz` | matplotlib, seaborn, plotly | Visualization and analysis |
| `rl` | stable-baselines3, torch | RL training |

Install specific groups:

```bash
# Just development tools
pip install -e ".[dev]"

# Development and visualization
pip install -e ".[dev,viz]"

# Everything
pip install -e ".[dev,viz,rl]"
```

---

## Optional Dependencies

### MARL Framework Integration

Coopetition-Gym integrates with major MARL frameworks. Install them separately:

#### Stable-Baselines3 (PyTorch)

```bash
pip install stable-baselines3[extra]
```

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import coopetition_gym

env = DummyVecEnv([lambda: coopetition_gym.make("TrustDilemma-v0")])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

#### RLlib (Ray)

```bash
pip install "ray[rllib]"
```

```python
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
import coopetition_gym

# Register environment with RLlib
from ray.tune.registry import register_env
register_env("trust_dilemma", lambda config: coopetition_gym.make_parallel("TrustDilemma-v0"))
```

#### TorchRL

```bash
pip install torchrl
```

#### CleanRL

```bash
pip install cleanrl
```

### Visualization

```bash
# Matplotlib (basic plotting)
pip install matplotlib

# Seaborn (statistical visualization)
pip install seaborn

# Plotly (interactive plots)
pip install plotly
```

---

## GPU Setup

GPU acceleration is optional but recommended for training RL agents.

### PyTorch with CUDA

```bash
# Check your CUDA version
nvidia-smi

# Install PyTorch with appropriate CUDA version
# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### TensorFlow with GPU

```bash
pip install tensorflow[and-cuda]
```

### Verifying GPU Access

```python
# PyTorch
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# TensorFlow
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")
```

### Memory Considerations

For GPUs with limited VRAM (4-8 GB):

```python
# PyTorch: Use smaller batch sizes
model = PPO("MlpPolicy", env, batch_size=64, policy_kwargs=dict(net_arch=[128, 128]))

# TensorFlow: Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

---

## Verification

After installation, verify everything works:

```python
import coopetition_gym
import gymnasium
import pettingzoo

# Check versions
print(f"Coopetition-Gym environments: {len(coopetition_gym.list_environments())}")
print(f"Gymnasium version: {gymnasium.__version__}")
print(f"PettingZoo version: {pettingzoo.__version__}")

# Test Gymnasium API
env = coopetition_gym.make("TrustDilemma-v0")
obs, info = env.reset(seed=42)
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")

# Test PettingZoo Parallel API
env_parallel = coopetition_gym.make_parallel("TrustDilemma-v0")
observations, infos = env_parallel.reset(seed=42)
print(f"Agents: {env_parallel.agents}")

# Test PettingZoo AEC API
env_aec = coopetition_gym.make_aec("TrustDilemma-v0")
env_aec.reset(seed=42)
print(f"AEC agents: {env_aec.agents}")

print("\nAll tests passed!")
```

### Expected Output

```
Coopetition-Gym environments: 10
Gymnasium version: 0.29.x
PettingZoo version: 1.24.x
Observation shape: (17,)
Action space: Box(0.0, 100.0, (2,), float32)
Agents: ['agent_0', 'agent_1']
AEC agents: ['agent_0', 'agent_1']

All tests passed!
```

---

## Troubleshooting

### Common Issues

#### ImportError: No module named 'coopetition_gym'

**Cause**: Package not installed or wrong Python environment.

**Solution**:
```bash
# Check which Python is active
which python

# Ensure you're in the correct virtual environment
source venv/bin/activate

# Reinstall
pip install -e .
```

#### Version Conflicts with Gymnasium/PettingZoo

**Cause**: Incompatible versions of Gymnasium or PettingZoo.

**Solution**:
```bash
# Install compatible versions
pip install "gymnasium>=0.29,<1.0"
pip install "pettingzoo>=1.24,<2.0"
```

#### CUDA Out of Memory

**Cause**: GPU memory exhausted during training.

**Solution**:
- Reduce batch size
- Use smaller networks
- Enable gradient checkpointing

```python
# Smaller batch size
model = PPO("MlpPolicy", env, batch_size=32)

# Smaller network
model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[64, 64]))
```

#### Slow Performance Without GPU

**Cause**: RL training running on CPU.

**Solution**:
- Verify GPU installation (see GPU Setup above)
- For CPU-only machines, use vectorized environments:

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

# Parallel CPU environments
env = SubprocVecEnv([lambda: coopetition_gym.make("TrustDilemma-v0") for _ in range(4)])
```

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/your-org/strategic-coopetition/issues)
2. Search existing issues for similar problems
3. Open a new issue with:
   - Python version (`python --version`)
   - Package versions (`pip list | grep -E "coopetition|gymnasium|pettingzoo"`)
   - Full error traceback
   - Minimal code to reproduce

---

## Next Steps

- [Quick Start Tutorial](tutorials/quickstart.md) - Get started with your first environment
- [Environment Reference](environments/index.md) - Explore all 10 environments
- [API Documentation](api/index.md) - Complete API reference
