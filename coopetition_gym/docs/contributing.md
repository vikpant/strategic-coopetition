# Contributing to Coopetition-Gym

Thank you for your interest in contributing to Coopetition-Gym! This guide explains how to contribute code, documentation, and research.

---

## Ways to Contribute

### Code Contributions

- **Bug fixes**: Fix issues in existing environments
- **New environments**: Implement additional coopetition scenarios
- **Performance improvements**: Optimize existing code
- **API enhancements**: Improve interfaces and usability

### Documentation Contributions

- **Tutorials**: Write guides for specific use cases
- **Examples**: Add code examples and notebooks
- **Corrections**: Fix errors or unclear explanations
- **Translations**: Help translate documentation

### Research Contributions

- **Benchmarks**: Publish MARL algorithm results
- **Case studies**: Validate environments against real data
- **Extensions**: Propose new theoretical frameworks

---

## Development Setup

### Prerequisites

- Python 3.9+
- Git

### Local Installation

```bash
# Clone the repository
git clone https://github.com/your-org/strategic-coopetition.git
cd strategic-coopetition/coopetition_gym

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=coopetition_gym

# Run specific test file
pytest tests/test_dyadic_envs.py
```

---

## Code Style

### Python Style

We follow PEP 8 with some modifications:

- Line length: 100 characters
- Use type hints for function signatures
- Use docstrings for all public functions

```python
def compute_trust_update(
    current_trust: float,
    signal: float,
    lambda_plus: float = 0.10,
    lambda_minus: float = 0.30,
) -> float:
    """
    Compute trust update from cooperation signal.

    Parameters
    ----------
    current_trust : float
        Current trust level in [0, 1].
    signal : float
        Cooperation signal (positive = cooperative).
    lambda_plus : float, optional
        Trust building rate. Default: 0.10.
    lambda_minus : float, optional
        Trust erosion rate. Default: 0.30.

    Returns
    -------
    float
        Updated trust level, clipped to [0, 1].
    """
    if signal > 0:
        delta = lambda_plus * signal * (1 - current_trust)
    else:
        delta = lambda_minus * signal * current_trust

    return np.clip(current_trust + delta, 0, 1)
```

### Formatting Tools

```bash
# Format code
black coopetition_gym/

# Sort imports
isort coopetition_gym/

# Type checking
mypy coopetition_gym/
```

---

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation only
- `refactor/` - Code refactoring

### 2. Make Changes

- Write clear, commented code
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run tests
pytest

# Check formatting
black --check coopetition_gym/
isort --check coopetition_gym/
```

### 4. Commit

```bash
git add .
git commit -m "Brief description of changes"
```

Commit message format:
- Start with verb (Add, Fix, Update, Remove)
- Keep first line under 50 characters
- Add details in body if needed

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

---

## Adding New Environments

### Environment Structure

New environments should follow this structure:

```python
from coopetition_gym.envs.base import AbstractCoopetitionEnv

class NewEnvironment(AbstractCoopetitionEnv):
    """
    NewEnvironment-v0: Brief description.

    This environment models [scenario description].

    Parameters
    ----------
    max_steps : int
        Maximum timesteps per episode.
    custom_param : float
        Description of custom parameter.

    Attributes
    ----------
    n_agents : int
        Number of agents (2 for dyadic).
    endowments : np.ndarray
        Agent endowments.

    References
    ----------
    [1] Author (Year). Title. Journal.
    """

    def __init__(
        self,
        max_steps: int = 100,
        custom_param: float = 0.5,
        render_mode: Optional[str] = None,
    ):
        # Initialize base class
        super().__init__(
            n_agents=2,
            endowments=np.array([100.0, 100.0]),
            baselines=np.array([35.0, 35.0]),
            alphas=np.array([0.50, 0.50]),
            max_steps=max_steps,
            render_mode=render_mode,
        )

        # Custom initialization
        self.custom_param = custom_param

    def _compute_rewards(self, actions: np.ndarray) -> np.ndarray:
        """Compute rewards for given actions."""
        # Implement reward logic
        pass

    def _compute_termination(self) -> bool:
        """Check for early termination."""
        # Implement termination logic
        return False
```

### Registration

Register the environment in `coopetition_gym/__init__.py`:

```python
from gymnasium.envs.registration import register

register(
    id="NewEnvironment-v0",
    entry_point="coopetition_gym.envs.your_module:NewEnvironment",
)
```

### Documentation

Create documentation in `docs/environments/new_environment.md` following the existing template:

1. MARL Classification table
2. Formal Specification
3. Game-Theoretic Background
4. Environment Specification
5. Example code
6. References

### Tests

Add tests in `tests/test_new_environment.py`:

```python
import pytest
import numpy as np
import coopetition_gym

class TestNewEnvironment:
    def test_creation(self):
        env = coopetition_gym.make("NewEnvironment-v0")
        assert env.n_agents == 2

    def test_reset(self):
        env = coopetition_gym.make("NewEnvironment-v0")
        obs, info = env.reset(seed=42)
        assert obs.shape == (expected_dim,)

    def test_step(self):
        env = coopetition_gym.make("NewEnvironment-v0")
        obs, info = env.reset(seed=42)
        actions = np.array([50.0, 50.0])
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert rewards.shape == (2,)
```

---

## Documentation Style

### Markdown Guidelines

- Use ATX-style headers (`#`, `##`, `###`)
- Use fenced code blocks with language specifiers
- Include table of contents for long documents
- Add cross-references to related sections

### Code Examples

- All examples should be runnable
- Use `seed=42` for reproducibility
- Include expected output where helpful

```python
import coopetition_gym
import numpy as np

env = coopetition_gym.make("TrustDilemma-v0")
obs, info = env.reset(seed=42)

# Expected: obs.shape == (17,)
print(f"Observation shape: {obs.shape}")
```

### Mathematical Notation

- Use LaTeX-style notation in markdown: `$\tau_{ij}$`
- Define all symbols on first use
- Keep notation consistent with TR-1 and TR-2 papers

---

## Reporting Issues

### Bug Reports

Include:
- Python version
- Package versions (`pip list`)
- Minimal code to reproduce
- Full error traceback
- Expected vs. actual behavior

### Feature Requests

Include:
- Use case description
- Proposed API or interface
- Relevant literature references
- Willingness to implement

---

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism
- Focus on collaboration
- Prioritize research integrity

### Unacceptable Behavior

- Harassment or discrimination
- Publishing others' work without credit
- Fabricating research results

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Questions?

- Open a [GitHub Issue](https://github.com/your-org/strategic-coopetition/issues)
- Contact the maintainers

---

## Acknowledgments

We thank all contributors who help improve Coopetition-Gym!
