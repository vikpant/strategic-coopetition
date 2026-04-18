### Vik Pant, PhD
**Computational Game Theory | Multi-Agent Systems | Strategic Coopetition | Reinforcement Learning**

Faculty of Information, University of Toronto

#### Research Program
I develop computational techniques for modeling strategic coopetition
by formalizing how actors cooperate and compete simultaneously in mixed-motive
multi-agent environments.
My research bridges conceptual modeling with computational game theory
and reinforcement learning.

---

## 📦 Strategic Coopetition

This repository is the home of the **Strategic Coopetition** research program. It contains two publicly released artifacts and the supporting validation suite.

| Folder | Contents |
|---|---|
| [`coopetition_gym/`](coopetition_gym/) | The Coopetition-Gym Python package — 20 multi-agent reinforcement learning environments for studying coopetitive dynamics. See [coopetition_gym/README.md](coopetition_gym/README.md). |
| [`TR_validation/`](TR_validation/) | Validation suites that reproduce the empirical results reported in the four technical reports. |

### Installation

```bash
git clone https://github.com/vikpant/strategic-coopetition.git
cd strategic-coopetition/coopetition_gym
pip install -e .
```

### Quick start

```python
import coopetition_gym

env = coopetition_gym.make("TrustDilemma-v0")
obs, info = env.reset(seed=42)

for _ in range(100):
    obs, reward, terminated, truncated, info = env.step([60.0, 55.0])
    if terminated or truncated:
        break
```

### Reproducing paper results

See [REPRODUCE.md](REPRODUCE.md) for step-by-step instructions to reproduce the tables, figures, and datasets in the accompanying NeurIPS 2026 submission.

---

#### Technical Reports
- [arXiv:2510.18802](https://arxiv.org/abs/2510.18802) — Formalizing Interdependence and Complementarity (TR-1)
- [arXiv:2510.24909](https://arxiv.org/abs/2510.24909) — Formalizing Trust and Reputation Dynamics (TR-2)
- [arXiv:2601.16237](https://arxiv.org/abs/2601.16237) — Formalizing Collective Action and Loyalty (TR-3)
- [arXiv:2604.01240](https://arxiv.org/abs/2604.01240) — Formalizing Sequential Interaction and Reciprocity (TR-4)

#### Conference paper (under review)
- Pant, V. and Yu, E. (2026). *Reward-Type Ablation Reveals Mechanism-Dependent Algorithm Rankings in Mixed-Motive Multi-Agent Evaluation.* Submitted to NeurIPS 2026 Evaluations and Datasets Track.

#### Validated Case Studies
| Case study | Validation score | Technical report |
|---|---|---|
| Samsung–Sony S-LCD Joint Venture (2004–2011) | 58/60 (96.7%) | TR-1 §8 |
| Renault–Nissan Alliance (multi-phase) | 49/60 (81.7%) | TR-2 §9 |
| Apache HTTP Server community evolution | 52/60 (86.7%) | TR-3 §7 |
| Apple iOS App Store platform dynamics | 48/55 (87.3%) | TR-4 §8 |

#### Citation

If you use this work in your research, please cite the accompanying technical reports and the conference paper (BibTeX entries in [coopetition_gym/README.md](coopetition_gym/README.md#-citation)).

#### License

- **Code** (`coopetition_gym/`): MIT License — see [coopetition_gym/LICENSE](coopetition_gym/LICENSE).
- **Validation suite** (`TR_validation/`): MIT License.

#### Connect
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vikpant)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-4285F4?style=flat&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?hl=en&user=eoKMjOMAAAAJ)
