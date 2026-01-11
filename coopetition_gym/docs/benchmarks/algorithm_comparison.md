# Algorithm Comparison

**Deep Analysis of 20 MARL Algorithms on Coopetitive Dynamics**

This page provides detailed analysis of each algorithm category's performance, failure modes, and suitability for coopetitive environments.

---

## Algorithm Taxonomy

### Learning Paradigms

```
MARL Algorithms
├── Independent Learning
│   ├── IPPO (Independent PPO)
│   ├── ISAC (Independent SAC)
│   └── IA2C (Independent A2C)
├── Centralized Training, Decentralized Execution (CTDE)
│   ├── Continuous Action
│   │   ├── MAPPO (Multi-Agent PPO)
│   │   ├── MADDPG (Multi-Agent DDPG)
│   │   ├── MATD3 (Multi-Agent TD3)
│   │   └── MASAC (Multi-Agent SAC)
│   └── Discrete Action (Value Decomposition)
│       ├── QMIX
│       ├── VDN
│       └── COMA
├── Opponent Modeling
│   ├── LOLA (Learning with Opponent-Learning Awareness)
│   └── M3DDPG (Minimax MADDPG)
├── Population-Based
│   ├── SelfPlay_PPO
│   └── FCP (Fictitious Co-Play)
└── Mean-Field
    └── MeanFieldAC
```

---

## Category Analysis

### 1. Heuristic Baselines

**Algorithms:** Random, Constant_035, Constant_050, Constant_075, TitForTat

#### Performance Summary

| Algorithm | Mean Return | Cooperation | Trust | Training Time |
|-----------|-------------|-------------|-------|---------------|
| Constant_050 | 72,494 | 55.0% | 98.7% | 0 |
| Constant_075 | 69,594 | 78.7% | 98.7% | 0 |
| Constant_035 | 53,390 | 38.5% | 55.5% | 0 |
| TitForTat | 38,556 | 35.2% | 48.9% | 0 |
| Random | 24,939 | 47.7% | 7.1% | 0 |

#### Key Insights

**Why Constant_050 Wins:**
1. **Predictability builds trust**: Partners can anticipate behavior, enabling coordination
2. **Moderate cooperation avoids exploitation**: Neither too generous (exploitable) nor too stingy (triggering defection)
3. **Trust compounds**: High sustained trust (98.7%) unlocks cooperative surplus over time

**Why Random Fails:**
- Despite 47.7% average cooperation (similar to Constant_050), unpredictability destroys trust
- Trust erodes to 7.1% because partners cannot distinguish cooperation from noise
- Validates TR-2's prediction that consistent signaling is essential

**TitForTat Underperformance:**
- Classic game theory optimal strategy underperforms in continuous action spaces
- Reactive strategies create oscillations rather than stable cooperation
- Initial defection (common in learning opponents) triggers defection spirals

#### Recommendation
Use Constant_050 as the primary baseline. Any learning algorithm should beat this to be considered successful.

---

### 2. Independent Learning

**Algorithms:** IPPO, ISAC, IA2C

#### Performance Summary

| Algorithm | Mean Return | Cooperation | Trust | Training Time |
|-----------|-------------|-------------|-------|---------------|
| ISAC | 65,563 | 80.3% | 98.6% | 3.3 hrs |
| IA2C | 28,089 | 19.6% | 15.3% | 1.1 hrs |
| IPPO | 20,981 | 11.6% | 6.5% | 0.9 hrs |

#### Why ISAC Succeeds

ISAC dramatically outperforms other independent learners. Analysis reveals:

1. **Entropy regularization**: SAC's maximum entropy objective encourages exploration, helping discover cooperative equilibria
2. **Off-policy learning**: Replay buffer enables learning from past cooperative experiences
3. **Continuous action space fit**: SAC handles continuous cooperation levels naturally

#### Why IPPO Fails

IPPO achieves only 11.6% cooperation—worse than Random's 47.7%:

1. **On-policy limitation**: PPO discards experiences, losing memory of successful cooperation
2. **Clipped objective**: Conservative updates prevent rapid adaptation to partner behavior
3. **Convergence to defection**: Without coordination signals, agents converge to safe (but suboptimal) defection

#### IA2C Analysis

IA2C falls between ISAC and IPPO:
- Higher cooperation (19.6%) than IPPO but still trust-eroding
- Faster training but worse asymptotic performance
- Variance in gradient estimates leads to unstable cooperation

#### Recommendation
If using independent learning, use ISAC exclusively. IPPO and IA2C are unsuitable for coopetitive environments.

---

### 3. CTDE-Continuous

**Algorithms:** MAPPO, MADDPG, MATD3, MASAC

#### Performance Summary

| Algorithm | Mean Return | Cooperation | Trust | Training Time |
|-----------|-------------|-------------|-------|---------------|
| M3DDPG | 50,916 | 75.8% | 73.5% | 8.1 hrs |
| MATD3 | 50,913 | 75.8% | 73.5% | 8.0 hrs |
| MASAC | 50,908 | 75.8% | 73.5% | 7.8 hrs |
| MADDPG | 50,903 | 75.8% | 73.5% | 8.1 hrs |
| MAPPO | 35,423 | 36.0% | 27.1% | 20.8 hrs |

#### The CTDE Clustering Phenomenon

MADDPG, MATD3, MASAC, and M3DDPG achieve nearly identical results (within 0.03% of each other). This remarkable convergence suggests:

1. **Centralized critic dominance**: Access to joint observations provides identical coordination signals
2. **Policy gradient equivalence**: In cooperative settings, actor variants converge to similar policies
3. **Replay buffer alignment**: All use similar off-policy sampling strategies

#### MAPPO Anomaly

MAPPO underperforms despite being the most popular MARL algorithm:

| Metric | MAPPO | MADDPG | Difference |
|--------|-------|--------|------------|
| Return | 35,423 | 50,903 | -30.4% |
| Cooperation | 36.0% | 75.8% | -52.5% |
| Trust | 27.1% | 73.5% | -63.1% |
| Training Time | 20.8 hrs | 8.1 hrs | +157% |

**Why MAPPO Fails:**
1. **On-policy constraint**: Cannot leverage past cooperative experiences
2. **Shared critic instability**: Credit assignment in mixed-motive games is harder
3. **Longer training**: More samples needed, but still converges to suboptimal policy

#### Recommendation
Use MADDPG for CTDE. It achieves equivalent performance to newer methods with established implementations.

---

### 4. CTDE-Discrete (Value Decomposition)

**Algorithms:** QMIX, VDN, COMA

#### Performance Summary

| Algorithm | Mean Return | Cooperation | Trust | Training Time |
|-----------|-------------|-------------|-------|---------------|
| VDN | 51,916 | 49.6% | 61.9% | 2.8 hrs |
| COMA | 44,342 | 54.6% | 58.8% | 3.2 hrs |
| QMIX | 44,340 | 54.6% | 58.8% | 3.1 hrs |

#### Action Discretization Impact

These methods require discretizing the continuous action space (11 bins: 0%, 10%, ..., 100%). This creates:

**Advantages:**
- Faster training (2.8-3.2 hrs vs 8+ hrs for continuous CTDE)
- More stable convergence
- Clearer credit assignment

**Disadvantages:**
- Loss of action precision
- Suboptimal policies at bin boundaries
- 15-20% performance gap vs continuous methods

#### VDN vs QMIX

VDN outperforms QMIX despite simpler architecture:

| Aspect | VDN | QMIX |
|--------|-----|------|
| Mixing | Additive | Monotonic network |
| Return | 51,916 | 44,340 |
| Interpretation | Simpler credit | Complex credit |

**Hypothesis:** In coopetitive settings, additive value decomposition better captures the structure of cooperative surplus. QMIX's nonlinear mixing may overfit to competitive dynamics.

#### Recommendation
Use VDN for discrete action spaces. Simpler is better for coopetition.

---

### 5. Opponent Modeling

**Algorithms:** LOLA, M3DDPG

#### Performance Summary

| Algorithm | Mean Return | Cooperation | Trust | Training Time |
|-----------|-------------|-------------|-------|---------------|
| M3DDPG | 50,916 | 75.8% | 73.5% | 8.1 hrs |
| LOLA | 37,896 | 51.3% | 50.3% | 4.8 hrs |

#### LOLA Analysis

LOLA explicitly models opponent learning dynamics:

```
Policy update: θ ← θ + α∇θ[R(θ, φ + β∇φR(φ, θ))]
```

**Expected behavior:** Account for how own actions affect opponent's future policy
**Observed behavior:** Moderate cooperation (51.3%) but volatile

**Why LOLA Underperforms:**
1. **Lookahead instability**: Second-order gradients amplify noise
2. **Mutual modeling collapse**: When both agents model each other, dynamics become chaotic
3. **Computation cost**: Each update requires multiple forward passes

#### M3DDPG Analysis

M3DDPG adds minimax objective to MADDPG:
- Performance identical to MADDPG (centralized critic dominates)
- Minimax term has negligible effect in cooperative-sum games
- Useful in zero-sum but not mixed-motive settings

#### Recommendation
Opponent modeling provides minimal benefit in coopetitive settings. Use standard CTDE instead.

---

### 6. Population-Based

**Algorithms:** SelfPlay_PPO, FCP

#### Performance Summary

| Algorithm | Mean Return | Cooperation | Trust | Training Time |
|-----------|-------------|-------------|-------|---------------|
| SelfPlay_PPO | 21,136 | 11.8% | 6.8% | 2.1 hrs |
| FCP | 21,019 | 11.7% | 6.8% | 2.0 hrs |

#### The Self-Play Failure Mode

Both methods converge to nearly identical (catastrophic) outcomes:

```
Cooperation: ~12%  →  Near-total defection
Trust: ~7%         →  Complete trust collapse
Return: ~21,000    →  Worse than Random
```

**Root Cause Analysis:**

1. **Nash convergence**: Self-play finds Nash equilibria, which in Prisoner's Dilemma variants are mutual defection
2. **No escape mechanism**: Once defection dominates, no population pressure to cooperate
3. **Opponent pool homogeneity**: Training against similar policies reinforces defection

#### Why This Matters

Self-play has achieved superhuman performance in:
- Go (AlphaGo/Zero)
- Chess (AlphaZero)
- StarCraft (AlphaStar)
- Dota 2 (OpenAI Five)

But all these are **zero-sum games** where Nash = Optimal. In mixed-motive games, Nash equilibria can be arbitrarily bad.

#### Fictitious Co-Play (FCP) Analysis

FCP maintains a population of past policies:
- Samples opponents from historical checkpoints
- Intended to prevent overfitting to current opponent
- In practice: all checkpoints converge to defection, providing no diversity

#### Recommendation
Do not use population-based methods for coopetitive environments without explicit cooperation-promoting mechanisms.

---

### 7. Mean-Field

**Algorithms:** MeanFieldAC

#### Performance Summary

| Algorithm | Mean Return | Cooperation | Trust | Training Time |
|-----------|-------------|-------------|-------|---------------|
| MeanFieldAC | 48,368 | 46.8% | 33.9% | 4.4 hrs |

**Note:** MeanFieldAC only runs on ecosystem environments (PlatformEcosystem-v0, DynamicPartnerSelection-v0) per design.

#### Mean-Field Approximation

MeanFieldAC approximates multi-agent interactions:

```
Q(s, a_i, a_{-i}) ≈ Q(s, a_i, ā)  where ā = mean(a_{-i})
```

**Advantages:**
- Scales to large agent populations
- Reduces observation space complexity
- Faster training

**Disadvantages:**
- Loses individual agent information
- Cannot model targeted cooperation/defection
- Trust dynamics become averaged

#### Ecosystem Performance

| Environment | MeanFieldAC | Best Overall |
|-------------|-------------|--------------|
| PlatformEcosystem-v0 | 92,022 | 146,967 (Constant_035) |
| DynamicPartnerSelection-v0 | 4,714 | 107,314 (Constant_050) |

Mean-field approximation works better in PlatformEcosystem (platform-developer structure) than DynamicPartnerSelection (peer relationships).

#### Recommendation
Use MeanFieldAC only for large-scale ecosystems where individual tracking is infeasible.

---

## Cross-Category Insights

### Training Efficiency

| Category | Mean Return | Training Time | Return/Hour |
|----------|-------------|---------------|-------------|
| Heuristic | 51,795 | 0 hrs | ∞ |
| CTDE-Discrete | 46,866 | 3.0 hrs | 15,622 |
| Independent | 38,211 | 1.8 hrs | 21,228 |
| Population | 21,077 | 2.0 hrs | 10,539 |
| Opponent-Modeling | 44,406 | 6.5 hrs | 6,832 |
| CTDE-Continuous | 47,037 | 11.2 hrs | 4,200 |

**Insight:** When training time matters, ISAC (Independent) provides the best return/hour ratio among learning algorithms.

### Cooperation-Trust-Return Relationship

Algorithms cluster into behavioral profiles:

**Sustainable Coopetition (High Coop, High Trust, High Return):**
- Constant_050, Constant_075, ISAC
- Stable long-term partnerships

**Moderate Success (Medium Coop, Medium Trust, Medium Return):**
- MADDPG, VDN, COMA, LOLA
- Functional but suboptimal cooperation

**Defection Spiral (Low Coop, Low Trust, Low Return):**
- IPPO, SelfPlay_PPO, FCP
- Catastrophic trust collapse

---

## Algorithm Selection Guide

### Decision Tree

```
Is training feasible?
├── No  → Use Constant_050
└── Yes
    ├── Is action space continuous?
    │   ├── Yes
    │   │   ├── Is training time limited?
    │   │   │   ├── Yes → Use ISAC
    │   │   │   └── No  → Use MADDPG
    │   │   └── Is opponent learning modeled?
    │   │       └── Not recommended for coopetition
    │   └── No (Discrete)
    │       └── Use VDN
    └── Is population diversity needed?
        └── Not recommended for coopetition
```

### Quick Reference

| Scenario | Recommended | Avoid |
|----------|-------------|-------|
| Fast prototyping | Constant_050 | Any learning |
| Best learning performance | ISAC | IPPO, Population |
| Multi-agent coordination | MADDPG | MAPPO |
| Discrete actions | VDN | QMIX |
| Large populations | MeanFieldAC | Individual tracking |

---

## Reproducibility Notes

### Seed Sensitivity

| Stability | Algorithms | Recommendation |
|-----------|------------|----------------|
| High (CV < 5%) | ISAC, IPPO, FCP, SelfPlay_PPO | 3 seeds sufficient |
| Low (CV > 20%) | MAPPO, LOLA, QMIX, MADDPG | 5+ seeds required |

### Hyperparameter Sensitivity

**Robust to hyperparameters:**
- Heuristics (no hyperparameters)
- ISAC (entropy auto-tuning helps)

**Sensitive to hyperparameters:**
- MAPPO (clipping, GAE lambda)
- LOLA (lookahead steps, learning rates)
- QMIX (mixer architecture)

---

## Navigation

- [Benchmark Overview](index.md)
- [Trust Dynamics](trust_dynamics.md)
- [Environment Analysis](environment_analysis.md)
- [Research Insights](research_insights.md)
