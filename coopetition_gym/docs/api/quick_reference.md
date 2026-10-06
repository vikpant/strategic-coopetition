# Quick Reference

**Coopetition-Gym 1.0.8 is an unreleased source candidate.** These are constructor defaults in the candidate code, not historical campaign overrides or validation scores.

## Available environments

| ID | Default agents | Default max steps | Joint observation shape |
|---|---:|---:|---|
| `TrustDilemma-v0` | 2 | 100 | `(15,)` |
| `PartnerHoldUp-v0` | 2 | 100 | `(15,)` |
| `PlatformEcosystem-v0` | 5 | 100 | `(81,)` |
| `DynamicPartnerSelection-v0` | 6 | 50 | `(121,)` |
| `RecoveryRace-v0` | 2 | 150 | `(15,)` |
| `SynergySearch-v0` | 2 | 100 | `(15,)` |
| `SLCD-v0` | 2 | 100 | `(15,)` |
| `RenaultNissan-v0` | 2 | 100 | `(15,)` |
| `CooperativeNegotiation-v0` | 2 | 100 | `(18,)` |
| `ReputationMarket-v0` | 5 | 100 | `(86,)` |
| `TeamProduction-v0` | 4 | 100 | `(53,)` |
| `LoyaltyTeam-v0` | 4 | 100 | `(53,)` |
| `CoalitionFormation-v0` | 6 | 150 | `(115,)` |
| `ApacheProject-v0` | 40 | 60 | `(4841,)` |
| `PublicGoods-v0` | 5 | 100 | `(81,)` |
| `ReciprocalDilemma-v0` | 2 | 100 | `(15,)` |
| `GiftExchange-v0` | 2 | 100 | `(15,)` |
| `IndirectReciprocity-v0` | 4 | 150 | `(53,)` |
| `GraduatedSanction-v0` | 6 | 200 | `(115,)` |
| `AppleAppStore-v0` | 3 | 66 | `(31,)` |

ApacheProject defaults to the `maturity` phase; other phases can change the agent count. DynamicPartnerSelection and ReputationMarket permit `n_agents`, PlatformEcosystem permits `n_developers`, and the general `max_steps` override changes the episode limit. Termination can occur before that limit. PettingZoo observation dimensions also depend on observation configuration and AEC's revealed-action fields.

For Apache and Apple case-study score discrepancies, see [score provenance](../benchmarks/score_provenance.md). No constructor default above selects a canonical validation score.

## Factories and objectives

```python
import coopetition_gym as cg

env = cg.make("SLCD-v0", reward_type="private", max_steps=40)
obs, info = env.reset(seed=42)
obs, rewards, terminated, truncated, info = env.step([50.0, 50.0])
env.close()
```

| Factory | Actions | Rewards |
|---|---|---|
| `make` | Joint NumPy array | One-element-per-agent NumPy vector |
| `make_parallel` | Agent-keyed dictionary | Agent-keyed dictionary of scalars |
| `make_aec` | Current agent's action, or `None` when finished | `last()` supplies the selected agent's accumulated reward |

`gymnasium.make("coopetition_gym:SLCD-v0")` is also supported and retains vector rewards. Reward modes are `private`, `integrated` and `cooperative`; invalid values and unknown constructor keywords are rejected. Cooperative base reward is mean integrated utility, with mechanism-specific modifiers applied afterward.

## Info dictionaries

The joint-action base API includes `step`, `mean_trust`, `mean_reputation_damage`, `total_value`, `mean_cooperation` and `cooperation_rate`; steps also report `actions`. Subclasses may add diagnostics. Agent-specific PettingZoo info includes `step`, `own_action`, `own_trust_mean` and `cooperation_rate`; do not assume every legacy diagnostic is present there.

## Observation configuration

```python
from coopetition_gym import make_parallel, ObservationConfig

env = make_parallel("TrustDilemma-v0",
                    obs_config=ObservationConfig.realistic_asymmetry())
observations, infos = env.reset(seed=42)
env.close()
```

See [API examples](index.md), [environment classes](environments.md), [wrapper behavior](wrappers.md), [configuration](configuration.md), and the [quickstart](../tutorials/quickstart.md).
