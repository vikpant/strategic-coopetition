# Environment Classes

Reference for the **unreleased 1.0.8 source candidate**. Counts, horizons and observation shapes below are read from default constructors. See [quick reference](quick_reference.md#available-environments) for the complete comparison table.

## Base classes

`CoopetitionEnv` inherits from `gymnasium.Env` and `AbstractCoopetitionEnv`. Its `reset(seed=None, options=None)` returns `(observation, info)`; `step(joint_action)` returns `(observation, reward_vector, terminated, truncated, info)`.

| Implemented method | Purpose |
|---|---|
| `process_actions(actions)` | Process a complete agent-keyed action round and update shared dynamics |
| `get_observation_for(agent)` | Build an agent-specific observation |
| `get_info_for(agent)` | Build an agent-specific info dictionary |
| `compute_loyalty_score(agent_idx, lookback=10)` | Compute a recent-cooperation score |
| `reset`, `step`, `render`, `close` | Joint-action environment interface |

Trust and reward updates occur inside the action-processing implementation. There are no public base methods named `update_trust()`, `get_observation()` or `get_info()`. Use the returned transition and documented methods rather than those old reference names.

```python
import coopetition_gym as cg

env = cg.make("TrustDilemma-v0", max_steps=10, reward_type="integrated")
obs, info = env.reset(seed=42)
while True:
    obs, rewards, terminated, truncated, info = env.step([50.0, 50.0])
    if terminated or truncated:
        break
env.close()
```

Constructor kwargs can override `EnvironmentConfig` fields. Specific subclasses also accept their documented scenario parameters; use `inspect.signature(type(env).__init__)` to inspect the candidate implementation. Rendering supports the declared `human` and `ansi` modes, not a general RGB-array renderer.

## Dyadic Environments

### TrustDilemmaEnv

`TrustDilemma-v0`: **2 agents**, **100 maximum steps**, joint observation shape `(15,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/dyadic_envs.py).

### PartnerHoldUpEnv

`PartnerHoldUp-v0`: **2 agents**, **100 maximum steps**, joint observation shape `(15,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/dyadic_envs.py).

## Ecosystem Environments

### PlatformEcosystemEnv

`PlatformEcosystem-v0`: **5 agents**, **100 maximum steps**, joint observation shape `(81,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/ecosystem_envs.py).

### DynamicPartnerSelectionEnv

`DynamicPartnerSelection-v0`: **6 agents**, **50 maximum steps**, joint observation shape `(121,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/ecosystem_envs.py).

Reputation can persist across resets. Pass `options={"reset_reputation": True}` to clear it.

## Benchmark Environments

### RecoveryRaceEnv

`RecoveryRace-v0`: **2 agents**, **150 maximum steps**, joint observation shape `(15,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/benchmark_envs.py).

### SynergySearchEnv

`SynergySearch-v0`: **2 agents**, **100 maximum steps**, joint observation shape `(15,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/benchmark_envs.py).

Episode gamma is sampled from `gamma_range=(0.2, 0.9)` after applying the reset seed; `reveal_gamma_in_obs=True` adds it to the joint observation.

## Case Study Environments

### SLCDEnv

`SLCD-v0`: **2 agents**, **100 maximum steps**, joint observation shape `(15,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/case_study_envs.py).

### RenaultNissanEnv

`RenaultNissan-v0`: **2 agents**, **100 maximum steps**, joint observation shape `(15,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/case_study_envs.py).

The default phase is `mature`. Other choices are `formation`, `crisis` and `strained`.

## Extended Environments

### CooperativeNegotiationEnv

`CooperativeNegotiation-v0`: **2 agents**, **100 maximum steps**, joint observation shape `(18,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/extended_envs.py).

### ReputationMarketEnv

`ReputationMarket-v0`: **5 agents**, **100 maximum steps**, joint observation shape `(86,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/extended_envs.py).

## Collective Action Environments

### TeamProductionEnv

`TeamProduction-v0`: **4 agents**, **100 maximum steps**, joint observation shape `(53,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/collective_action_envs.py).

### LoyaltyTeamEnv

`LoyaltyTeam-v0`: **4 agents**, **100 maximum steps**, joint observation shape `(53,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/collective_action_envs.py).

### CoalitionFormationEnv

`CoalitionFormation-v0`: **6 agents**, **150 maximum steps**, joint observation shape `(115,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/collective_action_envs.py).

### ApacheProjectEnv

`ApacheProject-v0`: **40 agents**, **60 maximum steps**, joint observation shape `(4841,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/collective_action_envs.py).

The default phase is `maturity`; phase selection changes the agent population. Validation-score conflicts are recorded in [score provenance](../benchmarks/score_provenance.md).

### PublicGoodsEnv

`PublicGoods-v0`: **5 agents**, **100 maximum steps**, joint observation shape `(81,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/collective_action_envs.py).

## Reciprocity Environments

### ReciprocalDilemmaEnv

`ReciprocalDilemma-v0`: **2 agents**, **100 maximum steps**, joint observation shape `(15,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/reciprocity_envs.py).

### GiftExchangeEnv

`GiftExchange-v0`: **2 agents**, **100 maximum steps**, joint observation shape `(15,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/reciprocity_envs.py).

### IndirectReciprocityEnv

`IndirectReciprocity-v0`: **4 agents**, **150 maximum steps**, joint observation shape `(53,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/reciprocity_envs.py).

### GraduatedSanctionEnv

`GraduatedSanction-v0`: **6 agents**, **200 maximum steps**, joint observation shape `(115,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/reciprocity_envs.py).

### AppleAppStoreEnv

`AppleAppStore-v0`: **3 agents**, **66 maximum steps**, joint observation shape `(31,)` by default. [Constructor and implementation](https://github.com/vikpant/strategic-coopetition/blob/master/coopetition_gym/coopetition_gym/envs/reciprocity_envs.py).

The 66-step default is distinct from any campaign horizon override. Validation-score conflicts are recorded in [score provenance](../benchmarks/score_provenance.md).

See [factories](index.md#factory-functions), [wrappers](wrappers.md), and the [detailed scenario catalog](../environments/index.md). Model defaults and historical validation claims are separate evidence.
