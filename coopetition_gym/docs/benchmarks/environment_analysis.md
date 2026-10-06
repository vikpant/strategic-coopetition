# Environment defaults and grouping provenance

These are the current factory defaults in the 1.0.8 candidate, not inferred
historical run configurations. All twenty environments accept an explicit reward
mode. A recorded run must retain its actual agent count and horizon.

| Environment | Agents | Maximum steps |
| --- | ---: | ---: |
| TrustDilemma-v0 | 2 | 100 |
| PartnerHoldUp-v0 | 2 | 100 |
| PlatformEcosystem-v0 | 5 | 100 |
| DynamicPartnerSelection-v0 | 6 | 50 |
| RecoveryRace-v0 | 2 | 150 |
| SynergySearch-v0 | 2 | 100 |
| SLCD-v0 | 2 | 100 |
| RenaultNissan-v0 | 2 | 100 |
| CooperativeNegotiation-v0 | 2 | 100 |
| ReputationMarket-v0 | 5 | 100 |
| TeamProduction-v0 | 4 | 100 |
| LoyaltyTeam-v0 | 4 | 100 |
| CoalitionFormation-v0 | 6 | 150 |
| ApacheProject-v0 | 40 | 60 |
| PublicGoods-v0 | 5 | 100 |
| ReciprocalDilemma-v0 | 2 | 100 |
| GiftExchange-v0 | 2 | 100 |
| IndirectReciprocity-v0 | 4 | 150 |
| GraduatedSanction-v0 | 6 | 200 |
| AppleAppStore-v0 | 3 | 66 |

The Apache default uses its maturity phase; phase selection can change its agent
count. Former documentation described six agents without establishing the run
configuration. Size training resources from the actual constructed environment,
not from that obsolete table.

`experiments.config.ENVIRONMENTS_BY_TR` preserves historical campaign allocation;
`ANALYSIS_ENVIRONMENTS_BY_TR` preserves the different historical analysis grouping.
Neither mapping should silently replace the other in an archived analysis. The
case-study report mapping is S-LCD → TR-1, Renault–Nissan → TR-2, Apache → TR-3,
and Apple → TR-4. See [score provenance](score_provenance.md) for their distinct
rubrics and unresolved conflicts.

[Previously reported rankings](algorithm_comparison.md) remain historical
observations. [Implementation semantics](implementation_protocol.md) must be
established before using those rankings to compare scientific learning paradigms.
