# Extension Version Compatibility

Each extension declares the base `coopetition_gym` version it was developed against.

| Extension | Requires `coopetition_gym` | Last tested | Notes |
| --- | --- | --- | --- |
| `slcd_2d` | `>=0.2.0,<1.0` | 2026-04-21 | Subclasses `SLCDEnv`; relies on `process_actions` hook in `AbstractCoopetitionEnv`. |
