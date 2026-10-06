"""Analysis pipeline for the Coopetition-Gym v1 benchmark dataset.

This module summarizes explicit result records into CSVs, text tables, and
figures. Stable algorithm IDs identify the historical implementation roster;
they do not establish equivalence to independent-learning or CTDE paradigms.
Current outputs are not a certification that historical paper tables have been
reproduced. Source/configuration provenance must be established separately.

Consolidates:

* ``analysis/analyze_all.py`` — returns summary, tier rankings, oracle
  comparison (with Gap% formula), MASAC instability report, learning-curve
  exports, publication plots.
* ``analyze_reward_ablation.py`` — private vs integrated vs cooperative
  reward comparison, information premium, case study highlights, incentive
  gradient analysis.

Subcommands::

    all                Run every analysis (matches the original campaign's
                       analyze_all.py entry point). Produces all outputs in
                       one pass.
    returns-summary    Per-(algo, env) mean/std/sem across seeds. Writes
                       ``returns_summary.csv``.
    oracle-comparison  Gap% = (algo - oracle_ref) / |oracle_ref| * 100 by
                       TR tier. Writes ``oracle_comparison.txt``.
    tier-summary       Aggregate ranking table per TR tier. Writes
                       ``tier_summary.txt``.
    masac-instability  Detailed MASAC instability diagnostic. Writes
                       ``masac_instability.txt``.
    training-metrics   Final gradient metric values per algo/env. Writes
                       ``training_metrics_final.csv``.
    learning-curves    Export per-(algo, env) training-return CSVs.
    plots              Generate publication figures (PNG).
    reward-ablation    Compare returns across private/integrated/cooperative
                       reward configurations.

Inputs are recursive JSON or JSONL native result records. Select a treatment
and evaluation fold explicitly when the input contains several arms or folds.
The ``returns_summary.csv`` records the actual eligible seeds; the oracle comparison and tier summary
consume the aggregated returns directly.

Gap% definition (used throughout):
    ``Gap = (Algorithm_return - Oracle_reference_return) / |Oracle_reference_return| * 100``.
    Positive values indicate the algorithm exceeds the oracle reference.
    See :data:`experiments.config.ENV_ORACLE_REF` for the reference oracle
    per environment.

Expected training-result JSON schema:
    ``algorithm``, ``environment``, ``training_seed``, ``status``,
    ``tr_mode``, ``training_time_seconds``, and a ``metrics`` subdict with
    ``mean_return``, ``std_return``, ``mean_cooperation_rate``,
    ``training_returns``, ``training_timesteps``, ``training_metrics``,
    and (optionally) ``tr_metrics``.

Usage::

    # Full analysis (all subcommands in one pass)
    python -m experiments.analyze all \\
        --input-dir data/training/baseline_integrated/raw/ \\
        --output-dir data/analysis/

    # Individual subcommands
    python -m experiments.analyze oracle-comparison \\
        --input-dir data/training/baseline_integrated/raw/ \\
        --output data/analysis/oracle_comparison.txt

    # Reward-type ablation comparison (needs all three input directories)
    python -m experiments.analyze reward-ablation \\
        --input-baseline    data/training/baseline_integrated/raw/ \\
        --input-private     data/training/ablation_private/raw/ \\
        --input-cooperative data/training/ablation_cooperative/raw/ \\
        --output-dir        data/analysis/reward_ablation/
"""

import os
import re
import json
import math
import csv
import warnings
import numpy as np
from collections import defaultdict

from .records import RecordError, comparison_context, ensure_comparable, load_records, is_successful_result
from .config import ANALYSIS_ENVIRONMENTS_BY_TR


BASE = os.path.dirname(os.path.abspath(__file__))
MERGED_TR123 = os.path.join(BASE, "merged", "tr1_2_3")
MERGED_TR4 = os.path.join(BASE, "merged", "tr4")
OUTPUT_DIR = os.path.join(BASE, "output")

# ── Algorithm categorisation ────────────────────────────────────────────────

TRAINING_ALGOS = [
    "MADDPG", "MATD3", "M3DDPG", "MASAC", "ISAC",
    "QMIX", "VDN", "COMA",
    "IPPO", "IA2C", "MAPPO", "MeanFieldAC",
    "FCP", "SelfPlay_PPO", "LOLA",
    "IndependentREINFORCE", "Random", "TitForTat",
]

ORACLE_ALGOS = [
    "Oracle_Nash",
    "Oracle_TrustAware", "Oracle_Equilibrium",
    "Oracle_Loyalty", "Oracle_SocialOptimum",
    "Oracle_ReciprocityEquilibrium", "Oracle_BoundedReciprocity",
]

# Historical analysis grouping; distinct from campaign allocation/source origin.
TR1_ENVS = list(ANALYSIS_ENVIRONMENTS_BY_TR["tr1"])
TR2_ENVS = list(ANALYSIS_ENVIRONMENTS_BY_TR["tr2"])
TR3_ENVS = list(ANALYSIS_ENVIRONMENTS_BY_TR["tr3"])
TR4_ENVS = list(ANALYSIS_ENVIRONMENTS_BY_TR["tr4"])

ENV_TO_TR = {}
for e in TR1_ENVS: ENV_TO_TR[e] = "TR-1"
for e in TR2_ENVS: ENV_TO_TR[e] = "TR-2"
for e in TR3_ENVS: ENV_TO_TR[e] = "TR-3"
for e in TR4_ENVS: ENV_TO_TR[e] = "TR-4"

# ACTUAL oracle-to-environment coverage (derived from experiment files, not paper tier):
#   Oracle_Equilibrium: DynamicPartnerSelection, PartnerHoldUp, PlatformEcosystem,
#                       SynergySearch (TR-1) + RenaultNissan (TR-2) — TR-1 game-theoretic oracle
#   Oracle_TrustAware:  CooperativeNegotiation, RecoveryRace, ReputationMarket,
#                       SLCD (TR-2) + TrustDilemma (TR-1) — TR-2 trust-dynamics oracle
#   Oracle_Nash:        All TR-3 envs — Nash equilibrium (LOWER bound for TR-3)
#   Oracle_Loyalty:     All TR-3 envs — Social optimum (UPPER bound for TR-3)
#   Oracle_SocialOptimum: All TR-3 envs — Social optimum (same as Loyalty)
#   Oracle_ReciprocityEquilibrium: All TR-4 envs
#   Oracle_BoundedReciprocity: All TR-4 envs
#
# Note: TrustDilemma (TR-1 env) is benchmarked by Oracle_TrustAware (TR-2 oracle)
#       RenaultNissan (TR-2 env) is benchmarked by Oracle_Equilibrium (TR-1 oracle)
ENV_TO_ORACLE = {
    # TR-1 envs
    "TrustDilemma-v0":            ["Oracle_TrustAware"],
    "PartnerHoldUp-v0":           ["Oracle_Equilibrium"],
    "PlatformEcosystem-v0":       ["Oracle_Equilibrium"],
    "DynamicPartnerSelection-v0": ["Oracle_Equilibrium"],
    "SynergySearch-v0":           ["Oracle_Equilibrium"],
    # TR-2 envs
    "RecoveryRace-v0":            ["Oracle_TrustAware"],
    "CooperativeNegotiation-v0":  ["Oracle_TrustAware"],
    "ReputationMarket-v0":        ["Oracle_TrustAware"],
    "SLCD-v0":                    ["Oracle_TrustAware"],
    "RenaultNissan-v0":           ["Oracle_Equilibrium"],
    # TR-3 envs (3 oracles: Nash=Nash-eq lower bound [LB], Loyalty/SocialOptimum=social optimum [UB])
    "ApacheProject-v0":           ["Oracle_Nash", "Oracle_Loyalty", "Oracle_SocialOptimum"],
    "CoalitionFormation-v0":      ["Oracle_Nash", "Oracle_Loyalty", "Oracle_SocialOptimum"],
    "LoyaltyTeam-v0":             ["Oracle_Nash", "Oracle_Loyalty", "Oracle_SocialOptimum"],
    "PublicGoods-v0":             ["Oracle_Nash", "Oracle_Loyalty", "Oracle_SocialOptimum"],
    "TeamProduction-v0":          ["Oracle_Nash", "Oracle_Loyalty", "Oracle_SocialOptimum"],
    # TR-4 envs — ReciprocityEquilibrium=Nash-style lower bound, BoundedReciprocity=cooperation upper bound
    "ReciprocalDilemma-v0":       ["Oracle_ReciprocityEquilibrium", "Oracle_BoundedReciprocity"],
    "GiftExchange-v0":            ["Oracle_ReciprocityEquilibrium", "Oracle_BoundedReciprocity"],
    "IndirectReciprocity-v0":     ["Oracle_ReciprocityEquilibrium", "Oracle_BoundedReciprocity"],
    "GraduatedSanction-v0":       ["Oracle_ReciprocityEquilibrium", "Oracle_BoundedReciprocity"],
    "AppleAppStore-v0":           ["Oracle_ReciprocityEquilibrium", "Oracle_BoundedReciprocity"],
}

# Per-env reference oracle for Gap% column.
# Gap% = (Algorithm - Oracle_ref) / |Oracle_ref| * 100
# Positive = algorithm EXCEEDS reference; Negative = below reference.
# For TR-3: reference = Oracle_Loyalty (social optimum, the highest achievable).
# For TR-1/2: reference = the single oracle covering that env (Nash-style equilibrium).
# For TR-4: reference = Oracle_BoundedReciprocity (cooperation target).
ENV_ORACLE_REF = {
    "TrustDilemma-v0":            "Oracle_TrustAware",
    "PartnerHoldUp-v0":           "Oracle_Equilibrium",
    "PlatformEcosystem-v0":       "Oracle_Equilibrium",
    "DynamicPartnerSelection-v0": "Oracle_Equilibrium",
    "SynergySearch-v0":           "Oracle_Equilibrium",
    "RecoveryRace-v0":            "Oracle_TrustAware",
    "CooperativeNegotiation-v0":  "Oracle_TrustAware",
    "ReputationMarket-v0":        "Oracle_TrustAware",
    "SLCD-v0":                    "Oracle_TrustAware",
    "RenaultNissan-v0":           "Oracle_Equilibrium",
    "ApacheProject-v0":           "Oracle_Loyalty",
    "CoalitionFormation-v0":      "Oracle_Loyalty",
    "LoyaltyTeam-v0":             "Oracle_Loyalty",
    "PublicGoods-v0":             "Oracle_Loyalty",
    "TeamProduction-v0":          "Oracle_Loyalty",
    "ReciprocalDilemma-v0":       "Oracle_BoundedReciprocity",
    "GiftExchange-v0":            "Oracle_BoundedReciprocity",
    "IndirectReciprocity-v0":     "Oracle_BoundedReciprocity",
    "GraduatedSanction-v0":       "Oracle_BoundedReciprocity",
    "AppleAppStore-v0":           "Oracle_BoundedReciprocity",
}

# Oracle groupings for tier comparison tables
# TR-1: Oracle_Equilibrium (covers 4 TR-1 envs + RenaultNissan)
#        Oracle_TrustAware (covers TrustDilemma + 4 TR-2 envs)
# For per-tier table, show the primary oracle for that tier.
TIER_ORACLE = {
    "TR-1": ["Oracle_Equilibrium", "Oracle_TrustAware"],   # Equilibrium=primary, TrustAware for TrustDilemma
    "TR-2": ["Oracle_TrustAware", "Oracle_Equilibrium"],   # TrustAware=primary, Equilibrium for RenaultNissan
    "TR-3": ["Oracle_Nash", "Oracle_Loyalty", "Oracle_SocialOptimum"],  # Nash=lower, Loyalty/Social=upper
    "TR-4": ["Oracle_ReciprocityEquilibrium", "Oracle_BoundedReciprocity"],
}

KNOWN_ENVS = TR1_ENVS + TR2_ENVS + TR3_ENVS + TR4_ENVS
ENV_PAT = "|".join(re.escape(e) for e in KNOWN_ENVS)


# ── Data loading ─────────────────────────────────────────────────────────────

def parse_filename(fname):
    """Return (algo, env, seed) or (None, None, None)."""
    m = re.search(rf'({ENV_PAT})_(\d+)\.json$', fname)
    if not m:
        return None, None, None
    env = m.group(1)
    seed = m.group(2)
    algo = fname[:m.start()].rstrip("_")
    return algo, env, seed


def load_directory(dirpath, *, reward_type=None, seeds=None, include_invalid=False):
    """Load recursive native JSON/JSONL results with explicit provenance checks."""
    records = []
    for data in load_records(dirpath, reward_type=reward_type, seeds=seeds,
                             successful_only=not include_invalid):
        metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
        records.append({
            "algo": data["algorithm"], "env": data["environment"],
            "seed": data["training_seed"], "tr": ENV_TO_TR.get(data["environment"], "?"),
            "status": data.get("status"), "mean_return": metrics.get("mean_return"),
            "std_return": metrics.get("std_return"),
            "mean_cooperation": metrics.get("mean_cooperation_rate"),
            "training_timesteps_list": metrics.get("training_timesteps", []),
            "training_returns": metrics.get("training_returns", []),
            "training_metrics": metrics.get("training_metrics", {}),
            "tr_metrics": metrics.get("tr_metrics", {}),
            "training_time_s": data.get("training_time_seconds"),
            "reward_type": data.get("reward_type"), "record": data,
        })
    return records


def _load_input(args, *, include_invalid=False):
    return load_directory(args.input_dir, reward_type=getattr(args, "reward_type", None),
                          seeds=getattr(args, "seeds", None), include_invalid=include_invalid)


# ── Statistics helpers ────────────────────────────────────────────────────────

def safe_float(v):
    if v is None: return float("nan")
    try:
        f = float(v)
        return f if math.isfinite(f) else float("nan")
    except Exception:
        return float("nan")


def seed_stats(values):
    """Compute mean, std, sem across seed values (ignoring nan/inf)."""
    finite = [safe_float(v) for v in values]
    finite = [v for v in finite if not math.isnan(v)]
    if not finite:
        return float("nan"), float("nan"), float("nan"), 0
    n = len(finite)
    mean = sum(finite) / n
    if n > 1:
        std = math.sqrt(sum((x - mean) ** 2 for x in finite) / (n - 1))
        sem = std / math.sqrt(n)
    else:
        std = 0.0
        sem = 0.0
    return mean, std, sem, n


# ── Section 1: Returns Summary ────────────────────────────────────────────────

def compute_returns_summary(records):
    """Aggregate successful finite independent seed records without pooling arms."""
    eligible = [r for r in records if r.get("status") == "success"
                and math.isfinite(safe_float(r.get("mean_return")))
                and ("record" not in r or is_successful_result(r["record"]))]
    if eligible and all("record" in r for r in eligible):
        ensure_comparable([r["record"] for r in eligible])
    grouped, cells = defaultdict(list), set()
    for record in eligible:
        cell = (record["algo"], record["env"], record["seed"])
        if cell in cells:
            raise RecordError(f"repeated seed cell in summary: {cell}")
        cells.add(cell)
        grouped[cell[:2]].append(record)
    results = {}
    for pair, entries in grouped.items():
        vals = [safe_float(r["mean_return"]) for r in entries]
        mean, std, sem, n = seed_stats(vals)
        results[pair] = {"mean": mean, "std": std, "sem": sem, "n": n, "vals": vals,
                         "seeds": sorted(r["seed"] for r in entries),
                         "reward_type": entries[0].get("reward_type"),
                         "treatment_sources": sorted({r.get("record", {}).get("_reward_type_assumption", "record") for r in entries}),
                         "configuration": comparison_context(entries[0]["record"]) if "record" in entries[0] else {}}
    return results


def write_returns_csv(summary, out_path):
    """Write returns summary to CSV."""
    rows = []
    for (algo, env), s in sorted(summary.items()):
        rows.append({
            "algorithm": algo,
            "environment": env,
            "comparison_basis": "historical-roster",
            "tr": ENV_TO_TR.get(env, "?"),
            "n_seeds": s["n"],
            "seeds": json.dumps(s["seeds"]),
            "reward_type": s["reward_type"],
            "treatment_sources": json.dumps(s["treatment_sources"]),
            "configuration": json.dumps(s["configuration"], sort_keys=True),
            "mean_return": round(s["mean"], 4) if not math.isnan(s["mean"]) else "NA",
            "std_return": round(s["std"], 4) if not math.isnan(s["std"]) else "NA",
            "sem_return": round(s["sem"], 4) if not math.isnan(s["sem"]) else "NA",
            "min_val": round(min(safe_float(v) for v in s["vals"]), 4),
            "max_val": round(max(safe_float(v) for v in s["vals"]), 4),
        })
    if not rows:
        raise RecordError("no eligible algorithm results to summarize")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote: {out_path}")


# ── Section 2: Oracle Comparison ─────────────────────────────────────────────

def oracle_comparison_table(records):
    """Build oracle comparison tables per TR tier using correct per-env oracle coverage.

    Key structure (derived from actual experiment files):
      Oracle_Equilibrium: TR-1 oracle — covers 4 TR-1 envs + RenaultNissan (TR-2)
      Oracle_TrustAware:  TR-2 oracle — covers 4 TR-2 envs + TrustDilemma (TR-1)
      Oracle_Nash:        TR-3 Nash equilibrium — LOWER BOUND for TR-3 envs
      Oracle_Loyalty:     TR-3 social optimum  — UPPER BOUND for TR-3 envs
      Oracle_SocialOptimum: TR-3 social optimum — same as Loyalty
      Oracle_ReciprocityEquilibrium: TR-4 lower bound
      Oracle_BoundedReciprocity:     TR-4 upper bound
    """
    summary = compute_returns_summary(records)

    lines = []
    lines.append("=" * 105)
    lines.append("ORACLE COMPARISON TABLES — Training Algorithms vs Oracle Benchmarks")
    lines.append("=" * 105)
    lines.append("")
    lines.append("Comparison basis: historical implementation roster; labels do not certify algorithm paradigms.")
    lines.append("Values: Mean Return ± Std (observed eligible seeds; see returns_summary.csv).")
    lines.append("Gap%: (Algorithm - Oracle_ref) / |Oracle_ref| × 100.")
    lines.append("  Positive = algorithm EXCEEDS the reference oracle.")
    lines.append("  Negative = algorithm is BELOW the reference oracle.")
    lines.append("  Reference oracle per env is listed in ENV_ORACLE_REF:")
    lines.append("    TR-1/2: the Nash-style equilibrium oracle for that env.")
    lines.append("    TR-3:   Oracle_Loyalty (social optimum upper bound).")
    lines.append("    TR-4:   Oracle_BoundedReciprocity (cooperation upper bound).")
    lines.append("")
    lines.append("Oracle structure (corrected from actual experiment files):")
    lines.append("  TR-1: Oracle_Equilibrium covers DynamicPartnerSel, PartnerHoldUp,")
    lines.append("        PlatformEcosystem, SynergySearch, RenaultNissan(TR-2)")
    lines.append("        Oracle_TrustAware covers TrustDilemma(TR-1), RecoveryRace,")
    lines.append("        CooperativeNeg, ReputationMarket, SLCD")
    lines.append("  TR-3: Oracle_Nash=Nash equilibrium (lower bound),")
    lines.append("        Oracle_Loyalty=Oracle_SocialOptimum=social optimum (upper bound)")
    lines.append("")

    for tier, envs in [("TR-1", TR1_ENVS), ("TR-2", TR2_ENVS),
                       ("TR-3", TR3_ENVS), ("TR-4", TR4_ENVS)]:
        oracles_for_tier = TIER_ORACLE[tier]

        lines.append("=" * 105)
        lines.append(f"  {tier} ENVIRONMENTS: {', '.join(e.replace('-v0','') for e in envs)}")

        # For each env, show which oracle covers it
        env_oracle_notes = []
        for env in envs:
            oc_list = ENV_TO_ORACLE.get(env, [])
            env_oracle_notes.append(f"{env.replace('-v0','')}/{'/'.join(o.replace('Oracle_','') for o in oc_list)}")
        lines.append(f"  Coverage: {', '.join(env_oracle_notes)}")
        lines.append("=" * 105)

        # Oracle reference mean per env for Gap% calculation.
        # Uses ENV_ORACLE_REF (primary reference oracle per env).
        # Gap% = (Algo - Oracle_ref) / |Oracle_ref| * 100
        oracle_ref_mean = {}  # env -> reference oracle mean
        for env in envs:
            ref_name = ENV_ORACLE_REF.get(env)
            if ref_name:
                s = summary.get((ref_name, env))
                if s and not math.isnan(s["mean"]):
                    oracle_ref_mean[env] = s["mean"]

        # Header row
        env_labels = [e.replace("-v0", "")[:14] for e in envs]
        hdr = f"  {'Algorithm':<24}" + "".join(f"  {e:>21}" for e in env_labels) + f"  {'Avg Gap%':>10}"
        lines.append(hdr)
        lines.append("  " + "-"*24 + ("  " + "-"*21) * len(envs) + "  " + "-"*10)

        # Oracle rows first — show all oracles that have data for any env in this tier
        shown_oracles = set()
        for env in envs:
            for oc in ENV_TO_ORACLE.get(env, []):
                shown_oracles.add(oc)

        for oc in sorted(shown_oracles):
            has_any = any(summary.get((oc, env)) and not math.isnan(summary[(oc, env)]["mean"]) for env in envs)
            if not has_any:
                continue
            # Label: mark lower/upper bounds
            if oc in ("Oracle_Nash", "Oracle_ReciprocityEquilibrium"):
                label = f"{oc} [LB]"
            elif oc in ("Oracle_Loyalty", "Oracle_SocialOptimum",
                        "Oracle_BoundedReciprocity"):
                label = f"{oc} [UB]"
            else:  # Oracle_Equilibrium, Oracle_TrustAware — Nash-style reference oracles
                label = f"{oc} [ref]"
            row = f"  {label:<24}"
            for env in envs:
                # Only show oracle value if this oracle covers this env
                if oc in ENV_TO_ORACLE.get(env, []):
                    s = summary.get((oc, env))
                    if s and not math.isnan(s["mean"]):
                        cell = f"{s['mean']:>11.1f}±{s['std']:>8.1f}"
                        row += f"  {cell:>21}"
                    else:
                        row += f"  {'N/A':>21}"
                else:
                    row += f"  {'---':>21}"
            row += f"  {'--':>10}"
            lines.append(row)

        lines.append("  " + "·" * (26 + 23 * len(envs) + 12))

        # Training algo rows (sorted by avg gap vs upper-bound oracle)
        algo_data = []
        for algo in TRAINING_ALGOS:
            cells = []
            gap_vals = []
            for env in envs:
                s = summary.get((algo, env))
                if s is None or math.isnan(s["mean"]):
                    cells.append(None)
                    continue
                cells.append((s["mean"], s["std"]))
                # Gap% = (Algo - Oracle_ref) / |Oracle_ref| * 100
                # Positive = exceeds reference; Negative = below reference
                if env in oracle_ref_mean and oracle_ref_mean[env] != 0:
                    gap = (s["mean"] - oracle_ref_mean[env]) / abs(oracle_ref_mean[env]) * 100
                    gap_vals.append(gap)
            avg_gap = sum(gap_vals) / len(gap_vals) if gap_vals else float("nan")
            algo_data.append((algo, cells, avg_gap))

        # Sort descending: highest Gap% first (most exceeds reference = best)
        algo_data.sort(key=lambda x: x[2] if not math.isnan(x[2]) else -1e9, reverse=True)

        for algo, cells, avg_gap in algo_data:
            row = f"  {algo:<24}"
            for cell in cells:
                if cell is None:
                    row += f"  {'--':>21}"
                else:
                    row += f"  {cell[0]:>11.1f}±{cell[1]:>8.1f}"
            if not math.isnan(avg_gap):
                row += f"  {avg_gap:>+9.1f}%"
            else:
                row += f"  {'--':>10}"
            lines.append(row)

        lines.append("")

    return "\n".join(lines)


# ── Section 3: MASAC Instability ──────────────────────────────────────────────

def masac_instability_report(records):
    """Report only observed MASAC diagnostics, without historical counts/causes."""
    masac = [record for record in records if record["algo"] == "MASAC"]
    lines = ["MASAC NUMERICAL DIAGNOSTICS", f"Input records: {len(masac)}",
             "Counts describe supplied records; no historical corpus size is assumed."]
    if not masac:
        lines.append("No MASAC records supplied.")
        return "\n".join(lines)
    unstable, missing = 0, 0
    for record in sorted(masac, key=lambda r: (r["env"], r["seed"])):
        metrics = record.get("training_metrics") if isinstance(record.get("training_metrics"), dict) else {}
        observations = []
        for name in ("critic_loss", "actor_loss"):
            for point in metrics.get(name, []):
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    observations.append((name, point[0], point[1]))
        nonfinite = [(name, step) for name, step, value in observations
                     if not math.isfinite(safe_float(value))]
        unstable += bool(nonfinite)
        missing += not observations
        final_finite = math.isfinite(safe_float(record.get("mean_return")))
        lines.append(f"{record['env']} seed={record['seed']} status={record.get('status')} "
                     f"final_return_finite={final_finite} nonfinite_loss_points={len(nonfinite)} "
                     f"loss_points={len(observations)} first_nonfinite={nonfinite[0] if nonfinite else None}")
    lines.append(f"Records with observed nonfinite loss: {unstable}/{len(masac)}")
    lines.append(f"Records without actor/critic loss observations: {missing}/{len(masac)}")
    lines.append("A finite evaluation return does not establish successful or numerically stable training.")
    return "\n".join(lines)


# ── Section 4: Training Metrics Final Values ──────────────────────────────────

def training_metrics_summary(records):
    """Compute final gradient metric values per (algo, env) across seeds."""
    # Group by (algo, env)
    grouped = defaultdict(lambda: defaultdict(list))  # (algo,env) -> metric_name -> [values]

    for r in records:
        tm = r["training_metrics"]
        if not tm:
            continue
        for metric_name, pts in tm.items():
            if not pts:
                continue
            # Take last finite value
            last_val = None
            for step, val in reversed(pts):
                if math.isfinite(float(val)) if isinstance(val, (int, float)) else False:
                    last_val = float(val)
                    break
            if last_val is not None:
                grouped[(r["algo"], r["env"])][metric_name].append(last_val)

    rows = []
    for (algo, env), metrics in sorted(grouped.items()):
        for mname, vals in sorted(metrics.items()):
            mean, std, sem, n = seed_stats(vals)
            rows.append({
                "algorithm": algo,
                "environment": env,
                "tr": ENV_TO_TR.get(env, "?"),
                "metric": mname,
                "n_seeds": n,
                "mean": round(mean, 6) if not math.isnan(mean) else "NA",
                "std": round(std, 6) if not math.isnan(std) else "NA",
            })
    return rows


# ── Section 5: Learning Curves ────────────────────────────────────────────────

def export_learning_curves(records, out_dir):
    """Export per-algo per-env training return curves as CSVs.

    Format: CSV with columns [seed, episode, timestep, return]
    One file per (algo, env).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Group by (algo, env)
    grouped = defaultdict(list)
    for r in records:
        if r["training_returns"] and r["algo"] in TRAINING_ALGOS:
            grouped[(r["algo"], r["env"])].append(r)

    file_count = 0
    for (algo, env), runs in sorted(grouped.items()):
        rows = []
        for run in sorted(runs, key=lambda x: x["seed"]):
            tr_list = run["training_returns"]
            ts_list = run["training_timesteps_list"]
            seed = run["seed"]
            for i, ret in enumerate(tr_list):
                ts = ts_list[i] if i < len(ts_list) else None
                rows.append({"algo": algo, "env": env, "seed": seed,
                             "episode": i+1, "timestep": ts,
                             "return": round(safe_float(ret), 4)})
        if rows:
            fname = f"{algo}_{env}.csv"
            fpath = os.path.join(out_dir, fname)
            with open(fpath, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["algo", "env", "seed", "episode", "timestep", "return"])
                w.writeheader()
                w.writerows(rows)
            file_count += 1
    return file_count


# ── Section 6: Summary Per-Tier Table ────────────────────────────────────────

def tier_summary_table(summary):
    """Compact per-tier mean return table across all environments.

    Oracle rows use the correct per-env oracle coverage (derived from actual files).
    For TR-3: Oracle_Nash shown as lower bound [LB], Loyalty/SocialOptimum as upper [UB].
    """
    lines = []
    lines.append("=" * 80)
    lines.append("PER-TIER AGGREGATE RETURNS — Averaged Across All Tier Environments")
    lines.append("=" * 80)
    lines.append("(Mean ± Std across seeds AND environments within tier)")
    lines.append("Oracle note: [UB]=social optimum upper bound, [LB]=Nash equilibrium lower bound, [ref]=equilibrium reference")
    lines.append("")

    # Correct oracle coverage per tier: only include oracle values for envs it covers
    def oracle_vals_for_tier(oc, envs):
        """Return oracle mean_return values only for environments this oracle covers."""
        vals = []
        for env in envs:
            if oc in ENV_TO_ORACLE.get(env, []):
                s = summary.get((oc, env))
                if s and not math.isnan(s["mean"]):
                    vals.append(s["mean"])
        return vals

    for tier, envs in [("TR-1", TR1_ENVS), ("TR-2", TR2_ENVS),
                       ("TR-3", TR3_ENVS), ("TR-4", TR4_ENVS)]:
        lines.append(f"  {tier} ({', '.join(e.replace('-v0','') for e in envs)})")
        lines.append(f"  {'Algorithm':<28} {'Mean':>12} {'Std':>10} {'N':>4}")
        lines.append("  " + "-"*28 + " " + "-"*12 + " " + "-"*10 + " " + "-"*4)

        # Oracle rows — use correct per-env coverage
        shown = set()
        for env in envs:
            for oc in ENV_TO_ORACLE.get(env, []):
                if oc not in shown:
                    shown.add(oc)
                    vals = oracle_vals_for_tier(oc, envs)
                    if not vals:
                        continue
                    m, sd, _, n = seed_stats(vals)
                    m_str = f"{m:>12,.1f}" if not math.isnan(m) else f"{'N/A':>12}"
                    sd_str = f"{sd:>10,.1f}" if not math.isnan(sd) else f"{'N/A':>10}"
                    if oc in ("Oracle_Loyalty", "Oracle_SocialOptimum", "Oracle_BoundedReciprocity"):
                        suffix = " [UB]"
                    elif oc in ("Oracle_Nash", "Oracle_ReciprocityEquilibrium"):
                        suffix = " [LB]"
                    else:  # Oracle_Equilibrium, Oracle_TrustAware — Nash-style reference oracles
                        suffix = " [ref]"
                    label = f"{oc}{suffix}"
                    lines.append(f"  {label:<28} {m_str} {sd_str} {n:>4}  ← ORACLE")

        lines.append("  " + "·"*56)

        # Training algo rows
        algo_rows = []
        for algo in TRAINING_ALGOS:
            vals = []
            for env in envs:
                s = summary.get((algo, env))
                if s and not math.isnan(s["mean"]):
                    vals.append(s["mean"])
            if vals:
                m, sd, _, n = seed_stats(vals)
                algo_rows.append((algo, m, sd, n))

        for algo, m, sd, n in sorted(algo_rows, key=lambda x: -x[1]):
            m_str = f"{m:>12,.1f}" if not math.isnan(m) else f"{'N/A':>12}"
            sd_str = f"{sd:>10,.1f}" if not math.isnan(sd) else f"{'N/A':>10}"
            lines.append(f"  {algo:<28} {m_str} {sd_str} {n:>4}")

        lines.append("")

    return "\n".join(lines)


# ── Matplotlib plots ──────────────────────────────────────────────────────────

def make_plots(summary, records, plot_dir):
    """Generate publication figures. Requires matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    os.makedirs(plot_dir, exist_ok=True)

    # Color palette for algorithms
    COLORS = {
        "MADDPG": "#1f77b4", "MATD3": "#aec7e8", "M3DDPG": "#ffbb78",
        "MASAC": "#2ca02c", "ISAC": "#98df8a",
        "QMIX": "#d62728", "VDN": "#ff9896", "COMA": "#9467bd",
        "IPPO": "#c5b0d5", "IA2C": "#8c564b", "MAPPO": "#c49c94",
        "MeanFieldAC": "#e377c2", "FCP": "#f7b6d2",
        "SelfPlay_PPO": "#7f7f7f", "LOLA": "#c7c7c7",
        "IndependentREINFORCE": "#bcbd22", "Random": "#dbdb8d",
        "TitForTat": "#17becf",
    }

    # ── Figure 1: Per-tier bar chart ──────────────────────────────────────────
    for tier, envs in [("TR-1", TR1_ENVS), ("TR-2", TR2_ENVS),
                       ("TR-3", TR3_ENVS), ("TR-4", TR4_ENVS)]:
        algos_in_tier = []
        means = []
        stds = []

        for algo in TRAINING_ALGOS:
            vals = []
            for env in envs:
                s = summary.get((algo, env))
                if s and not math.isnan(s["mean"]):
                    vals.append(s["mean"])
            if vals:
                m, sd, _, _ = seed_stats(vals)
                if not math.isnan(m):
                    algos_in_tier.append(algo)
                    means.append(m)
                    stds.append(sd)

        # Oracle reference lines
        oracle_means = {}
        for oc in TIER_ORACLE[tier]:
            oc_vals = []
            for env in envs:
                s = summary.get((oc, env))
                if s and not math.isnan(s["mean"]):
                    oc_vals.append(s["mean"])
            if oc_vals:
                oracle_means[oc] = sum(oc_vals) / len(oc_vals)

        if not algos_in_tier:
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        colors = [COLORS.get(a, "#888888") for a in algos_in_tier]
        bars = ax.bar(range(len(algos_in_tier)), means, yerr=stds,
                      color=colors, edgecolor="black", linewidth=0.5,
                      capsize=3, error_kw={"linewidth": 1})
        ax.set_xticks(range(len(algos_in_tier)))
        ax.set_xticklabels(algos_in_tier, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean Return (avg across environments)", fontsize=10)
        ax.set_title(f"{tier} — Algorithm Performance (eligible seeds, mean ± std across {len(envs)} envs)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        # Oracle reference lines
        oc_colors = ["gold", "orange", "darkorange"]
        for i, (oc_name, oc_val) in enumerate(oracle_means.items()):
            ax.axhline(oc_val, color=oc_colors[i % len(oc_colors)],
                       linestyle="--", linewidth=1.5, label=f"{oc_name}: {oc_val:,.0f}")
        ax.legend(fontsize=8)

        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"returns_{tier.replace('-','')}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: returns_{tier.replace('-','')}.png")

    # ── Figure 2: Learning curves for top-3 algorithms per tier ──────────────
    # Group training returns by (algo, env)
    curve_data = defaultdict(list)  # (algo, env) -> list of (seed, [returns])
    for r in records:
        if r["algo"] in TRAINING_ALGOS and r["training_returns"]:
            curve_data[(r["algo"], r["env"])].append((r["seed"], r["training_returns"]))

    for tier, envs in [("TR-1", TR1_ENVS), ("TR-2", TR2_ENVS),
                       ("TR-3", TR3_ENVS), ("TR-4", TR4_ENVS)]:
        # Pick first env in tier for representative curves
        env = envs[0]

        fig, ax = plt.subplots(figsize=(10, 5))
        has_any = False
        for algo in TRAINING_ALGOS:
            if algo in ("Random",):
                continue
            data = curve_data.get((algo, env), [])
            if not data:
                continue
            # Compute per-seed smoothed curve then mean across seeds
            all_curves = []
            for seed, returns in sorted(data):
                if not returns:
                    continue
                # Smooth with rolling window
                window = max(1, len(returns) // 100)
                smoothed = []
                for i in range(len(returns)):
                    s_start = max(0, i - window)
                    chunk = [safe_float(v) for v in returns[s_start:i+1]]
                    chunk_finite = [v for v in chunk if not math.isnan(v)]
                    smoothed.append(sum(chunk_finite) / len(chunk_finite) if chunk_finite else float("nan"))
                all_curves.append(smoothed)

            if not all_curves:
                continue

            # Align to same length
            min_len = min(len(c) for c in all_curves)
            aligned = [c[:min_len] for c in all_curves]
            ep_axis = list(range(1, min_len + 1))

            # Mean + std across seeds
            means_curve = []
            stds_curve = []
            for i in range(min_len):
                vals_i = [c[i] for c in aligned if not math.isnan(c[i])]
                if vals_i:
                    m = sum(vals_i) / len(vals_i)
                    sd = math.sqrt(sum((v - m)**2 for v in vals_i) / len(vals_i)) if len(vals_i) > 1 else 0
                    means_curve.append(m)
                    stds_curve.append(sd)
                else:
                    means_curve.append(float("nan"))
                    stds_curve.append(0.0)

            color = COLORS.get(algo, "#888888")
            ax.plot(ep_axis, means_curve, label=algo, color=color, linewidth=1.2, alpha=0.9)
            means_arr = np.array(means_curve)
            stds_arr = np.array(stds_curve)
            ax.fill_between(ep_axis, means_arr - stds_arr, means_arr + stds_arr,
                            color=color, alpha=0.12)
            has_any = True

        if has_any:
            ax.set_xlabel("Episode", fontsize=10)
            ax.set_ylabel("Training Return (smoothed)", fontsize=10)
            ax.set_title(f"{tier} — Learning Curves on {env} (mean ± std, eligible seeds)", fontsize=11)
            ax.legend(fontsize=7, ncol=3, loc="lower right")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"curves_{tier.replace('-','')}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: curves_{tier.replace('-','')}.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def _analyze_all_main(records=None):
    """Run all analysis outputs with one independent record set.

    Runs every analysis (returns summary, tier summary, oracle comparison,
    MASAC instability, training metrics, learning curves, plots) in a single
    pass. Called by the ``all`` subcommand of :func:`main`.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    curves_dir = os.path.join(OUTPUT_DIR, "learning_curves")
    plot_dir = os.path.join(OUTPUT_DIR, "plots")

    print("=" * 70)
    print("LOADING MERGED DATASET")
    print("=" * 70)

    if records is None:
        # Preserve the legacy entry point without reading directory aliases twice.
        paths = dict.fromkeys(os.path.realpath(p) for p in (MERGED_TR123, MERGED_TR4))
        all_records = [record for path in paths for record in load_directory(path, include_invalid=True)]
    else:
        all_records = records
    print(f"  Total: {len(all_records)} records")

    # Filter to training + oracle only (exclude Constant for most analyses)
    diagnostics = [r for r in all_records if r["algo"] in TRAINING_ALGOS + ORACLE_ALGOS]
    relevant = [r for r in diagnostics if is_successful_result(r["record"])]
    if len(relevant) != len(diagnostics):
        print(f"  Excluded {len(diagnostics) - len(relevant)} invalid/failed records from numerical outputs.")
    print(f"  Training + Oracle records: {len(relevant)}")
    print()

    # ── Returns Summary ───────────────────────────────────────────────────────
    print("[1/6] Computing returns summary...")
    summary = compute_returns_summary(relevant)
    csv_path = os.path.join(OUTPUT_DIR, "returns_summary.csv")
    write_returns_csv(summary, csv_path)

    # ── Tier Summary Table ────────────────────────────────────────────────────
    print("[2/6] Building tier summary table...")
    tier_txt = tier_summary_table(summary)
    tier_path = os.path.join(OUTPUT_DIR, "tier_summary.txt")
    with open(tier_path, "w") as f:
        f.write(tier_txt)
    print(f"  Wrote: {tier_path}")

    # ── Oracle Comparison ─────────────────────────────────────────────────────
    print("[3/6] Building oracle comparison tables...")
    oracle_txt = oracle_comparison_table(relevant)
    oracle_path = os.path.join(OUTPUT_DIR, "oracle_comparison.txt")
    with open(oracle_path, "w") as f:
        f.write(oracle_txt)
    print(f"  Wrote: {oracle_path}")

    # ── MASAC Instability ─────────────────────────────────────────────────────
    print("[4/6] Analyzing MASAC instability...")
    masac_txt = masac_instability_report(diagnostics)
    masac_path = os.path.join(OUTPUT_DIR, "masac_instability.txt")
    with open(masac_path, "w") as f:
        f.write(masac_txt)
    print(f"  Wrote: {masac_path}")

    # ── Training Metrics Summary ──────────────────────────────────────────────
    print("[5/6] Summarizing gradient-level training metrics...")
    tm_rows = training_metrics_summary(relevant)
    tm_path = os.path.join(OUTPUT_DIR, "training_metrics_final.csv")
    if tm_rows:
        with open(tm_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=tm_rows[0].keys())
            w.writeheader()
            w.writerows(tm_rows)
        print(f"  Wrote: {tm_path} ({len(tm_rows)} rows)")

    # ── Learning Curves Export ────────────────────────────────────────────────
    print("[6/6] Exporting learning curves and generating plots...")
    n_curve_files = export_learning_curves(relevant, curves_dir)
    print(f"  Wrote: {n_curve_files} curve CSVs to {curves_dir}/")
    make_plots(summary, relevant, plot_dir)

    # ── Final Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  {csv_path}")
    print(f"  {tier_path}")
    print(f"  {oracle_path}")
    print(f"  {masac_path}")
    print(f"  {tm_path}")
    print(f"  {curves_dir}/  ({n_curve_files} CSVs)")
    print(f"  {plot_dir}/")

    # Print tier summary and oracle comparison to stdout
    print()
    print(tier_txt)
    print()

    # Print abbreviated oracle comparison
    print(oracle_txt)
    print()

    # Print MASAC report
    print(masac_txt)


# =============================================================================
# Reward-type ablation comparison
# =============================================================================

def _load_and_aggregate(input_dir: str, reward_type: str, seeds=None) -> dict:
    """Load results from one reward-configuration directory and aggregate
    per-(algo, env) by taking mean across seeds.

    Returns per-cell summaries including seed and configuration provenance.
    """
    records = load_directory(input_dir, reward_type=reward_type, seeds=seeds)
    return compute_returns_summary(records)


def _cross_arm_context(configuration):
    """Remove only treatment identifiers; retain code, budget and policy design."""
    context = {key: value for key, value in configuration.items()
               if key not in {"reward_type", "actual_reward_type", "campaign_id", "config_hash"}}
    if isinstance(context.get("environment_config"), dict):
        context["environment_config"] = {
            key: value for key, value in context["environment_config"].items()
            if key != "reward_type"
        }
    return context


def _cross_arm_status(arms):
    if any(arm is None for arm in arms):
        return "incomparable: missing arm"
    if any(arm["seeds"] != arms[0]["seeds"] for arm in arms[1:]):
        return "incomparable: different seed folds"
    context = _cross_arm_context(arms[0]["configuration"])
    if any(_cross_arm_context(arm["configuration"]) != context for arm in arms[1:]):
        return "incomparable: different recorded scientific configurations"
    return "matching recorded seeds and context"


def compare_reward_configurations(
    baseline_dir: str,
    private_dir: str,
    cooperative_dir: str,
    output_dir: str,
    seeds=None,
) -> None:
    """Compare per-(algo, env) mean return across the three reward configurations.

    Writes ``reward_ablation_summary.csv`` with one row per (algo, env) and
    columns for each of private, integrated (baseline), and cooperative
    return, observed seeds/configuration, and differences only when all arms
    have identical seeds and matching recorded scientific context. Equality
    of partial legacy metadata is not proof of a controlled historical design.
    """
    baseline = _load_and_aggregate(baseline_dir, "integrated", seeds)
    private = _load_and_aggregate(private_dir, "private", seeds)
    cooperative = _load_and_aggregate(cooperative_dir, "cooperative", seeds)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "reward_ablation_summary.csv")

    all_keys = sorted(set(baseline) | set(private) | set(cooperative))
    rows = []
    for key in all_keys:
        algo, env = key
        b = baseline.get(key, {}).get("mean")
        p = private.get(key, {}).get("mean")
        c = cooperative.get(key, {}).get("mean")
        comparison_status = _cross_arm_status([baseline.get(key), private.get(key), cooperative.get(key)])
        aligned = comparison_status == "matching recorded seeds and context"
        rows.append({
            "algorithm": algo,
            "environment": env,
            "comparison_basis": "historical-roster",
            "tr": ENV_TO_TR.get(env, "?"),
            "comparison_status": comparison_status,
            "seeds_private": json.dumps(private.get(key, {}).get("seeds", [])),
            "seeds_integrated": json.dumps(baseline.get(key, {}).get("seeds", [])),
            "seeds_cooperative": json.dumps(cooperative.get(key, {}).get("seeds", [])),
            "configuration_private": json.dumps(private.get(key, {}).get("configuration", {}), sort_keys=True),
            "configuration_integrated": json.dumps(baseline.get(key, {}).get("configuration", {}), sort_keys=True),
            "configuration_cooperative": json.dumps(cooperative.get(key, {}).get("configuration", {}), sort_keys=True),
            "treatment_sources_private": json.dumps(private.get(key, {}).get("treatment_sources", [])),
            "treatment_sources_integrated": json.dumps(baseline.get(key, {}).get("treatment_sources", [])),
            "treatment_sources_cooperative": json.dumps(cooperative.get(key, {}).get("treatment_sources", [])),
            "mean_return_private": f"{p:.4f}" if p is not None else "",
            "mean_return_integrated": f"{b:.4f}" if b is not None else "",
            "mean_return_cooperative": f"{c:.4f}" if c is not None else "",
            "delta_integrated_minus_private": f"{(b - p):.4f}" if aligned else "",
            "delta_cooperative_minus_integrated": f"{(c - b):.4f}" if aligned else "",
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")


# =============================================================================
# Subcommand layer
# =============================================================================

def _cmd_all(args):
    """Load the unified input once and run each analysis on those records."""
    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir
    _analyze_all_main(_load_input(args, include_invalid=True))
    return 0


def _cmd_returns_summary(args):
    records = _load_input(args)
    records = [r for r in records if r["status"] == "success"]
    relevant = [r for r in records if r["algo"] in TRAINING_ALGOS + ORACLE_ALGOS]
    summary = compute_returns_summary(relevant)
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_returns_csv(summary, out_path)
    return 0


def _cmd_oracle_comparison(args):
    records = _load_input(args)
    records = [r for r in records if r["status"] == "success"]
    relevant = [r for r in records if r["algo"] in TRAINING_ALGOS + ORACLE_ALGOS]
    txt = oracle_comparison_table(relevant)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(txt)
    print(f"Wrote {args.output}")
    if args.print_to_stdout:
        print(txt)
    return 0


def _cmd_tier_summary(args):
    records = _load_input(args)
    records = [r for r in records if r["status"] == "success"]
    relevant = [r for r in records if r["algo"] in TRAINING_ALGOS + ORACLE_ALGOS]
    summary = compute_returns_summary(relevant)
    txt = tier_summary_table(summary)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(txt)
    print(f"Wrote {args.output}")
    if args.print_to_stdout:
        print(txt)
    return 0


def _cmd_masac(args):
    records = _load_input(args, include_invalid=True)
    relevant = [r for r in records if r["algo"] in TRAINING_ALGOS + ORACLE_ALGOS]
    txt = masac_instability_report(relevant)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(txt)
    print(f"Wrote {args.output}")
    return 0


def _cmd_training_metrics(args):
    records = _load_input(args)
    records = [r for r in records if r["status"] == "success"]
    relevant = [r for r in records if r["algo"] in TRAINING_ALGOS + ORACLE_ALGOS]
    rows = training_metrics_summary(relevant)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if rows:
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.output} ({len(rows)} rows)")
    else:
        print("No training metric rows produced.")
    return 0


def _cmd_learning_curves(args):
    records = _load_input(args)
    records = [r for r in records if r["status"] == "success"]
    relevant = [r for r in records if r["algo"] in TRAINING_ALGOS + ORACLE_ALGOS]
    os.makedirs(args.output_dir, exist_ok=True)
    n = export_learning_curves(relevant, args.output_dir)
    print(f"Wrote {n} CSVs to {args.output_dir}/")
    return 0


def _cmd_plots(args):
    records = _load_input(args)
    records = [r for r in records if r["status"] == "success"]
    relevant = [r for r in records if r["algo"] in TRAINING_ALGOS + ORACLE_ALGOS]
    summary = compute_returns_summary(relevant)
    os.makedirs(args.output_dir, exist_ok=True)
    make_plots(summary, relevant, args.output_dir)
    return 0


def _cmd_reward_ablation(args):
    compare_reward_configurations(
        baseline_dir=args.input_baseline,
        private_dir=args.input_private,
        cooperative_dir=args.input_cooperative,
        output_dir=args.output_dir,
        seeds=getattr(args, "seeds", None),
    )
    return 0


def _build_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Analysis pipeline for the Coopetition-Gym v1 benchmark dataset. "
                    "Run one of the subcommands below. See the module docstring "
                    "(experiments/analyze.py) for full details on each analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- all
    sp = sub.add_parser("all", help="Run every analysis in one pass.")
    sp.add_argument("--input-dir", required=True,
                    help="Directory of training result JSON files.")
    sp.add_argument("--output-dir", required=True,
                    help="Output directory for all artifacts.")
    sp.set_defaults(func=_cmd_all)

    # -- returns-summary
    sp = sub.add_parser("returns-summary",
                        help="Cross-seed returns CSV (mean/std/sem per algo-env).")
    sp.add_argument("--input-dir", required=True)
    sp.add_argument("--output", required=True,
                    help="Output CSV file path.")
    sp.set_defaults(func=_cmd_returns_summary)

    # -- oracle-comparison
    sp = sub.add_parser("oracle-comparison",
                        help="Gap-percent tables by TR tier using ENV_ORACLE_REF.")
    sp.add_argument("--input-dir", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--print-to-stdout", action="store_true")
    sp.set_defaults(func=_cmd_oracle_comparison)

    # -- tier-summary
    sp = sub.add_parser("tier-summary",
                        help="Aggregate ranking table per TR tier.")
    sp.add_argument("--input-dir", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--print-to-stdout", action="store_true")
    sp.set_defaults(func=_cmd_tier_summary)

    # -- masac-instability
    sp = sub.add_parser("masac-instability",
                        help="MASAC instability diagnostic report.")
    sp.add_argument("--input-dir", required=True)
    sp.add_argument("--output", required=True)
    sp.set_defaults(func=_cmd_masac)

    # -- training-metrics
    sp = sub.add_parser("training-metrics",
                        help="Final gradient metric values CSV.")
    sp.add_argument("--input-dir", required=True)
    sp.add_argument("--output", required=True)
    sp.set_defaults(func=_cmd_training_metrics)

    # -- learning-curves
    sp = sub.add_parser("learning-curves",
                        help="Export per-(algo, env) training-return CSVs.")
    sp.add_argument("--input-dir", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.set_defaults(func=_cmd_learning_curves)

    # -- plots
    sp = sub.add_parser("plots",
                        help="Generate publication figures (PNG).")
    sp.add_argument("--input-dir", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.set_defaults(func=_cmd_plots)

    # -- reward-ablation
    sp = sub.add_parser("reward-ablation",
                        help="Compare private vs integrated vs cooperative returns.")
    sp.add_argument("--input-baseline", required=True,
                    help="Directory with integrated-reward results.")
    sp.add_argument("--input-private", required=True,
                    help="Directory with private-reward results.")
    sp.add_argument("--input-cooperative", required=True,
                    help="Directory with cooperative-reward results.")
    sp.add_argument("--output-dir", required=True)
    sp.set_defaults(func=_cmd_reward_ablation)

    for command, command_parser in sub.choices.items():
        command_parser.add_argument("--seeds", type=_parse_seeds,
                                    help="Explicit comma-separated evaluation fold; otherwise use all supplied seeds.")
        if command != "reward-ablation":
            command_parser.add_argument("--reward-type", choices=("private", "integrated", "cooperative"),
                                        help="Select this treatment; explicitly supplies missing legacy treatment metadata.")
    return parser


def _parse_seeds(value):
    import argparse
    try:
        seeds = [int(part.strip()) for part in value.split(",")]
        if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError
        return seeds
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be distinct nonnegative integers separated by commas") from exc


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RecordError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    import sys
    sys.exit(main())