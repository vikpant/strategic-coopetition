# Computational Foundations for Strategic Coopetition: Formalizing Interdependence and Complementarity - Validation Suite

[![arXiv](https://img.shields.io/badge/arXiv-2510.18802-b31b1b.svg)](https://arxiv.org/abs/2510.18802)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the validation suite for the technical report:

> **Computational Foundations for Strategic Coopetition: Formalizing Interdependence and Complementarity**  
> Vik Pant, Eric Yu  
> Faculty of Information, University of Toronto  
> arXiv:2510.18802

The validation suite provides complete reproducibility for all experimental and empirical validation results presented in Sections 7-8 of the technical report.

## Key Results Reproduced

| Metric | Power (β=0.75, γ=0.5) | Logarithmic (θ=20, γ=0.65) |
|--------|----------------------|---------------------------|
| **Validation Score** | 46/60 | **58/60** |
| **Cooperation Increase** | 166% | **41%** |
| **Historical Alignment** | 0% | **100%** |
| **Monte Carlo Win Rate** | 0% | **100%** |

Statistical significance: p < 0.001, Cohen's d > 9 (very large effect size)

## Repository Structure

```
.
├── LICENSE                   # MIT License
├── README.md                 # This documentation file
├── TR1_validation_suite.py   # Consolidated validation script
├── requirements.txt          # Python dependencies
```

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- SciPy ≥ 1.7

## Installation

```bash
# Clone or download the repository
git clone https://github.com/[username]/strategic-coopetition-validation.git
cd strategic-coopetition-validation

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the complete validation suite:

```bash
python TR1_validation_suite.py
```

This reproduces all results from the technical report, including:
- TR parameter validation (Section 8.3.3)
- Monte Carlo robustness testing (Section 7.6.2)
- Statistical significance tests
- Convergence verification
- Multi-case validation
- Sensitivity analysis

## Usage Examples

### Run All Experiments

```bash
python TR1_validation_suite.py
```

### Run Specific Experiment

```bash
# TR parameter validation only
python TR1_validation_suite.py --experiment tr

# Monte Carlo robustness testing
python TR1_validation_suite.py --experiment monte_carlo

# Statistical significance tests
python TR1_validation_suite.py --experiment statistics
```

### Configure Parameters

```bash
# Run with 1000 Monte Carlo trials
python TR1_validation_suite.py --trials 1000

# Use different random seed
python TR1_validation_suite.py --seed 123

# Save results to JSON file
python TR1_validation_suite.py --output results.json

# Quiet mode (suppress verbose output)
python TR1_validation_suite.py --quiet
```

### Command-Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--experiment` | `-e` | Experiment to run | `all` |
| `--trials` | `-n` | Number of Monte Carlo trials | `500` |
| `--seed` | `-s` | Random seed for reproducibility | `42` |
| `--output` | `-o` | Output JSON file | None |
| `--quiet` | `-q` | Suppress verbose output | False |
| `--version` | `-v` | Show version info | - |

Available experiments: `all`, `tr`, `monte_carlo`, `statistics`, `convergence`, `multi_case`, `sensitivity`

## Mathematical Framework

The validation suite implements the complete mathematical framework from the technical report:

### Core Equations

**Interdependence (Equation 1):**
```
D_ij = Σ_d (w_d · Dep(i,j,d) · crit(i,j,d)) / Σ_d w_d
```

**Value Creation (Equations 5-8):**
- Power function: `f_i(a_i) = a_i^β`
- Logarithmic function: `f_i(a_i) = θ · ln(1 + a_i)`
- Synergy: `g(a) = (∏ a_i)^(1/N)`
- Total value: `V(a|γ) = Σf_i(a_i) + γ·g(a)`

**Integrated Utility (Equation 13):**
```
U_i(a) = π_i(a) + Σ_{j≠i} D_ij · π_j(a)
```

**Coopetitive Equilibrium (Definition 3):**
```
a_i* ∈ argmax_{a_i} U_i(a_i, a_{-i}*)
```

### S-LCD Case Study Parameters (Section 8.2)

| Parameter | Value | Source |
|-----------|-------|--------|
| D_Sony,Samsung | 0.86 | i* dependency analysis |
| D_Samsung,Sony | 0.64 | i* dependency analysis |
| α_Samsung | 0.55 | JV ownership structure |
| α_Sony | 0.45 | JV ownership structure |
| Endowments | [100, 100] | Normalized scale |

## Validation Scoring Framework

The 60-point validation score (Section 8.3) comprises five categories:

| Category | Points | Criteria |
|----------|--------|----------|
| **Convergence & Stability** | 10 | Numerical convergence, bounded actions, finite values |
| **Cooperation Dynamics** | 15 | Cooperation increase within historical bounds (15-50%) |
| **Value Creation** | 15 | Positive value creation, realistic magnitudes |
| **Value Distribution** | 10 | Pareto improvement, realistic actor shares |
| **Strategic Realism** | 10 | Convergence, asymmetry within bounds |

## Expected Output

Running the complete validation suite produces:

```
======================================================================
COMPUTATIONAL FOUNDATIONS FOR STRATEGIC COOPETITION
Comprehensive Validation Suite
======================================================================
arXiv ID: 2510.18802
Authors: Vik Pant, Eric Yu
Version: 1.0.0
...

TR_1 VALIDATION RESULTS:

1. VALIDATION SCORES (TR Claims: Power=46/60, Log=58/60):
   - Power (β=0.75, γ=0.5): 46/60 ✓
   - Logarithmic (θ=20, γ=0.65): 58/60 ✓
   - Difference: 12 criteria ✓

2. COOPERATION INCREASE (TR Claims: Power=166%, Log=41%):
   - Power: 166.1% ✓
   - Logarithmic: 41.5% ✓

3. MONTE CARLO ROBUSTNESS (TR Claim: Log wins 100%):
   - Log win rate: 100% ✓
   - Power historical alignment: 0%
   - Log historical alignment: 100%

4. STATISTICAL SIGNIFICANCE (TR Claims: p<0.001, d>9):
   - p-value: 0.00e+00 ✓
   - Cohen's d: 13.99 ✓
   - Log significantly better: True ✓

CONCLUSION: All TR_1 validation claims VERIFIED.
```

## Research Program Context

This validation suite is part of a coordinated research program on computational approaches to strategic coopetition:

| Technical Report | Topic | arXiv |
|-----------------|-------|-------|
| **TR-1** (this work) | Interdependence & Complementarity | [2510.18802](https://arxiv.org/abs/2510.18802) |
| TR-2 | Trust Dynamics and Trustworthiness | [2510.24909](https://arxiv.org/abs/2510.24909) |
| TR-3 | Complex Actors | (forthcoming) |
| TR-4 | Reciprocity & Sequential Cooperation | (forthcoming) |

## Citation

If you use this validation suite in your research, please cite:

```bibtex
@techreport{pant2025foundations,
  title={Computational Foundations for Strategic Coopetition: 
         Formalizing Interdependence and Complementarity},
  author={Pant, Vik and Yu, Eric},
  year={2025},
  institution={University of Toronto},
  number = {TR-2025-01},
  note={arXiv:2510.18802}
}
```

## Authors

- **Vik Pant** - [vik.pant@mail.utoronto.ca](mailto:vik.pant@mail.utoronto.ca)
- **Eric Yu** - [eric.yu@utoronto.ca](mailto:eric.yu@utoronto.ca)

Faculty of Information  
University of Toronto  
140 St George St, Toronto, ON M5S 3G6, Canada

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work extends research from Vik Pant's doctoral thesis on strategic coopetition, supervised by Professor Eric Yu at the University of Toronto. The computational framework builds on the conceptual modeling foundations established in the *i** framework.

---

**Note:** This validation suite is provided for research reproducibility. The code implements the mathematical framework exactly as specified in the technical report to enable independent verification of all claimed results.
