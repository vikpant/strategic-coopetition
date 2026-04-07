# Computational Foundations for Strategic Coopetition: Formalizing Collective Action and Loyalty - Validation Suite

[![arXiv](https://img.shields.io/badge/arXiv-2601.16237-b31b1b.svg)](https://arxiv.org/abs/2601.16237)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the validation suite for the technical report:

> **Computational Foundations for Strategic Coopetition: Formalizing Collective Action and Loyalty**
> Vik Pant, Eric Yu
> Faculty of Information, University of Toronto
> arXiv: [2601.16237](https://arxiv.org/abs/2601.16237)

The validation suite provides complete reproducibility for all experimental and empirical validation results presented in Sections 6-7 of the technical report.

## Key Results Reproduced

| Metric | Value | Target |
|--------|-------|--------|
| **Free-riding Baseline Accuracy** | 99.7% | < 5% deviation |
| **Loyalty Effect** | 100% | Monotonic increase |
| **Effort Differentiation** | 4.12x (median) | > 2.0x |
| **Mechanism Synergy** | 98.4% | > 1.1 ratio |
| **Apache Validation Score** | 52/60 (86.7%) | Pass |

Statistical significance: p < 0.001, Cohen's d = 8.73 (very large effect size)

## Repository Structure

```
.
├── LICENSE                     # MIT License
├── README.md                   # This documentation file
├── TR3_validation_suite.py     # Consolidated validation script
├── requirements.txt            # Python dependencies
```

## Requirements

- Python >= 3.8
- NumPy >= 1.21
- Pandas >= 1.3
- Matplotlib >= 3.4
- Seaborn >= 0.11
- SciPy >= 1.7

## Installation

```bash
# Clone or download the repository
git clone https://github.com/[username]/strategic-coopetition-validation.git
cd strategic-coopetition-validation/TR3_loyalty

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the complete validation suite:

```bash
python TR3_validation_suite.py
```

This reproduces all results from the technical report, including:
- Comprehensive 7-parameter sweep (15,625 configurations)
- Behavioral target validation (Section 6)
- Monte Carlo robustness testing (2,000 trials)
- Statistical significance tests
- Apache HTTP Server case study (Section 7)

## Usage Examples

### Run All Validation

```bash
python TR3_validation_suite.py --mode all --granularity standard
```

### Run Specific Validation

```bash
# Experimental validation only
python TR3_validation_suite.py --mode experimental

# Empirical validation (Apache case) only
python TR3_validation_suite.py --mode empirical
```

### Configure Parameters

```bash
# Quick test with coarse granularity (~2 minutes)
python TR3_validation_suite.py --granularity coarse

# Fine-grained sweep (~75 minutes)
python TR3_validation_suite.py --granularity fine

# Use different random seed
python TR3_validation_suite.py --seed 123

# Custom output directory
python TR3_validation_suite.py --output ./results
```

### Command-Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--mode` | `-m` | Validation mode | `all` |
| `--granularity` | `-g` | Parameter sweep granularity | `standard` |
| `--seed` | `-s` | Random seed for reproducibility | `42` |
| `--output` | `-o` | Output directory | `./TR3_validation_output` |
| `--quiet` | `-q` | Suppress verbose output | False |
| `--version` | `-v` | Show version info | - |

**Granularity Options:**
- `coarse`: 1,215 configurations (~2 minutes)
- `standard`: 15,625 configurations (~30 minutes)
- `fine`: 38,880 configurations (~75 minutes)

## Mathematical Framework

The validation suite implements the complete mathematical framework from the technical report:

### Core Equations

**Team Production Function (Equation 1):**
```
Q(a) = omega * (sum_i a_i)^beta
```

**Base Team Payoff (Equation 2):**
```
pi_i^team = (1/n) * Q(a) - c * a_i
```

**Loyalty Modifier (Equation 3):**
```
L_i = theta_i * [phi_B * pi_bar_{-i} + phi_C * c * a_i]
```

**Free-riding Equilibrium (Proposition 1):**
```
a* = (omega * beta / (n * c))^(1/(1-beta))
```

**Team Cohesion (Equation 5):**
```
C = sum(D_{T,i} * theta_i) / sum(D_{T,i})
```

### Parameter Space (Section 6.1)

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Productivity | omega | [10, 50] | Team capability factor |
| Returns to Scale | beta | [0.5, 0.9] | Diminishing returns exponent |
| Effort Cost | c | [0.5, 2.0] | Individual cost coefficient |
| Team Size | n | {3, 4, 5, 6, 8} | Number of members |
| Loyalty | theta | [0, 1] | Individual loyalty level |
| Loyalty Benefit | phi_B | [0.4, 1.0] | Benefit mechanism strength |
| Cost Tolerance | phi_C | [0.1, 0.5] | Cost tolerance strength |

### Apache HTTP Server Case Study (Section 7)

| Phase | Period | Core Team | Loyalty Multiplier | Expected Effort |
|-------|--------|-----------|-------------------|-----------------|
| Formation | 1995-1997 | 8 | 1.0 | 6.8 |
| Growth | 1998-2003 | 25 | 0.85 | 5.2 |
| Maturation | 2004-2015 | 40 | 0.70 | 4.1 |
| Evolution | 2016-2023 | 35 | 0.60 | 3.8 |

## Behavioral Targets

The validation suite evaluates six behavioral targets (Section 6.3):

| Target | Criterion | Achievement |
|--------|-----------|-------------|
| Free-riding Baseline | < 5% deviation from analytical | 99.7% |
| Loyalty Effect | Monotonic increase with theta | 100% |
| Effort Differentiation | Ratio (theta=0.9)/(theta=0.1) > 2.0 | 100% |
| Team Size Effect | da*/dn < 0 at low theta | 100% |
| Mechanism Synergy | Combined effect > sum of individual | 98.4% |
| Bounded Outcomes | a* in [0, a_max] | 100% |

## Output Files

Running the validation suite generates:

| File | Description |
|------|-------------|
| `comprehensive_parameter_sweep.csv` | Full experimental results |
| `sensitivity_analysis.csv` | Parameter sensitivity matrix |
| `behavioral_targets.json` | Target achievement summary |
| `enhanced_experimental_validation.png` | 12-panel visualization |
| `apache_enhanced_results.json` | Empirical validation data |
| `apache_enhanced_validation.png` | 8-panel case visualization |
| `validation_summary.json` | Final summary |

## Expected Output

Running the complete validation suite produces:

```
======================================================================
COMPUTATIONAL FOUNDATIONS FOR STRATEGIC COOPETITION
Technical Report 3: Collective Action and Loyalty
Comprehensive Validation Suite
======================================================================
Version: 1.0.0
Authors: Vik Pant, Eric Yu
...

TR-3 VALIDATION RESULTS:

1. EXPERIMENTAL VALIDATION (15,625 configurations):
   - Free-riding baseline: 99.7% < 5% error
   - Loyalty effect: 100% monotonic
   - Effort differentiation: median 4.12x
   - Mechanism synergy: 98.4% > 1.1
   - Bounded outcomes: 100%

2. STATISTICAL SIGNIFICANCE:
   - Paired t-test: p < 0.001
   - Cohen's d: 8.73 (very large)
   - Bootstrap 95% CI: [3.89, 4.35]

3. MONTE CARLO ROBUSTNESS (2000 trials, +/-15% noise):
   - Loyalty monotonicity: 99.8%
   - Effort diff > 2.0: 100%
   - Mean differentiation: 4.15 +/- 0.82

4. EMPIRICAL VALIDATION (Apache HTTP Server):
   - Validation score: 52/60 (86.7%)
   - All phases validated

CONCLUSION: All TR-3 validation claims VERIFIED.
```

## Research Program Context

This validation suite is part of a coordinated research program on computational approaches to strategic coopetition:

| Technical Report | Topic | arXiv |
|-----------------|-------|-------|
| TR-1 | Interdependence & Complementarity | [2510.18802](https://arxiv.org/abs/2510.18802) |
| TR-2 | Trust and Reputation Dynamics | [2510.24909](https://arxiv.org/abs/2510.24909) |
| **TR-3** (this work) | Collective Action & Loyalty | [2601.16237](https://arxiv.org/abs/2601.16237) |
| TR-4 | Sequential Interaction & Reciprocity | [2604.01240](https://arxiv.org/abs/2604.01240) |

## Citation

If you use this validation suite in your research, please cite:

```bibtex
@techreport{pant2025collective,
  title={Computational Foundations for Strategic Coopetition:
         Formalizing Collective Action and Loyalty},
  author={Pant, Vik and Yu, Eric},
  year={2025},
  institution={University of Toronto},
  number = {TR-2025-03},
  note={arXiv:2601.16237}
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

This work extends research from Vik Pant's doctoral thesis on strategic coopetition, supervised by Professor Eric Yu at the University of Toronto. The collective action and loyalty framework builds on the foundational models established in TR-1 (Interdependence) and TR-2 (Trust Dynamics).

---

**Note:** This validation suite is provided for research reproducibility. The code implements the mathematical framework exactly as specified in the technical report to enable independent verification of all claimed results.
