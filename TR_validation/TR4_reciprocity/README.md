# Computational Foundations for Strategic Coopetition: Formalizing Sequential Interaction and Reciprocity - Validation Suite

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the validation suite for the technical report:

> **Computational Foundations for Strategic Coopetition: Formalizing Sequential Interaction and Reciprocity**
> Vik Pant, Eric Yu
> Faculty of Information, University of Toronto
> arXiv: (forthcoming)

The validation suite provides complete reproducibility for all experimental and empirical validation results presented in Sections 7-8 of the technical report.

## Key Results Reproduced

| Metric | Value | Threshold |
|--------|-------|-----------|
| **Cooperation Emergence** | 87.0% | > 85% |
| **Defection Punishment** | 98.0% | > 95% |
| **Forgiveness Dynamics** | 84.0% | > 80% |
| **Asymmetric Differentiation** | 93.0% | > 90% |
| **Trust-Reciprocity Interaction** | 91.0% | > 90% |
| **Bounded Responses** | 100.0% | = 100% |
| **Apple iOS Validation Score** | 48.0/55 (87.3%) | > 83% |

Statistical significance: p < 0.001, Cohen's d = 0.68 (medium-to-large effect size)

## Repository Structure

```
.
├── LICENSE                     # MIT License
├── README.md                   # This documentation file
├── TR4_validation_suite.py     # Consolidated validation script
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
pip install -r requirements.txt
```

## Usage

### Run All Validation

```bash
python TR4_validation_suite.py --mode all --granularity standard
```

### Run Only Experimental Validation

```bash
# Standard: 5^6 = 15,625 configurations (~30 min)
python TR4_validation_suite.py --mode experimental --granularity standard

# Quick test: 3^6 = 729 configurations (~2 min)
python TR4_validation_suite.py --mode experimental --granularity coarse
```

### Run Only Empirical Validation

```bash
python TR4_validation_suite.py --mode empirical
```

### Custom Output Directory

```bash
python TR4_validation_suite.py --output ./my_results --seed 123
```

## Granularity Options

| Granularity | Configurations | Approximate Runtime |
|-------------|---------------|---------------------|
| coarse | 729 (3^6) | ~2 minutes |
| standard | 15,625 (5^6) | ~30 minutes |
| fine | 46,656 (6^6) | ~90 minutes |

## Mathematical Framework

The validation suite implements the reciprocity dynamics model from TR-4:

- **Cooperation signal**: s_ij = tanh(kappa * (a_j - baseline))
- **Memory average**: s_bar = (1/k) * sum of recent signals
- **Reciprocity modifier**: phi = clip(rho_0 * D^eta * s_bar, -kappa, kappa)
- **Trust-gated effect**: effective = T_ij * lambda_R * phi_recip
- **Trust dynamics** (from TR-2): 3:1 negativity bias (lambda- / lambda+ = 3)

### Parameter Space

| Parameter | Symbol | Range |
|-----------|--------|-------|
| Base reciprocity | rho_0 | [0.5, 2.0] |
| Dependency elasticity | eta | [0.8, 2.0] |
| Response sensitivity | kappa | [0.5, 2.0] |
| Memory window | k | [1, 20] |
| Reciprocity weight | lambda_R | [0.5, 2.0] |
| Initial trust | T_0 | [0.3, 0.9] |

## Empirical Case Study

**Apple iOS App Store (2008-2024)**: Three-actor ecosystem (Apple, Major Developers, Small Developers) across five phases:

1. **Symbiosis** (2008-2012): Platform launch and mutual cooperation
2. **Maturation** (2013-2017): Stable high cooperation
3. **Tension** (2018-2020): Declining reciprocity, developer grievances
4. **Crisis** (2020-2021): Epic Games lawsuit, reciprocal defection
5. **Adjustment** (2021-2024): Partial restoration of cooperation

Validation uses a 12-indicator x 5-phase scoring matrix (48.0/55 applicable points = 87.3%).

## Output Files

| File | Description |
|------|-------------|
| `comprehensive_parameter_sweep.csv` | Full experimental results |
| `sensitivity_analysis.csv` | Parameter sensitivity matrix |
| `behavioral_targets.json` | Target achievement summary |
| `functional_experiments.json` | Functional experiment results |
| `apple_ios_results.json` | Empirical validation data |
| `enhanced_experimental_validation.png` | 12-panel experimental visualization |
| `apple_ios_validation.png` | 8-panel case study visualization |
| `validation_summary.json` | Final summary |

## Citation

```bibtex
@techreport{pant2025reciprocity,
  title={Computational Foundations for Strategic Coopetition:
         Formalizing Sequential Interaction and Reciprocity},
  author={Pant, Vik and Yu, Eric},
  year={2025},
  institution={University of Toronto}
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
