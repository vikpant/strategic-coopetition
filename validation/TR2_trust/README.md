# Computational Foundations for Strategic Coopetition: Formalizing Trust Dynamics and Trustworthiness - Validation Suite

[![arXiv](https://img.shields.io/badge/arXiv-2510.24909-b31b1b.svg)](https://arxiv.org/abs/2510.24909)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

> **Computational Foundations for Strategic Coopetition: Formalizing Trust Dynamics and Trustworthiness**  
> Vik Pant, Eric Yu  
> Faculty of Information, University of Toronto  
> arXiv:2510.24909

The validation suite provides complete reproducibility for all experimental and empirical validation results presented in Sections 8-9 of the technical report.

1. **Experimental Validation** (Section 8): Comprehensive parameter sweep across 78,125 configurations
2. **Empirical Validation** (Section 9): Renault-Nissan Alliance case study (1999-2025)

---

## Quick Start

### Installation

```bash
# Clone or download this repository
cd code

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Quick verification (coarse granularity, ~4 minutes)
python TR2_validation_suite.py --mode all --granularity coarse

# Full replication (standard granularity, ~2 hours)
python TR2_validation_suite.py --mode all --granularity standard

# Experimental validation only
python TR2_validation_suite.py --mode experimental --granularity standard

# Empirical validation only (Renault-Nissan case)
python TR2_validation_suite.py --mode empirical
```

---

## Expected Results

### Experimental Validation (78,125 Configurations)

| Metric | Expected Value | Paper Reference |
|--------|----------------|-----------------|
| Negativity Ratio Range | [1.0, 9.0] | Section 8.2 |
| Negativity Ratio Median | 3.0 | Table 1 |
| Hysteresis Recovery (35 periods) | Median 1.11 (111%) | Section 8.2.2 |
| Hysteresis Recovery Range | [0.79, 1.17] | Table 1 |
| Cumulative Damage Amplification | Median 1.97 | Section 8.2.3 |
| Dependency Amplification | Mean 1.27× | Section 8.2.4 |

### Empirical Validation (Renault-Nissan Alliance)

| Dimension | Expected Score | Paper Reference |
|-----------|----------------|-----------------|
| Trust State Alignment | 10/15 | Section 9.5.1 |
| Behavioral Prediction | 15/15 | Section 9.5.2 |
| Mechanism Validation | 15/15 | Section 9.5.3 |
| Outcome Correspondence | 9/15 | Section 9.5.4 |
| **Total Score** | **49/60 (81.7%)** | Section 9.5.5 |
| Phase ANOVA F-statistic | 35.05 | Section 9.4 |
| Phase ANOVA p-value | <0.0001 | Section 9.4 |

---

## Repository Structure

```
.
├── LICENSE                       # MIT License
├── TR2_validation_suite.py       # Main validation framework
├── requirements.txt              # Python dependencies
├── README.md                     # This file
```

---

## Detailed Usage

### Command-Line Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--mode` | `experimental`, `empirical`, `all` | `all` | Validation mode |
| `--granularity` | `coarse`, `standard`, `fine`, `ultra` | `standard` | Parameter sweep resolution |
| `--case` | string | `renault_nissan` | Case study name |
| `--output` | path | `enhanced_validation_results` | Output directory |

### Granularity Options

| Level | Configurations | Estimated Runtime | Use Case |
|-------|----------------|-------------------|----------|
| `coarse` | 3⁷ = 2,187 | ~4 minutes | Quick verification |
| `standard` | 5⁷ = 78,125 | ~2 hours | Paper replication |
| `fine` | 6⁷ = 279,936 | ~8 hours | Extended analysis |
| `ultra` | 8⁷ = 2,097,152 | ~48 hours | Comprehensive sweep |

---

## Mathematical Model

The framework implements the trust dynamics model from TR-2025-02:

### Core Equations

**Cooperation Signal (Eq. 6):**
```
s_ij = tanh(κ · (a_j - a_j^baseline))
```

**Trust Update - Building (Eq. 7):**
```
ΔT_ij = λ+ · s_ij · (1 - T_ij) · (1 - R_ij)
```

**Trust Update - Erosion (Eq. 8):**
```
ΔT_ij = λ- · s_ij · T_ij · (1 + ξ · interdependence_ij)
```

**Reputation Damage (Eq. 9):**
```
ΔR_ij = -μ_R · s_ij · (1 - R_ij)  if s_ij < 0
ΔR_ij = -δ_R · R_ij               if s_ij ≥ 0
```

### Default Parameters

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| Trust building rate | λ+ | 0.10 | Rate of trust increase |
| Trust erosion rate | λ- | 0.30 | Rate of trust decrease |
| Negativity ratio | λ-/λ+ | 3.0 | Asymmetry factor |
| Reputation damage | μ_R | 0.60 | Severity of reputation damage |
| Reputation decay | δ_R | 0.03 | Rate of reputation recovery |
| Interdependence amp. | ξ | 0.50 | Amplification factor |
| Reciprocity strength | ρ | 0.20 | Reciprocity weight |
| Signal sensitivity | κ | 1.0 | Cooperation signal scaling |

---

## Output Files

### Experimental Validation Outputs

1. **comprehensive_parameter_sweep.csv**
   - Complete results for all configurations
   - Columns: config_id, all parameters, all metrics

2. **sensitivity_analysis.csv**
   - Parameter-outcome correlation matrix
   - Identifies most influential parameters

3. **pareto_optimal_configs.csv**
   - Pareto-optimal configurations
   - Multi-objective optimization results

4. **enhanced_experimental_validation.png**
   - 12-panel visualization including:
     - Negativity ratio distribution
     - Hysteresis recovery scatter plots
     - 3D parameter space
     - Cumulative damage distribution
     - Parameter correlation heatmap
     - Sensitivity heatmap
     - Pareto frontier projection

### Empirical Validation Outputs

1. **{case}_enhanced_results.json**
   - Complete validation results
   - Trust trajectories
   - Validation scores
   - Statistical tests

2. **{case}_enhanced_validation.png**
   - 8-panel visualization including:
     - Trust evolution (all phases)
     - Reputation damage evolution
     - Trust ceiling mechanism
     - Phase-wise trust changes
     - Trust distribution over time
     - Trust vs reputation scatter

---

## Renault-Nissan Case Study

### Phase Structure

| Phase | Period | Duration | Years | Cooperation Level |
|-------|--------|----------|-------|-------------------|
| 1. Formation | 0-11 | 12 | 1999-2002 | Above baseline (+1.5) |
| 2. Mature Cooperation | 12-51 | 40 | 2002-2018 | High (+2.0) |
| 3. Crisis | 52-55 | 4 | 2018-2019 | Severe violation (-2.0) |
| 4. Recovery | 56-70 | 15 | 2019-2023 | Moderate (+0.5) |
| 5. Current State | 71-79 | 8 | 2023-2025 | Gradual (+1.0) |

### Interdependence Matrix

```
              Renault  Nissan
Renault         0.0     0.65
Nissan         0.75     0.0
```

---

## Verification

To verify installation and basic functionality:

```bash
python -c "
from TR2_validation_suite import TrustParameters, TrustDynamicsModel

params = TrustParameters()
print(f'Negativity ratio: {params.lambda_minus/params.lambda_plus}')
print('Installation verified successfully!')
"
```

Expected output:
```
Negativity ratio: 3.0
Installation verified successfully!
```

---

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Error** (with `ultra` granularity): Reduce granularity or increase available RAM

3. **Plotting Issues**: Ensure matplotlib backend is configured:
   ```python
   import matplotlib
   matplotlib.use('Agg')  # For headless environments
   ```

### System Requirements

- **Python**: 3.9 or higher
- **RAM**: 4 GB minimum (8 GB for `fine`/`ultra` granularity)
- **Disk**: 500 MB for output files
- **Time**: See granularity table above

---

## Research Program Context

This validation suite is part of a coordinated research program on computational approaches to strategic coopetition:

| Technical Report | Topic | arXiv |
|-----------------|-------|-------|
| TR-1 | Interdependence & Complementarity | [2510.18802](https://arxiv.org/abs/2510.18802) |
| **TR-2** (this work) | Trust Dynamics and Trustworthiness | [2510.24909](https://arxiv.org/abs/2510.24909) |
| TR-3 | Complex Actors | (forthcoming) |
| TR-4 | Reciprocity & Sequential Cooperation | (forthcoming) |

## Citation

If you use this code in your research, please cite:

```bibtex
@techreport{pant2025trust,
  title = {Computational Foundations for Strategic Coopetition: 
           Formalizing Trust Dynamics and Trustworthiness},
  author={Pant, Vik and Yu, Eric},
  year={2025},
  institution={University of Toronto},
  number = {TR-2025-02},
  note={arXiv:2510.24909}
}
```

## Related Work

This code accompanies TR-2025-02, which builds on:

- **TR-2025-01**: *Computational Foundations for Strategic Coopetition: Formalizing Interdependence and Complementarity* (Samsung-Sony S-LCD validation, 58/60 score under strict historical alignment scoring)

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
