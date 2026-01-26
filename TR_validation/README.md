# Computational Foundations for Strategic Coopetition - Validation Suites

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This directory contains validation suites for the **Computational Foundations for Strategic Coopetition** research program. The program develops computational game-theoretic frameworks for analyzing mixed-motive strategic interactions where actors simultaneously cooperate and compete.

> **Authors:** Vik Pant, Eric Yu
> Faculty of Information, University of Toronto

## Research Program

The research program comprises four technical reports, each addressing a fundamental dimension of coopetitive dynamics:

| Technical Report | Topic | arXiv | Validation Suite |
|-----------------|-------|-------|------------------|
| TR-1 | Interdependence & Complementarity | [2510.18802](https://arxiv.org/abs/2510.18802) | [TR1_foundations/](TR1_foundations/) |
| TR-2 | Trust Dynamics & Trustworthiness | [2510.24909](https://arxiv.org/abs/2510.24909) | [TR2_trust/](TR2_trust/) |
| TR-3 | Collective Action & Loyalty | [2601.16237](https://arxiv.org/abs/2601.16237) | [TR3_loyalty/](TR3_loyalty/) |
| TR-4 | Sequential Interaction & Reciprocity | (forthcoming) | [TR4_reciprocity/](TR4_reciprocity/) |

## Validation Methodology

Each validation suite implements a dual-track validation strategy:

### Experimental Validation
- Comprehensive parameter space exploration (thousands of configurations)
- Behavioral target validation against theoretical predictions
- Statistical significance testing (t-tests, effect sizes, bootstrap confidence intervals)
- Monte Carlo robustness testing with parameter noise

### Empirical Validation
- Real-world case study analysis
- Structured scoring methodology (60-point validation framework)
- Phase-wise temporal analysis
- Cross-validation against documented organizational outcomes

## Directory Structure

```
TR_validation/
├── README.md                    # This file
├── TR1_foundations/             # Interdependence & Complementarity validation
│   ├── README.md
│   ├── TR1_validation_suite.py
│   └── requirements.txt
├── TR2_trust/                   # Trust Dynamics validation
│   ├── README.md
│   ├── TR2_validation_suite.py
│   └── requirements.txt
├── TR3_loyalty/                 # Collective Action & Loyalty validation
│   ├── README.md
│   ├── TR3_validation_suite.py
│   └── requirements.txt
└── TR4_reciprocity/             # Reciprocity validation (forthcoming)
    └── LICENSE
```

## Quick Start

### Run Individual Validation Suites

```bash
# TR-1: Interdependence & Complementarity
cd TR1_foundations
pip install -r requirements.txt
python TR1_validation_suite.py

# TR-2: Trust Dynamics
cd TR2_trust
pip install -r requirements.txt
python TR2_validation_suite.py

# TR-3: Collective Action & Loyalty
cd TR3_loyalty
pip install -r requirements.txt
python TR3_validation_suite.py
```

### Requirements

All validation suites require:
- Python >= 3.8
- NumPy >= 1.21
- Pandas >= 1.3
- Matplotlib >= 3.4
- Seaborn >= 0.11
- SciPy >= 1.7

## Key Validation Results

### TR-1: Interdependence & Complementarity
- **22,000+ configurations** tested across 7-parameter space
- **100%** complementarity effect validated
- **Renault-Nissan case study:** 54/60 points (90%)

### TR-2: Trust Dynamics
- **78,125 configurations** tested (full $5^7$ factorial)
- **Negativity bias:** 3:1 ratio robustly emerges
- **SLCD case study:** 49/60 points (81.7%)

### TR-3: Collective Action & Loyalty
- **15,625 configurations** tested
- **Free-riding baseline:** 99.7% accuracy
- **Apache HTTP Server case study:** 52/60 points (86.7%)

### TR-4: Sequential Interaction & Reciprocity
- Validation suite under development
- Will include Apple iOS ecosystem case study

## Mathematical Framework Integration

The complete coopetitive utility function synthesizes all four dimensions:

$$U_i = \underbrace{\pi_i^{\text{base}}}_{\text{Self-interest}} + \underbrace{\sum_{j \neq i} D_{ij} \pi_j}_{\text{TR-1: Interdependence}} + \underbrace{T_{ij}^t \cdot \text{[trust terms]}}_{\text{TR-2: Trust}} + \underbrace{\theta_i \cdot \text{[loyalty terms]}}_{\text{TR-3: Loyalty}} + \underbrace{\rho_{ij} \cdot \text{[reciprocity terms]}}_{\text{TR-4: Reciprocity}}$$

## Citation

If you use these validation suites in your research, please cite the relevant technical reports:

```bibtex
@techreport{pant2025foundations,
  title={Computational Foundations for Strategic Coopetition:
         Formalizing Interdependence and Complementarity},
  author={Pant, Vik and Yu, Eric},
  year={2025},
  institution={University of Toronto},
  note={arXiv:2510.18802}
}

@techreport{pant2025trust,
  title={Computational Foundations for Strategic Coopetition:
         Formalizing Trust and Reputation Dynamics},
  author={Pant, Vik and Yu, Eric},
  year={2025},
  institution={University of Toronto},
  note={arXiv:2510.24909}
}

@techreport{pant2025collective,
  title={Computational Foundations for Strategic Coopetition:
         Formalizing Collective Action and Loyalty},
  author={Pant, Vik and Yu, Eric},
  year={2025},
  institution={University of Toronto},
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

This project is licensed under the MIT License - see individual validation suite directories for details.

## Acknowledgments

This work extends research from Vik Pant's doctoral thesis on strategic coopetition, supervised by Professor Eric Yu at the University of Toronto. The computational frameworks bridge conceptual modeling (*i** Strategic Dependency models) with game-theoretic analysis to enable quantitative study of mixed-motive strategic interactions.

---

**Note:** These validation suites are provided for research reproducibility. Each suite implements the mathematical framework exactly as specified in the corresponding technical report to enable independent verification of all claimed results.
