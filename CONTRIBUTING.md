# Contributing

Thank you for your interest in contributing to Coopetition-Gym. This
document outlines how to propose changes, file issues, and submit pull
requests.

## Code of conduct

By participating in this project you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting issues

Please use GitHub Issues to report bugs and request features. When
filing a bug report, include:

- A minimal reproducible example (Python snippet that triggers the
  problem).
- The output of `python -c "import coopetition_gym; print(coopetition_gym.__version__)"`.
- Your operating system and Python version.

## Proposing changes

For non-trivial changes, open a GitHub Issue first to discuss the
proposed direction before investing time in a pull request.

## Pull request workflow

1. Fork the repository and create a feature branch from `master`.
2. Make your changes with clear, focused commits.
3. Add or update tests under `coopetition_gym/coopetition_gym/tests/`.
4. Ensure the full test suite passes locally with `pytest`.
5. Open a pull request against `master` with a description of the
   change and any relevant issue numbers.

## Style

- Python code follows PEP 8 with reasonable line lengths.
- Docstrings on public functions and classes; mathematical notation
  permitted where it clarifies.
- Markdown content follows the authorial style guide at
  `coopetition_gym/docs/authorial_style_analysis.md` (no em dashes in
  prose, consistent terminology, no false coinage of named concepts).

## License

By contributing, you agree that your contributions will be licensed
under the same MIT License that covers this project.
