# Contributing to Quantum-Chemistry-Eigensolver

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/IsolatedSingularity/Quantum-Chemistry-Eigensolver.git
cd Quantum-Chemistry-Eigensolver

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Linting and Formatting

```bash
ruff check .        # lint
ruff format --check .  # format check
mypy quantum_chemistry/ --ignore-missing-imports
```

Or install pre-commit hooks to run these automatically:

```bash
pip install pre-commit
pre-commit install
```

All tests live in the `tests/` directory and are run automatically by GitHub Actions on every push and pull request.

## Regenerating H₂ Integrals

To regenerate the pre-computed integrals, install the `generate` extra (requires PySCF and rustworkx):

```bash
pip install -e ".[generate]"
python usage/generare_h2_integrals.py
```

## Code Style

- Keep functions and classes well-documented with docstrings.
- Use **camelCase** for all identifiers (variables, functions, parameters).
- Use type hints where practical.

## Project Structure

| Directory              | Purpose                                              |
|------------------------|------------------------------------------------------|
| `quantum_chemistry/`   | Core library: Pauli algebra, mapping, estimation, VQE |
| `quantum_chemistry/molecule/` | Molecular integrals, Hartree-Fock solver       |
| `tests/`               | pytest test suite                                    |
| `examples/`            | Tutorial scripts (mapping, estimation & VQE)         |
| `visualization/`       | Matplotlib visualizations and animations             |
| `h2_data/`             | Pre-computed H₂ spin-orbital integrals               |

## Pull Requests

1. Fork the repo and create a feature branch.
2. Make your changes and add/update tests.
3. Ensure `pytest` passes locally.
4. Open a PR against `main` with a clear description of the change.

## License

By contributing you agree that your contributions will be licensed under the [MIT License](LICENSE).
