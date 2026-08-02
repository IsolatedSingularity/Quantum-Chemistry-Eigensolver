# Open Problems: Quantum-Chemistry-Eigensolver

This document catalogs open problems, unresolved challenges, and improvement areas for the **Quantum-Chemistry-Eigensolver** VQE and molecular Hamiltonian simulation codebase (`vqe.py`, `mapping.py`, `estimation.py`).

---

## 1. Algorithmic & Implementation Problems

- **Particle-Number-Preserving VQE Ansatz (`Q1`)**
  - **Problem**: Current `h2_ansatz_circuit` in `vqe.py` uses unrestricted Ry/CNOT gates that violate particle number and spin conservation, allowing variational descent into unphysical Fock space sectors.
  - **Context**: Requires implementing a number-preserving ansatz (e.g., Givens rotations, EfficientSU2 with symmetry constraints, or Unitary Coupled Cluster Singles and Doubles / UCCSD).
- **Pauli-Wise Commuting (QWC) Measurement Grouping (`Q7`)**
  - **Problem**: Integrating Qubit-Wise Commuting (QWC) or Tensor Product Basis (TPB) Pauli grouping in `estimation.py` to reduce circuit execution count from 15 independent circuits to 3–5 grouped circuits for $H_2$.
- **Hamiltonian Mapping Loop Complexity (`Q8`)**
  - **Problem**: Optimizing `build_two_body_qubit_hamiltonian` in `mapping.py`, which currently relies on $O(n^4)$ nested Python loops and fails to scale efficiently for active spaces larger than 4 qubits.
- **Authentic VQE Visualization Data Pipelines (`Q3, Q12`)**
  - **Problem**: Replacing hardcoded toy energy formulas in `visualization/vqe_energy_curves.py` and `visualization/masterVisualization.py` with actual VQE or exact diagonalization data arrays.
- **Multi-Orbital Molecule Support (`Q2, Q11`)**
  - **Problem**: Extending integral generation and active-space reduction beyond $H_2$ to $LiH$ and $H_2O$ by properly integrating PySCF dependencies.

---

## 2. Bugs & Unresolved Issues

- **Classical Register Accumulation in `measure_all` (`Q14`)**
  - **Problem**: Calling `measure_all(add_bits=True)` in `estimation.py` repeatedly appends new classical registers, causing potential execution failures on real Qiskit hardware backends.
- **Documentation and Version Drift (`Q2, Q13, D9`)**
  - **Problem**: `README.md` claims UCCSD and larger molecule support are complete (`[x]`); `__init__.py` reports version `0.0.1` while `pyproject.toml` reports `0.1.0`; and `pauli.py::simplify` docstrings misreport the default threshold (`1e-0` vs `1e-9`).
- **Unpinned CI and Package Dependencies (`D1, D3`)**
  - **Problem**: All dependencies use floating `>=` version ranges without a lockfile, and GitHub Action workflows reference mutable `@v4`/`@v5` tags rather than immutable commit SHAs.

---

## 3. Theoretical & Scientific Problems

- **Variational Fock Space Symmetry Preservation**
  - **Problem**: Guaranteeing exact conservation of total electron number $\hat{N}$ and spin projection $\hat{S}_z$ across parameterized quantum circuits without excessive gate depth or CNOT overhead.
- **Qubit Mapping Tradeoffs (Jordan-Wigner vs. Bravyi-Kitaev)**
  - **Problem**: Evaluating Pauli operator weight and entanglement scaling differences between Jordan-Wigner and Bravyi-Kitaev transformations for multi-orbital molecular Hamiltonians.

---

## 4. Code Maintenance & Refactoring Opportunities

- **Binary Data Tracking (`D11`)**
  - **Opportunity**: 34 `.npz` binary files in `h2_data/` are tracked directly in Git. Migrating them to Git LFS or an automated download script will optimize repository clone performance.
- **Type Annotation Completeness (`Q9`)**
  - **Opportunity**: Add explicit return type annotations across `molecule/h2_molecule.py` and resolve typing warnings in `molecule/hartree_fock.py`.
