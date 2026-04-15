# Quantum-Chemistry-Eigensolver
*69 tests, 4 qubits, 34 bond distances: a ground-up VQE for H₂*

<p align="center">
  <a href="https://github.com/IsolatedSingularity/Quantum-Chemistry-Eigensolver/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/IsolatedSingularity/Quantum-Chemistry-Eigensolver/ci.yml?branch=main&label=CI&logo=github" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/install-pip%20install%20--e%20.%5Bdev%5D-brightgreen" alt="pip install"></a>
</p>



![VQE Energy Optimization Curves](https://github.com/IsolatedSingularity/quantum-chemistry-eigensolver/blob/main/visualization/vqe_energy_curves.png?raw=true)

## Objective

This repository implements a from-scratch **Variational Quantum Eigensolver (VQE)** for the H₂ molecule on Qiskit 2.0. Every component is built without Qiskit Nature: Pauli algebra engine, Jordan-Wigner mapper, Hartree-Fock solver, QWC measurement grouping, and parameter-shift gradients.

The VQE is a hybrid quantum-classical algorithm that approximates ground state energies. For molecular systems, this energy is determined by the electronic Hamiltonian:

$\hat{H} = \sum_{p,q} h_{pq} a_p^\dagger a_q + \frac{1}{2} \sum_{p,q,r,s} h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s$

where $a_p^\dagger$ and $a_q$ are fermionic creation and annihilation operators, while $h_{pq}$ and $h_{pqrs}$ represent one- and two-electron integrals.

**Goal:** Compute the H₂ dissociation curve at 34 bond distances using a from-scratch VQE pipeline, and validate against exact diagonalization.

<p align="center">
  <img src="./visualization/apple.gif?raw=true" alt="apple" width="52" height="50" />
</p>

## Theoretical Background

Quantum chemistry calculations begin with mapping the molecular Hamiltonian to a qubit representation. The most common approach uses the Jordan-Wigner transformation, which converts fermionic operators to Pauli operators:

$a_j^{\dagger} \rightarrow \tfrac12\bigl(X_j - iY_j\bigr) \prod_{k<j} Z_k, \quad a_j \rightarrow \tfrac12\bigl(X_j + iY_j\bigr) \prod_{k<j} Z_k$

For the H₂ molecule with a minimal basis, we need 4 qubits to represent the system's 4 spin orbitals. After mapping and applying symmetries, the qubit Hamiltonian can be expressed as a sum of tensor products of Pauli operators.

The VQE algorithm then works by:
1. Preparing a parameterized quantum state (ansatz) on a quantum computer
2. Measuring the expectation value of the Hamiltonian
3. Using a classical optimizer to update the parameters to minimize energy
4. Repeating until convergence

For H₂, a simple ansatz involves rotations and entangling gates to create superpositions of computational basis states that represent different electronic configurations.

## Code Functionality

### 1. Initialize Molecule and Parameters
The simulation loads pre-computed spin orbital integrals for H₂ at 34 bond distances. Each file contains one-electron ($h_{pq}$) and two-electron ($h_{pqrs}$) integrals plus nuclear repulsion energy.

```python
def load_h2_spin_orbital_integrals(data_path):
    """Load H2 spin orbital integral data from pre-computed files."""
    distances, molecule_data = [], []
    for file in sorted(os.listdir(data_path)):
        if file.startswith('h2_mo_integrals_d_') and file.endswith('.npz'):
            distance = float(file.split('_d_')[1].split('.npz')[0]) / 1000.0
            data = np.load(os.path.join(data_path, file))
            distances.append(distance)
            molecule_data.append((data['h1'], data['h2'], data['nuclear_repulsion']))
    return np.array(distances), molecule_data
```

### 2. Hamiltonian Construction and Mapping
The Jordan-Wigner transformation converts each fermionic operator $a_p^\dagger$ into Pauli X, Y operators at position $p$ with a Z-string on preceding qubits. One-body and two-body terms are transformed and combined into a single `Operator` representing the qubit Hamiltonian.

```python
def build_qubit_hamiltonian(one_body, two_body, creation_ops, annihilation_ops):
    """Build qubit Hamiltonian from fermionic integrals using JW mapping."""
    h1 = build_one_body_qubit_hamiltonian(one_body, creation_ops, annihilation_ops)
    h2 = build_two_body_qubit_hamiltonian(two_body, creation_ops, annihilation_ops)
    return (h1 + h2).combine().apply_threshold().sort()
```

### 3. VQE Implementation
A parameterized circuit (ansatz) prepares trial wavefunctions, and a classical optimizer adjusts parameters to minimize energy. For H₂, a particle-number-preserving ansatz starts from the Hartree-Fock state |0101⟩ and applies a CNOT staircase, reducing the double excitation to a single Ry rotation. The resulting state $\cos(\theta/2)|0101\rangle + \sin(\theta/2)|1010\rangle$ always contains exactly 2 electrons.

The cost function groups Pauli terms into qubitwise-commuting (QWC) sets. Analytic gradients are computed via the parameter-shift rule.

```python
def h2_ansatz_circuit():
    """Build a particle-number-preserving H2 ansatz with 1 parameter."""
    varform = QuantumCircuit(4)
    theta = Parameter('theta')
    varform.x([1, 3])           # Prepare |0101⟩
    varform.cx(1, 0)            # CNOT staircase
    varform.cx(2, 1)
    varform.cx(3, 2)
    varform.ry(theta, 3)        # Parametric rotation
    varform.cx(3, 2)            # Reverse staircase
    varform.cx(2, 1)
    varform.cx(1, 0)
    return varform
```

### 4. Visualizing H₂ Dissociation
Animates H₂ bond stretching with the corresponding potential energy curve, from compression through equilibrium (~0.74 Å) to dissociation.

```python
def animate_h2_dissociation(save_path):
    """Create animation of H2 molecule separation with energy curve."""
    distances, e_elec, e_nuc, e_total = load_and_process_data()
    fig = plt.figure(figsize=(14, 7))
    # Dual panel: molecule geometry + energy curve
    anim = animation.FuncAnimation(fig, animate, frames=len(distances)*2-2)
    anim.save(save_path, writer='pillow', fps=10)
```

### 5. Molecular Orbital Visualization
Visualizes how atomic 1s orbitals combine into bonding and antibonding molecular orbitals for H₂.

```python
def create_molecular_orbital_visualization(save_path):
    """Create static visualization of H2 molecular orbitals."""
    h1_1s = h1s(X, Y, h1_pos)  # Atomic orbital on H1
    h2_1s = h1s(X, Y, h2_pos)  # Atomic orbital on H2
    bonding_mo = h1_1s + h2_1s      # Constructive interference
    antibonding_mo = h1_1s - h2_1s  # Destructive interference
```

### 6. VQE Energy Curves Visualization
Computes the VQE energy landscape from the full Jordan-Wigner Hamiltonian across 34 bond distances, validated against exact diagonalization.

```python
def create_vqe_energy_curves_visualization(save_path):
    """Build VQE energy curves from real Hamiltonian data."""
    distances, molecule_data = load_h2_spin_orbital_integrals(data_path)
    ansatz = h2_ansatz_circuit()
    # Left: energy vs theta for 8 bond distances
    # Right: exact dissociation vs VQE minimum
```

## Results

Ground state energy at equilibrium: **-1.137 Ha** (within $10^{-6}$ Ha of exact diagonalization). 70 tests, 87% coverage on core modules.

1. **H₂ Molecular Orbitals**:

![H2 Molecular Orbitals](https://github.com/IsolatedSingularity/quantum-chemistry-eigensolver/blob/main/visualization/h2_molecular_orbitals.png?raw=true)

2. **H₂ Dissociation Curve**:

![H2 Dissociation Animation](https://github.com/IsolatedSingularity/quantum-chemistry-eigensolver/blob/main/visualization/h2_dissociation.gif?raw=true)

3. **VQE Energy Optimization Curves**:

![VQE Energy Optimization Curves](https://github.com/IsolatedSingularity/quantum-chemistry-eigensolver/blob/main/visualization/vqe_energy_curves.png?raw=true)

## Installation

```bash
# Clone and install
git clone https://github.com/IsolatedSingularity/Quantum-Chemistry-Eigensolver.git
cd Quantum-Chemistry-Eigensolver
pip install -e ".[dev]"

# Run the test suite
pytest
```

### CLI Entry Points

| Command | Description |
|---|---|
| `qce-dissociation` | Run VQE across all H₂ bond distances and plot the dissociation curve |
| `qce-visualize` | Generate molecular orbital and VQE energy visualizations |

```bash
# Example: compute dissociation curve and save the plot
qce-dissociation --shots 10000 -o dissociation.png

# Generate all visualizations
qce-visualize --which all
```

## Caveats

- **Ansatz Limitations**: The simple circuits used in this implementation may not capture all the relevant physics for larger molecules. More sophisticated ansatz designs are needed for scaling beyond H₂.

- **Classical Simulation Constraints**: While this implementation simulates the quantum algorithm classically, actual quantum devices would face issues like noise, decoherence, and measurement errors.

## Next Steps

- [ ] Implement more sophisticated ansatz circuits, such as the Unitary Coupled Cluster (UCC) ansatz for better accuracy.
- [ ] Extend the implementation to handle larger molecules like LiH, BeH₂, or H₂O.
- [ ] Incorporate noise models to simulate realistic quantum hardware performance.
- [ ] Implement quantum subspace expansion techniques to improve accuracy of excited state calculations.
- [ ] Add support for calculating molecular properties beyond the ground state energy (dipole moments, forces, etc.).

> [!TIP]
> Step-by-step tutorial scripts live in `examples/`; run `tutorial_1_mapping.py` and `tutorial_2_estimation.py` to walk through the full pipeline.

> [!NOTE]
> This project targets H₂ as a proof of concept; extending to larger systems would require a more expressive ansatz and additional orbital handling.

## Project Structure

| Directory | Purpose |
|---|---|
| `quantum_chemistry/` | Core library: Pauli algebra, Jordan-Wigner mapping, estimation, VQE |
| `quantum_chemistry/molecule/` | Molecular integrals, Hartree-Fock solver, linear algebra |
| `tests/` | 70 pytest tests (unit, integration, end-to-end VQE) |
| `examples/` | Step-by-step tutorials for mapping and estimation |
| `visualization/` | Matplotlib visualizations and animations |
| `h2_data/` | Pre-computed H₂ spin-orbital integrals (34 bond distances) |

## Acknowledgments

Built by [Jeffrey Morais](https://ichor.pages.dev/), Quantum Software Lead at BTQ Technologies.

Inspired by workshop notebooks from [Maxime Dion](https://www.usherbrooke.ca/iq/en/news-events/news/details/54588) at the [Institut Quantique](https://www.usherbrooke.ca/iq/).

