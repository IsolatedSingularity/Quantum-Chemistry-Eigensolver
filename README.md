# Quantum-Chemistry-Eigensolver

<p align="center">
  <a href="https://github.com/IsolatedSingularity/Quantum-Chemistry-Eigensolver/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/IsolatedSingularity/Quantum-Chemistry-Eigensolver/ci.yml?branch=main&label=CI&logo=github" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/install-pip%20install%20--e%20.%5Bdev%5D-brightgreen" alt="pip install"></a>
</p>



![VQE Energy Optimization Curves](https://github.com/IsolatedSingularity/quantum-chemistry-eigensolver/blob/main/visualization/vqe_energy_curves.png?raw=true)

## Objective

This repository implements a quantum chemistry eigensolver for simulating small molecular systems, with a focus on the H₂ molecule. Quantum chemistry is one of the most promising near-term applications of quantum computing, using variational quantum algorithms to calculate molecular ground state energies.

The core of this implementation is the **Variational Quantum Eigensolver (VQE)**, a hybrid quantum-classical algorithm that approximates the ground state energy of quantum systems. For molecular systems, this energy is determined by the electronic Hamiltonian:

$\hat{H} = \sum_{p,q} h_{pq} a_p^\dagger a_q + \frac{1}{2} \sum_{p,q,r,s} h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s$

where $a_p^\dagger$ and $a_q$ are fermionic creation and annihilation operators, while $h_{pq}$ and $h_{pqrs}$ represent one- and two-electron integrals.

**Goal:** Simulate the H₂ molecule's energy landscape at various bond distances using the VQE algorithm, visualize molecular orbitals, and demonstrate the convergence of the optimization process toward the ground state energy.

<p align="center">
  <img src="./visualization/apple.gif?raw=true" alt="apple" width="52" height="50" />
</p>

## Theoretical Background

Quantum chemistry calculations begin with mapping the molecular Hamiltonian to a qubit representation. The most common approach uses the Jordan–Wigner transformation, which converts fermionic operators to Pauli operators:

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
The molecular simulation begins by loading pre-computed spin orbital integrals for H₂ at various bond distances. These integrals encode the one-electron kinetic and nuclear attraction terms ($h_{pq}$) as well as two-electron repulsion terms ($h_{pqrs}$). The data files are organized by bond distance, allowing systematic study of the dissociation curve. Nuclear repulsion energy is also extracted for computing total molecular energies.

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
The Jordan-Wigner transformation converts the fermionic Hamiltonian into a qubit operator by mapping creation and annihilation operators to Pauli strings. Each fermionic operator $a_p^\dagger$ becomes a product of Pauli X, Y operators at position $p$ with a Z-string on all preceding qubits to preserve anticommutation relations. The one-body terms ($a_p^\dagger a_q$) and two-body terms ($a_p^\dagger a_q^\dagger a_r a_s$) are systematically transformed and combined into a single `Operator` object representing the qubit Hamiltonian.

```python
def build_qubit_hamiltonian(one_body, two_body, creation_ops, annihilation_ops):
    """Build qubit Hamiltonian from fermionic integrals using JW mapping."""
    h1 = build_one_body_qubit_hamiltonian(one_body, creation_ops, annihilation_ops)
    h2 = build_two_body_qubit_hamiltonian(two_body, creation_ops, annihilation_ops)
    return (h1 + h2).combine().apply_threshold().sort()
```

### 3. VQE Implementation
The VQE algorithm uses a hybrid quantum-classical approach where a parameterized quantum circuit (ansatz) prepares trial wavefunctions, and a classical optimizer adjusts parameters to minimize energy. For H₂, a particle-number-preserving ansatz starts from the Hartree-Fock state |0101⟩ and applies a CNOT staircase to reduce the double excitation to a single Ry rotation, producing states of the form $\cos(\theta/2)|0101\rangle + \sin(\theta/2)|1010\rangle$ that always contain exactly 2 electrons. The cost function computes the Hamiltonian expectation value by grouping Pauli terms into qubitwise-commuting (QWC) sets and summing weighted contributions. Analytic gradients are computed via the parameter-shift rule.

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
The dissociation visualization demonstrates the fundamental chemistry of bond breaking by plotting how electronic energy, nuclear repulsion, and total energy evolve as the H-H bond stretches. An animation shows the molecule transitioning from compressed state through equilibrium (~0.74 Å) to full dissociation, with synchronized updates of the molecular geometry and potential energy curve. This illustrates why molecules have stable bond lengths: the balance between attractive electronic forces and nuclear repulsion.

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
This visualization illustrates molecular orbital theory by showing how atomic 1s orbitals on each hydrogen atom combine to form bonding (σ) and antibonding (σ*) molecular orbitals. The bonding orbital exhibits constructive interference with increased electron density between nuclei, lowering the energy. The antibonding orbital shows destructive interference with a nodal plane between atoms. In H₂'s ground state, both electrons occupy the lower-energy bonding orbital.

```python
def create_molecular_orbital_visualization(save_path):
    """Create static visualization of H2 molecular orbitals."""
    h1_1s = h1s(X, Y, h1_pos)  # Atomic orbital on H1
    h2_1s = h1s(X, Y, h2_pos)  # Atomic orbital on H2
    bonding_mo = h1_1s + h2_1s      # Constructive interference
    antibonding_mo = h1_1s - h2_1s  # Destructive interference
```

### 6. VQE Energy Curves Visualization
This visualization shows the real VQE energy landscape computed from the Jordan-Wigner Hamiltonian. The left panel plots energy vs the single ansatz parameter theta for eight representative bond distances, colored by distance. The right panel overlays the exact diagonalization dissociation curve with the VQE-optimized minimum at each distance, confirming that the variational bound is tight.

```python
def create_vqe_energy_curves_visualization(save_path):
    """Build VQE energy curves from real Hamiltonian data."""
    distances, molecule_data = load_h2_spin_orbital_integrals(data_path)
    ansatz = h2_ansatz_circuit()
    # Left: energy vs theta for 8 bond distances
    # Right: exact dissociation vs VQE minimum
```

## Results

The implementation successfully simulates the H₂ molecule and visualizes key quantum chemistry concepts:

1. **H₂ Molecular Orbitals**:

![H2 Molecular Orbitals](https://github.com/IsolatedSingularity/quantum-chemistry-eigensolver/blob/main/visualization/h2_molecular_orbitals.png?raw=true)

This visualization shows the atomic 1s orbitals of individual hydrogen atoms and how they combine to form molecular orbitals. The bonding orbital (lower left) shows constructive interference between atomic orbitals, while the antibonding orbital (lower right) shows destructive interference.

2. **H₂ Dissociation Curve**:

![H2 Dissociation Animation](https://github.com/IsolatedSingularity/quantum-chemistry-eigensolver/blob/main/visualization/h2_dissociation.gif?raw=true)

The animation demonstrates how the H₂ molecule's energy changes as the bond distance varies. At the equilibrium distance (around 0.74 Å), the energy reaches its minimum. As the atoms move either closer (compression) or farther apart (dissociation), the energy increases.

3. **VQE Energy Optimization Curves**:

![VQE Energy Optimization Curves](https://github.com/IsolatedSingularity/quantum-chemistry-eigensolver/blob/main/visualization/vqe_energy_curves.png?raw=true)

The left panel shows the single-parameter energy landscape for eight bond distances computed from the full Jordan-Wigner Hamiltonian. The right panel overlays the exact diagonalization dissociation curve (black line) with the VQE-optimized minimum at each distance (dots), confirming the variational bound is tight across the entire dissociation range.

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
> This implementation serves as an educational resource for understanding quantum algorithms in chemistry applications rather than as a production-level quantum chemistry tool.

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

Based on the quantum chemistry workshop notebooks created by [Maxime Dion](https://www.usherbrooke.ca/iq/en/news-events/news/details/54588) at the [Institut Quantique](https://www.usherbrooke.ca/iq/).

