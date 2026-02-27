"""
Tutorial 2 — Estimation & VQE
==============================
Based on the quantum-chemistry workshop by Maxime Dion (QSciTech-QuantumBC).

This script walks through:
  1. Parameterised ansatz circuits for H₂
  2. Pauli-basis measurement / diagonal expectation values
  3. Full observable estimation on a simulator
  4. Running the Variational Quantum Eigensolver (VQE)
  5. Comparison with exact diagonalisation
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from quantum_chemistry.estimation import (
    bitstring_to_bits,
    diagonal_pauli_eigenvalue,
    diagonal_pauli_expectation_value,
    diagonal_pauli_with_circuit,
    estimate_observable_expectation_value,
    estimate_paulis_expectation_values,
    prepare_estimation_circuits_and_diagonal_paulis,
)
from quantum_chemistry.mapping import (
    build_qubit_hamiltonian,
    creation_annihilation_operators_with_jordan_wigner,
)
from quantum_chemistry.molecule.h2_molecule import load_h2_spin_orbital_integral
from quantum_chemistry.pauli import Operator, PauliString
from quantum_chemistry.vqe import h2_ansatz_circuit, minimize_expectation_value

# ── 1. Parameterised Circuits ───────────────────────────────────────────────

print("=" * 60)
print("1. Variational Quantum Circuits")
print("=" * 60)

# Simple 2-qubit example
a = Parameter("a")
b = Parameter("b")
example_qc = QuantumCircuit(2)
example_qc.ry(a, 0)
example_qc.rz(b, 0)
example_qc.cx(0, 1)
print(example_qc.draw("text"))

# H₂ 1-parameter ansatz
ansatz_1p = h2_ansatz_circuit()
print("\nH₂ 1-param ansatz:")
print(ansatz_1p.draw("text"))

# ── 2. Diagonal Pauli Measurements ──────────────────────────────────────────

print("\n" + "=" * 60)
print("2. Diagonal Pauli Measurements")
print("=" * 60)

ps = PauliString.from_str("ZIXY")
diag_ps, diag_circuit = diagonal_pauli_with_circuit(ps)
print(f"Original: {ps}  →  Diagonal: {diag_ps}")
print(diag_circuit.draw("text"))

# Eigenvalues
diag = PauliString.from_str("ZZZI")
for bs in ["0001", "0100", "1100", "1110"]:
    ev = diagonal_pauli_eigenvalue(diag, bitstring_to_bits(bs))
    print(f"  eigenvalue({bs}) = {ev:+d}")

# Expectation value from counts
diag = PauliString.from_str("ZIZZ")
counts = {"0110": 25, "1001": 75}
exp_val = diagonal_pauli_expectation_value(diag, counts)
print(f"\n  ⟨ZIZZ⟩ from counts {counts} = {exp_val:.4f}")

# ── 3. Full Estimation Pipeline ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. Observable Estimation on AerSimulator")
print("=" * 60)

backend = AerSimulator()

paulis = [PauliString.from_str("ZXZX"), PauliString.from_str("XZXZ")]
state_circuit = QuantumCircuit(4)
state_circuit.x([0, 2])
state_circuit.h([1, 3])

exp_values = estimate_paulis_expectation_values(paulis, state_circuit, backend)
print(f"Pauli expectations: {exp_values}")

operator = Operator(np.array([1, -1]), np.array(paulis))
obs_exp = estimate_observable_expectation_value(operator, state_circuit, backend)
print(f"Observable ⟨O⟩ = {obs_exp:.6f}")

# ── 4. VQE for H₂ ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. VQE Optimisation")
print("=" * 60)

DATA_PATH = "h2_data"

distance, one_body, two_body, nuc_eneg = load_h2_spin_orbital_integral(
    DATA_PATH, "h2_mo_integrals_d_0750.npz",
)
creation_ops, annihilation_ops = creation_annihilation_operators_with_jordan_wigner(4)
qubit_hamiltonian = build_qubit_hamiltonian(one_body, two_body, creation_ops, annihilation_ops)

# Evaluate H on |0101⟩
state_0101 = QuantumCircuit(4)
state_0101.x([1, 3])
e_0101 = estimate_observable_expectation_value(qubit_hamiltonian, state_0101, backend)
print(f"⟨H⟩ for |0101⟩ = {e_0101.real:.4f}  (expect ≈ −1.83)")

# VQE minimisation
ansatz = h2_ansatz_circuit()
minimizer_fn = lambda cost_fn, x0: minimize(
    cost_fn, x0,
    method="SLSQP",
    options={"maxiter": 5, "eps": 1e-1, "ftol": 1e-4, "disp": True, "iprint": 2},
)

result = minimize_expectation_value(qubit_hamiltonian, ansatz, backend, minimizer_fn)
opt_energy = result.fun + nuc_eneg
print(f"\nVQE ground-state molecular energy: {opt_energy:.6f} Ha")

# ── 5. Exact Comparison ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("5. Exact Diagonalisation Comparison")
print("=" * 60)

H_matrix = qubit_hamiltonian.to_matrix()
eig_vals = np.sort(np.linalg.eigvalsh(H_matrix))
exact_gs = eig_vals[0] + nuc_eneg
print(f"Exact ground-state energy:  {exact_gs:.6f} Ha")
print(f"VQE ground-state energy:    {opt_energy:.6f} Ha")
print(f"Error:                      {abs(opt_energy - exact_gs):.6f} Ha")

# Bar plot of exact ground-state wavefunction
eig_vals_full, eig_vecs = np.linalg.eigh(H_matrix)
order = np.argsort(eig_vals_full)
gs_vector = eig_vecs[:, order[0]]

n_qubits = 4
basis_labels = [f"|{i:0{n_qubits}b}⟩" for i in range(2**n_qubits)]

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(basis_labels, np.abs(gs_vector) ** 2)
ax.set_xlabel("Computational Basis State")
ax.set_ylabel("Probability")
ax.set_title("Exact Ground State Wavefunction")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("ground_state_wavefunction.png", dpi=150)
print("Saved ground_state_wavefunction.png")

print("\n✓ Tutorial 2 complete!")
