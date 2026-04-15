"""
Tutorial 1 — Mapping
====================
Based on the quantum-chemistry workshop by Maxime Dion (QSciTech-QuantumBC).

This script walks through:
  1. Creating and manipulating PauliStrings
  2. Building Operators (linear combinations of PauliStrings)
  3. Loading H₂ molecular integrals
  4. Jordan-Wigner mapping of fermionic → qubit Hamiltonians
"""

from __future__ import annotations

import numpy as np

from quantum_chemistry.mapping import (
    build_one_body_qubit_hamiltonian,
    build_qubit_hamiltonian,
    build_two_body_qubit_hamiltonian,
    creation_annihilation_operators_with_jordan_wigner,
)
from quantum_chemistry.molecule.h2_molecule import (
    load_h2_spin_orbital_integral,
    load_h2_spin_orbital_integrals,
)
from quantum_chemistry.pauli import Operator, PauliString

# ── Helpers ──────────────────────────────────────────────────────────────────

PAULI_MAP = {(False, False): "I", (True, False): "Z", (False, True): "X", (True, True): "Y"}


def show_pauli_mapping(z_bits, x_bits) -> None:
    """Print a table mapping z/x bit pairs to Pauli labels."""
    header = f"{'qubit':>6} | {'z_bit':>5} | {'x_bit':>5} | Pauli"
    print(header)
    print("-" * len(header))
    for i, (z, x) in enumerate(zip(z_bits, x_bits)):
        label = PAULI_MAP[(bool(z), bool(x))]
        print(f"{i:>6} | {int(z):>5} | {int(x):>5} | {label}")


def print_matrix(mat) -> None:
    for row in mat:
        print("  ".join(f"{v.real:+.1f}{v.imag:+.1f}j" for v in row))


# ── 1. PauliString ──────────────────────────────────────────────────────────

print("=" * 60)
print("1. PauliString Basics")
print("=" * 60)

# Create YXZI via z_bits / x_bits
z_bits = np.array([0, 1, 0, 1], dtype=bool)
x_bits = np.array([0, 0, 1, 1], dtype=bool)
show_pauli_mapping(z_bits, x_bits)
ps = PauliString(z_bits, x_bits)
print(f"PauliString: {ps}")  # YXZI

# Create ZZXY
z_bits = np.array([1, 0, 1, 1], dtype=bool)
x_bits = np.array([1, 1, 0, 0], dtype=bool)
print(f"ZZXY check:  {PauliString(z_bits, x_bits)}")

# from_str round-trip
ps = PauliString.from_str("YXZI")
print(f"from_str:    {ps}")
print(f"zx_bits:     {ps.to_zx_bits()}")
print(f"xz_bits:     {ps.to_xz_bits()}")
print(f"identities:  {ps.ids()}")

# Multiplication
ps1 = PauliString.from_str("IYZZ")
ps2 = PauliString.from_str("IIXZ")
new_ps, phase = ps1 * ps2
print(f"\n{ps1} × {ps2} = {phase} * {new_ps}")

ps1 = PauliString.from_str("ZZZZ")
ps2 = PauliString.from_str("XXXI")
new_ps, phase = ps1 * ps2
print(f"{ps1} × {ps2} = {phase} * {new_ps}")

# Matrix representation
ps = PauliString.from_str("ZX")
print(f"\nMatrix of {ps}:")
print_matrix(ps.to_matrix())

# ── 2. Operator ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. Operator Algebra")
print("=" * 60)

op = Operator(
    np.array([0.5, 0.5]),
    np.array([PauliString.from_str("IIXZ"), PauliString.from_str("IYZZ")]),
)
print(f"Operator: {op}")

# Multiplication of two single-term operators
op1 = 1 * PauliString.from_str("IIXZ")
op2 = 1 * PauliString.from_str("IYZZ")
print(f"Product:  {op1 * op2}")

# Add
op_sum = 0.5 * PauliString.from_str("IIXZ") + 0.5 * PauliString.from_str("IYZZ")
print(f"Sum:      {op_sum}")

# Combine + threshold
op_a = 1 * PauliString.from_str("IIIZ") - 0.5 * PauliString.from_str("IIZZ")
op_b = 1 * PauliString.from_str("ZZZI") + 0.5 * PauliString.from_str("ZZII")
product = (op_a * op_b).combine()
print(f"Before threshold: {product}")
product = product.apply_threshold()
print(f"After threshold:  {product}")

# Operator matrix
small_op = 1 * PauliString.from_str("ZZ") + 2 * PauliString.from_str("XX")
print(f"\nMatrix of ({small_op}):")
print_matrix(small_op.to_matrix())

# ── 3. Molecular Hamiltonian ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. Loading H₂ Integrals & Building the Qubit Hamiltonian")
print("=" * 60)

DATA_PATH = "h2_data"

distance, one_body, two_body, nuc_eneg = load_h2_spin_orbital_integral(
    DATA_PATH,
    "h2_mo_integrals_d_0750.npz",
)
print(f"Distance: {distance:.3f} Å")
print(f"Nuclear repulsion: {nuc_eneg:.6f} Ha")

distances, molecule_datas = load_h2_spin_orbital_integrals(DATA_PATH)
print(f"Loaded {len(distances)} distances: {distances[0]:.2f} – {distances[-1]:.2f} Å")

# Jordan-Wigner operators
creation_ops, annihilation_ops = creation_annihilation_operators_with_jordan_wigner(4)
print(f"\nCreation operators ({len(creation_ops)}):")
for i, op in enumerate(creation_ops):
    print(f"  a†_{i} = {op}")

# One-body Hamiltonian
h1 = build_one_body_qubit_hamiltonian(one_body, creation_ops, annihilation_ops)
h1 = h1.apply_threshold().combine().apply_threshold().sort()
print(f"\nOne-body Hamiltonian ({len(h1)} terms):")
print(h1)

# Two-body Hamiltonian
h2 = build_two_body_qubit_hamiltonian(two_body, creation_ops, annihilation_ops)
h2 = h2.apply_threshold().combine().apply_threshold().sort()
print(f"\nTwo-body Hamiltonian ({len(h2)} terms):")
print(h2)

# Full qubit Hamiltonian
qubit_hamiltonian = build_qubit_hamiltonian(one_body, two_body, creation_ops, annihilation_ops)
print(f"\nFull Qubit Hamiltonian ({len(qubit_hamiltonian)} terms):")
print(qubit_hamiltonian)

print("\n✓ Tutorial 1 complete — you have mapped H₂ to a qubit Hamiltonian!")
