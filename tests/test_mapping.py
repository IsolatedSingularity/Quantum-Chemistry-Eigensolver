"""Tests for quantum_chemistry.mapping — Jordan-Wigner transformation."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_chemistry.mapping import (
    build_one_body_qubit_hamiltonian,
    build_qubit_hamiltonian,
    build_two_body_qubit_hamiltonian,
    creation_annihilation_operators_with_jordan_wigner,
)


class TestJordanWigner:
    """Verify creation / annihilation operator construction."""

    def test_num_operators(self):
        n = 4
        c_ops, a_ops = creation_annihilation_operators_with_jordan_wigner(n)
        assert len(c_ops) == n
        assert len(a_ops) == n

    def test_creation_annihilation_adjoint(self):
        """a†_p should be the adjoint of a_p (coefficients are conjugates)."""
        c_ops, a_ops = creation_annihilation_operators_with_jordan_wigner(4)
        for c, a in zip(c_ops, a_ops):
            np.testing.assert_array_almost_equal(c.coefs, np.conj(a.coefs))

    def test_anticommutation_number(self):
        """{a†_p, a_p} = I  for each p."""
        n = 4
        c_ops, a_ops = creation_annihilation_operators_with_jordan_wigner(n)
        identity_matrix = np.eye(2**n, dtype=complex)
        for p in range(n):
            prod1 = (c_ops[p] * a_ops[p]).combine().apply_threshold()
            prod2 = (a_ops[p] * c_ops[p]).combine().apply_threshold()
            anticomm = (prod1 + prod2).combine().apply_threshold()
            mat = anticomm.to_matrix()
            np.testing.assert_array_almost_equal(mat, identity_matrix)

    def test_anticommutation_different(self):
        """{a†_p, a_q} = 0  for p ≠ q."""
        n = 4
        c_ops, a_ops = creation_annihilation_operators_with_jordan_wigner(n)
        zero_matrix = np.zeros((2**n, 2**n), dtype=complex)
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                prod1 = (c_ops[p] * a_ops[q]).combine().apply_threshold()
                prod2 = (a_ops[q] * c_ops[p]).combine().apply_threshold()
                anticomm = (prod1 + prod2).combine().apply_threshold()
                mat = anticomm.to_matrix()
                if mat is None:
                    # Empty operator after threshold → effectively zero
                    continue
                np.testing.assert_array_almost_equal(mat, zero_matrix, decimal=10)


class TestHamiltonianConstruction:
    """Verify Hamiltonian building from H₂ integrals."""

    @pytest.fixture()
    def h2_data(self):
        from quantum_chemistry.molecule.h2_molecule import load_h2_spin_orbital_integral

        distance, one_body, two_body, nuc_eneg = load_h2_spin_orbital_integral(
            "h2_data",
            "h2_mo_integrals_d_0750.npz",
        )
        return one_body, two_body, nuc_eneg

    def test_hamiltonian_hermitian(self, h2_data):
        one_body, two_body, _ = h2_data
        c_ops, a_ops = creation_annihilation_operators_with_jordan_wigner(one_body.shape[0])
        h = build_qubit_hamiltonian(one_body, two_body, c_ops, a_ops)
        mat = h.to_matrix()
        np.testing.assert_array_almost_equal(mat, mat.conj().T, decimal=10)

    def test_ground_state_energy(self, h2_data):
        """Exact ground state energy of H₂ at 0.75 Å should be ≈ −1.137 Ha."""
        one_body, two_body, nuc_eneg = h2_data
        c_ops, a_ops = creation_annihilation_operators_with_jordan_wigner(one_body.shape[0])
        h = build_qubit_hamiltonian(one_body, two_body, c_ops, a_ops)
        eig_vals = np.linalg.eigvalsh(h.to_matrix())
        gs_energy = np.min(eig_vals) + nuc_eneg
        assert -1.2 < gs_energy < -1.1, f"Unexpected ground-state energy: {gs_energy}"

    def test_one_body_nonzero(self, h2_data):
        one_body, _, _ = h2_data
        c_ops, a_ops = creation_annihilation_operators_with_jordan_wigner(one_body.shape[0])
        h1 = build_one_body_qubit_hamiltonian(one_body, c_ops, a_ops)
        h1 = h1.combine().apply_threshold()
        assert len(h1) > 0

    def test_two_body_nonzero(self, h2_data):
        _, two_body, _ = h2_data
        n = two_body.shape[0]
        c_ops, a_ops = creation_annihilation_operators_with_jordan_wigner(n)
        h2 = build_two_body_qubit_hamiltonian(two_body, c_ops, a_ops)
        h2 = h2.combine().apply_threshold()
        assert len(h2) > 0
