"""Tests for quantum_chemistry.pauli — PauliString and Operator classes."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_chemistry.pauli import Operator, PauliString


# ─── PauliString ─────────────────────────────────────────────────────────────

class TestPauliStringCreation:
    def test_str_roundtrip(self):
        for label in ("IIII", "XYZZ", "ZZXY", "YYYY", "XXXX"):
            assert str(PauliString.from_str(label)) == label

    def test_len(self):
        ps = PauliString.from_str("IXYZ")
        assert len(ps) == 4

    def test_identity_positions(self):
        ps = PauliString.from_str("YXZI")
        np.testing.assert_array_equal(ps.ids(), [True, False, False, False])

    def test_from_str_invalid_char(self):
        with pytest.raises(ValueError):
            PauliString.from_str("ABCD")


class TestPauliStringMultiplication:
    @pytest.mark.parametrize(
        "a, b, expected_label, expected_phase",
        [
            ("IYZZ", "IIXZ", "IYYI", 1j),
            ("ZZZZ", "XXXI", "YYYZ", -1j),
            ("XXXX", "XXXX", "IIII", 1),
            ("XYZZ", "XYZZ", "IIII", 1),
        ],
    )
    def test_mul(self, a, b, expected_label, expected_phase):
        ps1 = PauliString.from_str(a)
        ps2 = PauliString.from_str(b)
        result, phase = ps1 * ps2
        assert str(result) == expected_label
        assert np.isclose(phase, expected_phase), f"phase {phase} != {expected_phase}"

    def test_mul_length_mismatch(self):
        with pytest.raises(ValueError):
            PauliString.from_str("XY") * PauliString.from_str("XYZ")


class TestPauliStringMatrix:
    def test_identity_matrix(self):
        ps = PauliString.from_str("II")
        np.testing.assert_array_almost_equal(ps.to_matrix(), np.eye(4))

    def test_z_matrix(self):
        ps = PauliString.from_str("Z")
        expected = np.diag([1, -1]).astype(complex)
        np.testing.assert_array_almost_equal(ps.to_matrix(), expected)

    def test_x_matrix(self):
        ps = PauliString.from_str("X")
        expected = np.array([[0, 1], [1, 0]], dtype=complex)
        np.testing.assert_array_almost_equal(ps.to_matrix(), expected)

    def test_zx_matrix(self):
        """ZX string → qubit 0 = X, qubit 1 = Z → kron(X, Z) in code convention."""
        ps = PauliString.from_str("ZX")
        mat = ps.to_matrix()
        # Verify via eigenvalues: ZX has eigenvalues ±1 (each doubly degenerate)
        eig_vals = np.sort(np.linalg.eigvalsh(mat))
        np.testing.assert_array_almost_equal(eig_vals, [-1, -1, 1, 1])
        # Hermitian check
        np.testing.assert_array_almost_equal(mat, mat.conj().T)

    def test_matrix_hermitian(self):
        ps = PauliString.from_str("XYZI")
        mat = ps.to_matrix()
        np.testing.assert_array_almost_equal(mat, mat.conj().T)


class TestPauliStringBits:
    def test_zx_bits_roundtrip(self):
        ps = PauliString.from_str("YXZI")
        bits = ps.to_zx_bits()
        n = len(ps)
        reconstructed = PauliString(bits[:n], bits[n:])
        assert str(reconstructed) == str(ps)


# ─── Operator ────────────────────────────────────────────────────────────────

class TestOperatorCreation:
    def test_from_pauli_mul_coef(self):
        op = 0.5 * PauliString.from_str("IIXZ")
        assert len(op) == 1
        assert np.isclose(op.coefs[0], 0.5)

    def test_addition(self):
        op = 0.5 * PauliString.from_str("XX") + 0.3 * PauliString.from_str("ZZ")
        assert len(op) == 2

    def test_subtraction(self):
        op = 1.0 * PauliString.from_str("XX") - 0.5 * PauliString.from_str("ZZ")
        assert len(op) == 2
        assert np.isclose(op.coefs[1], -0.5)


class TestOperatorAlgebra:
    def test_mul_operator(self):
        op1 = 1 * PauliString.from_str("IIXZ")
        op2 = 1 * PauliString.from_str("IYZZ")
        result = op1 * op2
        assert len(result) == 1
        assert str(result.paulis[0]) == "IYYI"

    def test_combine(self):
        op = (
            1 * PauliString.from_str("XX")
            + 2 * PauliString.from_str("XX")
            + 0.5 * PauliString.from_str("ZZ")
        )
        combined = op.combine()
        assert len(combined) == 2
        # Find the XX term
        for c, p in zip(combined.coefs, combined.paulis):
            if str(p) == "XX":
                assert np.isclose(c, 3.0)

    def test_apply_threshold(self):
        op = 1 * PauliString.from_str("XX") + 1e-12 * PauliString.from_str("ZZ")
        filtered = op.apply_threshold()
        assert len(filtered) == 1

    def test_simplify(self):
        op = (
            1 * PauliString.from_str("XX")
            + 1 * PauliString.from_str("XX")
            + 1e-15 * PauliString.from_str("ZZ")
        )
        simplified = op.simplify()
        assert len(simplified) == 1
        assert np.isclose(simplified.coefs[0], 2.0)

    def test_sort(self):
        op = 0.1 * PauliString.from_str("XX") + 5.0 * PauliString.from_str("ZZ")
        sorted_op = op.sort()
        assert np.abs(sorted_op.coefs[0]) >= np.abs(sorted_op.coefs[1])


class TestOperatorMatrix:
    def test_matrix_matches_pauli_sum(self):
        op = 1 * PauliString.from_str("ZZ") + 2 * PauliString.from_str("XX")
        mat = op.to_matrix()
        expected = (
            PauliString.from_str("ZZ").to_matrix()
            + 2 * PauliString.from_str("XX").to_matrix()
        )
        np.testing.assert_array_almost_equal(mat, expected)

    def test_matrix_hermitian(self):
        """A real-coefficient Pauli operator is Hermitian."""
        op = (
            0.5 * PauliString.from_str("ZZ")
            + 0.3 * PauliString.from_str("XX")
            + 0.2 * PauliString.from_str("II")
        )
        mat = op.to_matrix()
        np.testing.assert_array_almost_equal(mat, mat.conj().T)

    def test_adjoint(self):
        op = (1 + 0.5j) * PauliString.from_str("XY")
        adj = op.adjoint()
        np.testing.assert_array_almost_equal(adj.coefs, np.conj(op.coefs))
