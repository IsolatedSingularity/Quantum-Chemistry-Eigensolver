"""Tests for quantum_chemistry.estimation — Pauli expectation values."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_chemistry.estimation import (
    bitstring_to_bits,
    diagonal_pauli_eigenvalue,
    diagonal_pauli_expectation_value,
    diagonal_pauli_with_circuit,
    group_paulis_qwc,
    prepare_estimation_circuits_and_diagonal_paulis,
    qubitwise_commutes,
    qwc_measurement_basis,
)
from quantum_chemistry.pauli import PauliString


class TestBitstringConversion:
    def test_basic(self):
        bits = bitstring_to_bits("1100")
        # Little-endian: '1100' → [0, 0, 1, 1]
        np.testing.assert_array_equal(bits, [False, False, True, True])

    def test_all_zeros(self):
        bits = bitstring_to_bits("0000")
        np.testing.assert_array_equal(bits, [False, False, False, False])

    def test_all_ones(self):
        bits = bitstring_to_bits("111")
        np.testing.assert_array_equal(bits, [True, True, True])


class TestDiagonalPauli:
    def test_diagonal_circuit_produces_z_and_i_only(self):
        ps = PauliString.from_str("ZIXY")
        diag_ps, circ = diagonal_pauli_with_circuit(ps)
        # Diagonal means all x_bits are False
        assert not np.any(diag_ps.x_bits)
        # Z/I remain where there was any non-identity operator
        assert str(diag_ps) == "ZIZZ"

    def test_already_diagonal(self):
        ps = PauliString.from_str("ZZII")
        diag_ps, circ = diagonal_pauli_with_circuit(ps)
        assert str(diag_ps) == "ZZII"
        assert circ.size() == 0  # No gates needed

    @pytest.mark.parametrize(
        "bitstring, expected",
        [("0001", 1), ("0100", -1), ("1100", 1), ("1110", -1)],
    )
    def test_eigenvalue(self, bitstring, expected):
        diag = PauliString.from_str("ZZZI")
        result = diagonal_pauli_eigenvalue(diag, bitstring_to_bits(bitstring))
        assert result == expected

    def test_expectation_value(self):
        diag = PauliString.from_str("ZIZZ")
        counts = {"0110": 25, "1001": 75}
        exp_val = diagonal_pauli_expectation_value(diag, counts)
        assert np.isclose(exp_val, 0.5)

    def test_expectation_all_same_eigenvalue(self):
        """If all outcomes give +1, expectation should be 1.0."""
        diag = PauliString.from_str("II")
        counts = {"00": 50, "01": 50}
        exp_val = diagonal_pauli_expectation_value(diag, counts)
        assert np.isclose(exp_val, 1.0)


class TestEstimationCircuits:
    def test_circuit_count(self):
        from qiskit.circuit import QuantumCircuit

        paulis = [PauliString.from_str("XZ"), PauliString.from_str("ZX")]
        state_qc = QuantumCircuit(2)
        circuits, diag_paulis = prepare_estimation_circuits_and_diagonal_paulis(
            paulis,
            state_qc,
        )
        assert len(circuits) == 2
        assert len(diag_paulis) == 2

    def test_circuits_have_measurements(self):
        from qiskit.circuit import QuantumCircuit

        paulis = [PauliString.from_str("XY")]
        state_qc = QuantumCircuit(2)
        circuits, _ = prepare_estimation_circuits_and_diagonal_paulis(paulis, state_qc)
        # Circuit should contain measurement instructions
        op_names = [inst.operation.name for inst in circuits[0].data]
        assert "measure" in op_names


class TestQubitwiseCommutes:
    def test_same_paulis_commute(self):
        p = PauliString.from_str("XYZ")
        assert qubitwise_commutes(p, p)

    def test_identity_commutes_with_anything(self):
        p1 = PauliString.from_str("III")
        p2 = PauliString.from_str("XYZ")
        assert qubitwise_commutes(p1, p2)

    def test_non_overlapping_commute(self):
        p1 = PauliString.from_str("XII")
        p2 = PauliString.from_str("IYZ")
        assert qubitwise_commutes(p1, p2)

    def test_conflicting_do_not_commute(self):
        p1 = PauliString.from_str("XZ")
        p2 = PauliString.from_str("ZX")
        assert not qubitwise_commutes(p1, p2)

    def test_z_and_zi(self):
        p1 = PauliString.from_str("ZI")
        p2 = PauliString.from_str("ZZ")
        assert qubitwise_commutes(p1, p2)


class TestGroupPaulisQwc:
    def test_all_commuting_single_group(self):
        paulis = [
            PauliString.from_str("ZI"),
            PauliString.from_str("IZ"),
            PauliString.from_str("ZZ"),
        ]
        groups = group_paulis_qwc(paulis)
        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1, 2]

    def test_conflicting_separate_groups(self):
        paulis = [
            PauliString.from_str("XZ"),
            PauliString.from_str("ZX"),
        ]
        groups = group_paulis_qwc(paulis)
        assert len(groups) == 2

    def test_all_indices_covered(self):
        paulis = [
            PauliString.from_str("XI"),
            PauliString.from_str("IZ"),
            PauliString.from_str("ZI"),
            PauliString.from_str("IX"),
        ]
        groups = group_paulis_qwc(paulis)
        all_indices = sorted(idx for g in groups for idx in g)
        assert all_indices == [0, 1, 2, 3]


class TestQwcMeasurementBasis:
    def test_basis_covers_all_non_identity(self):
        paulis = [
            PauliString.from_str("ZI"),
            PauliString.from_str("IZ"),
        ]
        combined, circuit = qwc_measurement_basis(paulis)
        assert str(combined) == "ZZ"

    def test_mixed_basis(self):
        paulis = [
            PauliString.from_str("XI"),
            PauliString.from_str("IX"),
        ]
        combined, circuit = qwc_measurement_basis(paulis)
        assert str(combined) == "XX"
