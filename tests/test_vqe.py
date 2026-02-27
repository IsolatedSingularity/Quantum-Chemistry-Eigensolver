"""Tests for quantum_chemistry.vqe — ansatz circuits."""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit

from quantum_chemistry.vqe import h2_ansatz_circuit


class TestH2Ansatz:
    def test_num_qubits(self):
        qc = h2_ansatz_circuit()
        assert qc.num_qubits == 4

    def test_single_parameter(self):
        qc = h2_ansatz_circuit()
        assert len(qc.parameters) == 1

    def test_assigns_without_error(self):
        qc = h2_ansatz_circuit()
        param = list(qc.parameters)[0]
        bound = qc.assign_parameters({param: 0.5})
        assert len(bound.parameters) == 0

    def test_hf_state_at_theta_zero(self):
        """At θ = 0 the ansatz should prepare |0101⟩ (the Hartree-Fock state)."""
        from qiskit_aer import AerSimulator

        qc = h2_ansatz_circuit()
        param = list(qc.parameters)[0]
        bound = qc.assign_parameters({param: 0.0})
        bound.measure_all()

        from qiskit_ibm_runtime import SamplerV2 as Sampler
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        backend = AerSimulator(shots=1024)
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa = pm.run([bound])
        sampler = Sampler(mode=backend)
        result = sampler.run(isa).result()
        counts = result[0].data.meas.get_counts()

        # |0101⟩ should dominate (bit-reversed Qiskit convention → "0101" or "1010")
        dominant = max(counts, key=counts.get)
        assert dominant in ("0101", "1010"), f"Unexpected dominant state: {dominant}"
