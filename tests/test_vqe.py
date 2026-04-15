"""Tests for quantum_chemistry.vqe — ansatz circuits and VQE loop."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_chemistry.vqe import h2_ansatz_circuit, minimize_expectation_value


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

        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2 as Sampler

        backend = AerSimulator(shots=1024)
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa = pm.run([bound])
        sampler = Sampler(mode=backend)
        result = sampler.run(isa).result()
        counts = result[0].data.meas.get_counts()

        # |0101⟩ should dominate (bit-reversed Qiskit convention → "0101" or "1010")
        dominant = max(counts, key=counts.get)
        assert dominant in ("0101", "1010"), f"Unexpected dominant state: {dominant}"


class TestVQEIntegration:
    """End-to-end VQE: build Hamiltonian, optimize, check energy."""

    @pytest.mark.slow
    def test_h2_equilibrium_energy(self):
        """VQE energy at d=0.75 Å should be within chemical accuracy of exact."""
        from pathlib import Path

        from qiskit_aer import AerSimulator
        from scipy.optimize import minimize

        from quantum_chemistry.mapping import (
            build_qubit_hamiltonian,
            creation_annihilation_operators_with_jordan_wigner,
        )
        from quantum_chemistry.molecule.h2_molecule import load_h2_spin_orbital_integral

        data_path = str(Path(__file__).resolve().parent.parent / "h2_data")
        filename = "h2_mo_integrals_d_0750.npz"
        distance, one_body, two_body, nuc_energy = load_h2_spin_orbital_integral(
            data_path,
            filename,
        )

        num_orbs = one_body.shape[0]
        creation_ops, annihilation_ops = creation_annihilation_operators_with_jordan_wigner(num_orbs)
        hamiltonian = build_qubit_hamiltonian(
            one_body,
            two_body,
            creation_ops,
            annihilation_ops,
        ).simplify()

        # Exact ground state from diagonalization
        eigvals = np.linalg.eigvalsh(hamiltonian.to_matrix().real)
        exact_energy = eigvals[0] + float(nuc_energy)

        ansatz = h2_ansatz_circuit()
        backend = AerSimulator(shots=20_000, seed_simulator=42)

        def minimizer_fn(f, x0):
            return minimize(
                f,
                x0,
                method="SLSQP",
                options={"maxiter": 30, "eps": 0.05, "ftol": 1e-6},
            )

        result = minimize_expectation_value(
            hamiltonian,
            ansatz,
            backend,
            minimizer_fn,
            initial_point=np.array([0.1]),
            use_parameter_shift=False,
        )
        vqe_energy = result.fun + float(nuc_energy)

        # Chemical accuracy: 1.6 mHa = 0.0016 Ha
        assert abs(vqe_energy - exact_energy) < 0.02, (
            f"VQE energy {vqe_energy:.6f} Ha too far from exact {exact_energy:.6f} Ha"
        )
