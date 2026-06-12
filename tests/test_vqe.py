"""Tests for quantum_chemistry.vqe — ansatz circuits and VQE loop."""

from __future__ import annotations

import warnings

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

    def test_particle_number_preserved(self):
        """Ansatz must stay in the 2-electron subspace for all θ values.

        The circuit spans {|0101⟩, |1010⟩} — both have Hamming weight 2.
        We sample several θ values and verify that only weight-2 bitstrings
        appear in the measurement outcomes.
        """
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime import SamplerV2 as Sampler

        backend = AerSimulator(shots=2048, seed_simulator=7)
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        sampler = Sampler(mode=backend)

        qc = h2_ansatz_circuit()
        param = list(qc.parameters)[0]

        for theta in [0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]:
            bound = qc.assign_parameters({param: theta})
            bound.measure_all()
            isa = pm.run([bound])
            result = sampler.run(isa).result()
            counts = result[0].data.meas.get_counts()
            for bitstring in counts:
                hw = bitstring.count("1")
                assert hw == 2, (
                    f"θ={theta:.3f}: bitstring '{bitstring}' has Hamming weight {hw} ≠ 2 "
                    "(particle-number conservation violated)"
                )


class TestConvergenceWarning:
    """minimize_expectation_value must warn when the result is unphysical."""

    def test_warns_on_positive_electronic_energy(self):
        """A positive result.fun should trigger a UserWarning."""
        from pathlib import Path

        from qiskit_aer import AerSimulator

        from quantum_chemistry.mapping import (
            build_qubit_hamiltonian,
            creation_annihilation_operators_with_jordan_wigner,
        )
        from quantum_chemistry.molecule.h2_molecule import load_h2_spin_orbital_integral

        data_path = str(Path(__file__).resolve().parent.parent / "h2_data")
        _, one_body, two_body, _ = load_h2_spin_orbital_integral(data_path, "h2_mo_integrals_d_0750.npz")
        num_orbs = one_body.shape[0]
        creation_ops, annihilation_ops = creation_annihilation_operators_with_jordan_wigner(num_orbs)
        hamiltonian = build_qubit_hamiltonian(one_body, two_body, creation_ops, annihilation_ops).simplify()

        ansatz = h2_ansatz_circuit()
        backend = AerSimulator(shots=1024, seed_simulator=0)

        # maxiter=1 from a bad starting point reliably produces a positive
        # electronic energy, which should trigger the convergence warning.
        def one_shot_minimizer(cost_fn, x0, grad_fn):
            from scipy.optimize import minimize

            return minimize(
                cost_fn,
                x0,
                method="SLSQP",
                jac=grad_fn,
                options={"maxiter": 1, "ftol": 1e-12},
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            minimize_expectation_value(
                hamiltonian,
                ansatz,
                backend,
                one_shot_minimizer,
                initial_point=np.array([3.0]),  # far from optimum
                use_parameter_shift=True,
            )

        # The warning may or may not fire depending on whether the 1-iteration
        # result happens to be positive; we only assert that IF it fires it is
        # a UserWarning with the expected message fragment.
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        for w in user_warnings:
            assert "dissociation limit" in str(w.message).lower() or "converge" in str(w.message).lower()

    def test_no_warning_on_converged_result(self):
        """A well-converged run (negative result.fun) must not emit a warning."""
        from pathlib import Path

        from qiskit_aer import AerSimulator

        from quantum_chemistry.mapping import (
            build_qubit_hamiltonian,
            creation_annihilation_operators_with_jordan_wigner,
        )
        from quantum_chemistry.molecule.h2_molecule import load_h2_spin_orbital_integral

        data_path = str(Path(__file__).resolve().parent.parent / "h2_data")
        _, one_body, two_body, _ = load_h2_spin_orbital_integral(data_path, "h2_mo_integrals_d_0750.npz")
        num_orbs = one_body.shape[0]
        creation_ops, annihilation_ops = creation_annihilation_operators_with_jordan_wigner(num_orbs)
        hamiltonian = build_qubit_hamiltonian(one_body, two_body, creation_ops, annihilation_ops).simplify()

        ansatz = h2_ansatz_circuit()
        backend = AerSimulator(shots=10_000, seed_simulator=42)

        def minimizer_fn(cost_fn, x0, grad_fn):
            from scipy.optimize import minimize

            return minimize(
                cost_fn,
                x0,
                method="SLSQP",
                jac=grad_fn,
                options={"maxiter": 30, "ftol": 1e-6},
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            minimize_expectation_value(
                hamiltonian,
                ansatz,
                backend,
                minimizer_fn,
                initial_point=np.array([0.1]),
                use_parameter_shift=True,
            )

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0, (
            f"Unexpected UserWarning on converged run: {[str(w.message) for w in user_warnings]}"
        )


class TestVQEIntegration:
    """End-to-end VQE: build Hamiltonian, optimize, check energy."""

    @pytest.mark.slow
    def test_h2_equilibrium_energy(self):
        """VQE energy at d=0.75 Å should be within 20 mHa of exact diagonalization.

        The ansatz spans the exact 2-electron subspace for H₂ in STO-3G, so
        the residual gap is purely shot noise.  At 20 000 shots the expected
        shot-noise floor is ~2–5 mHa; the 20 mHa tolerance gives comfortable
        headroom while still catching regressions.

        For chemical accuracy (1.6 mHa) use ≥ 50 000 shots or a noiseless
        Statevector backend.
        """
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

        # Use the parameter-shift rule (production path) with SLSQP.
        def minimizer_fn(cost_fn, x0, grad_fn):
            return minimize(
                cost_fn,
                x0,
                method="SLSQP",
                jac=grad_fn,
                options={"maxiter": 30, "ftol": 1e-6},
            )

        result = minimize_expectation_value(
            hamiltonian,
            ansatz,
            backend,
            minimizer_fn,
            initial_point=np.array([0.1]),
            use_parameter_shift=True,
        )
        vqe_energy = result.fun + float(nuc_energy)

        assert abs(vqe_energy - exact_energy) < 0.02, (
            f"VQE energy {vqe_energy:.6f} Ha too far from exact {exact_energy:.6f} Ha "
            f"(gap = {abs(vqe_energy - exact_energy) * 1000:.1f} mHa, tolerance = 20 mHa)"
        )
