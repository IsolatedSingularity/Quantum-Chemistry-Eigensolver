from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.providers import Backend

from quantum_chemistry.estimation import EstimationContext, estimate_observable_expectation_value
from quantum_chemistry.pauli import Operator

# Electronic energy above this threshold (Ha) triggers a convergence warning.
# H₂ fully dissociates to two hydrogen atoms at E_elec ≈ −1.0 Ha; anything
# above 0.0 Ha is unphysical for a bound molecule.
_DISSOCIATION_LIMIT_HA: float = 0.0


def h2_ansatz_circuit() -> QuantumCircuit:
    """Build a particle-number-preserving H₂ ansatz with 1 parameter.

    **Particle-number conservation**

    The circuit implements a rotation strictly within the two-dimensional
    subspace spanned by the Hartree-Fock reference |0101⟩ and the doubly
    excited configuration |1010⟩::

        |ψ(θ)⟩ = cos(θ/2)|0101⟩ + sin(θ/2)|1010⟩

    Both basis states carry exactly 2 electrons, so the total electron count
    is conserved for every value of θ.  The unitary realised is
    ``exp(−iθ/2 · X₀X₁X₂Y₃)``, a double-excitation operator.

    **Circuit structure** (6 CNOTs + 1 Ry)::

        q0: ─────────────────────────────────────────────
        q1: ─[X]─[CX ctrl]──────────────────[CX ctrl]───
        q2: ─────[CX targ]─[CX ctrl]──[CX ctrl]─[CX targ]─
        q3: ─[X]────────────[CX targ]─[Ry(θ)]─[CX targ]──

    At θ = 0 the circuit prepares the Hartree-Fock state |0101⟩.

    **Accuracy note**

    This ansatz spans the *exact* two-electron subspace for H₂ in the
    minimal (STO-3G) basis, so the gap between the VQE minimum and the
    FCI energy is purely due to finite shot noise, not an expressibility
    deficit.  At 10 000 shots the expected shot-noise floor is roughly
    3–10 mHa — above the 1.6 mHa chemical-accuracy threshold.  Use
    ≥ 50 000 shots or ``Statevector``-based noiseless evaluation to reach
    chemical accuracy reliably.

    Returns:
        QuantumCircuit: Parameterized 4-qubit circuit with one parameter (theta).
    """
    varform = QuantumCircuit(4)
    theta = Parameter("theta")

    # Prepare reference state |0101> (occupy spin-orbitals 1 and 3)
    varform.x([1, 3])

    # CNOT staircase: maps |0101> -> |0111> and |1010> -> |1111>
    # so both target states differ only at qubit 3
    varform.cx(1, 0)
    varform.cx(2, 1)
    varform.cx(3, 2)

    # Parametric rotation on qubit 3
    varform.ry(theta, 3)

    # Reverse CNOT staircase
    varform.cx(3, 2)
    varform.cx(2, 1)
    varform.cx(1, 0)

    return varform


def minimize_expectation_value(
    hamiltonian: Operator,
    ansatz_circuit: QuantumCircuit,
    backend: Backend,
    minimizer: Callable,
    initial_point: np.ndarray | None = None,
    use_parameter_shift: bool = True,
) -> tuple:
    """Minimize the expectation value ⟨ψ(θ)|H|ψ(θ)⟩ over the ansatz parameters.

    **Minimizer contract**

    The ``minimizer`` callable must accept different signatures depending on
    ``use_parameter_shift``:

    * ``use_parameter_shift=True`` (default, recommended)::

          result = minimizer(cost_fn, x0, grad_fn)

      ``grad_fn`` supplies the exact analytic gradient via the
      **parameter-shift rule** (``∂E/∂θ = ½[E(θ+π/2) − E(θ−π/2)]``).
      This is exact for gates of the form ``exp(−iθG/2)`` and requires
      2 extra circuit evaluations per parameter per iteration.  Pass
      ``jac=grad_fn`` to ``scipy.optimize.minimize``.

    * ``use_parameter_shift=False``::

          result = minimizer(cost_fn, x0)

      The minimizer is responsible for gradient estimation (e.g. scipy's
      built-in finite differences).  **Important:** scipy's default
      finite-difference step (``eps ≈ 1.5e-8``) is far too small for
      shot-noise-dominated cost functions.  Use ``eps`` in the range
      0.01–0.1 (e.g. ``options={"eps": 0.05}``), or prefer
      ``use_parameter_shift=True``.

    **Recommended scipy call** (SLSQP with parameter-shift)::

        from scipy.optimize import minimize

        def my_minimizer(cost_fn, x0, grad_fn):
            return minimize(
                cost_fn, x0,
                method="SLSQP",
                jac=grad_fn,
                options={"maxiter": 100, "ftol": 1e-6},
            )

        result = minimize_expectation_value(H, ansatz, backend, my_minimizer)

    **Post-optimization sanity check**

    If the optimized *electronic* energy (``result.fun``) exceeds
    ``_DISSOCIATION_LIMIT_HA`` (0.0 Ha), a ``UserWarning`` is emitted.
    This indicates the optimizer has not converged — common causes are an
    insufficient iteration budget (``maxiter`` too small) or a step-size
    mismatch when using finite differences without the parameter-shift rule.

    Args:
        hamiltonian (Operator): The qubit Hamiltonian (electronic part only,
            without nuclear repulsion).
        ansatz_circuit (QuantumCircuit): The parameterized ansatz circuit.
        backend (Backend): Qiskit backend to execute circuits on.
        minimizer (Callable): Optimization driver; see contract above.
        initial_point (np.ndarray | None): Initial parameter vector.
            Defaults to ``[0.1] * num_params``.
        use_parameter_shift (bool): If ``True`` (default), supply analytic
            gradients via the parameter-shift rule.  If ``False``, the
            minimizer handles gradient estimation internally.

    Returns:
        tuple: The ``OptimizeResult`` (or equivalent) returned by *minimizer*.
            ``result.fun`` is the minimized electronic energy in Hartree.
            Add the nuclear repulsion energy to obtain the total energy.

    Warns:
        UserWarning: If ``result.fun > 0.0`` Ha, indicating the optimizer
            likely did not converge to a physically meaningful minimum.
    """
    # Build a reusable estimation context to avoid recreating Sampler/PassManager
    ctx = EstimationContext(backend)

    # Define the cost function
    def cost_function(params):
        param_dict = dict(zip(ansatz_circuit.parameters, params))
        bound_circuit = ansatz_circuit.assign_parameters(param_dict)
        expectation_value = estimate_observable_expectation_value(
            hamiltonian,
            bound_circuit,
            backend,
            ctx=ctx,
        )
        return expectation_value.real

    def parameter_shift_gradient(params):
        """Analytic gradient via the parameter-shift rule.

        ∂E/∂θᵢ = ½ · [E(θ + π/2 · eᵢ) − E(θ − π/2 · eᵢ)]

        Exact for gates of the form exp(−iθG/2).  Requires 2 circuit
        evaluations per parameter per call.
        """
        grad = np.zeros_like(params)
        shift = np.pi / 2
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += shift
            params_minus[i] -= shift
            grad[i] = 0.5 * (cost_function(params_plus) - cost_function(params_minus))
        return grad

    # Set initial point if not provided
    if initial_point is None:
        initial_point = np.array([0.1] * len(ansatz_circuit.parameters))

    # Perform the minimization
    if use_parameter_shift:
        result = minimizer(cost_function, initial_point, parameter_shift_gradient)
    else:
        result = minimizer(cost_function, initial_point)

    # --- Post-optimization sanity check -----------------------------------
    # result.fun is the *electronic* energy (nuclear repulsion not included).
    # For any bound H₂ geometry the electronic energy must be negative.
    # A positive value almost always means the optimizer ran out of budget
    # or the finite-difference step was mismatched to the shot-noise level.
    try:
        final_energy = float(result.fun)  # type: ignore[union-attr]
    except (AttributeError, TypeError):
        final_energy = float("nan")

    if not np.isnan(final_energy) and final_energy > _DISSOCIATION_LIMIT_HA:
        warnings.warn(
            f"VQE returned a positive electronic energy ({final_energy:+.4f} Ha), "
            "which is above the dissociation limit (0.0 Ha) and indicates the "
            "optimizer did not converge to a physically meaningful minimum. "
            "Common causes: (1) maxiter too small, (2) finite-difference eps "
            "mismatched to shot noise — use use_parameter_shift=True or set "
            "eps in the range 0.01–0.1.",
            UserWarning,
            stacklevel=2,
        )

    return result  # type: ignore[return-value]
