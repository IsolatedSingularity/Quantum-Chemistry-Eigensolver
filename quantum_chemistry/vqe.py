from typing import Callable, Tuple, Union

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.providers import Backend

from quantum_chemistry.estimation import estimate_observable_expectation_value
from quantum_chemistry.pauli import Operator


def h2_ansatz_circuit() -> QuantumCircuit:
    """Build a particle-number-preserving H2 ansatz with 1 parameter.

    Implements a rotation in the {|0101>, |1010>} subspace using a CNOT
    staircase to reduce the double excitation to a single-qubit Ry gate.
    The resulting unitary is exp(-i theta/2 X0 X1 X2 Y3), which preserves
    the total electron count (2 particles) and maps:

        |psi(theta)> = cos(theta/2)|0101> + sin(theta/2)|1010>

    The circuit depth is 6 CNOTs + 1 Ry, much shorter than a full UCC
    doubles decomposition while remaining exact for the H2 two-state problem.

    Returns:
        QuantumCircuit: Parameterized 4-qubit circuit with one parameter (theta).
    """
    varform = QuantumCircuit(4)
    theta = Parameter('theta')

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
    initial_point: Union[np.ndarray, None] = None,
) -> Tuple:
    """Minimizes the expectation value of the Hamiltonian with the ansatz circuit.

    Args:
        hamiltonian (Operator): The Hamiltonian operator
        ansatz_circuit (QuantumCircuit): The parameterized circuit
        backend (Backend): The backend to run the circuits on
        minimizer (Callable): Function that performs the minimization
        initial_point (Union[np.ndarray, None], optional): Initial parameters. Defaults to None.

    Returns:
        Tuple: Result of the minimization
    """
    # Define the cost function
    def cost_function(params):
        # Create parameter dictionary
        param_dict = dict(zip(ansatz_circuit.parameters, params))
        
        # Bind parameters to the circuit
        bound_circuit = ansatz_circuit.assign_parameters(param_dict)
        
        # Calculate expectation value
        expectation_value = estimate_observable_expectation_value(hamiltonian, bound_circuit, backend)
        
        return expectation_value.real
    
    # Set initial point if not provided
    if initial_point is None:
        # For this H2 problem, start with a small angle to initialize near |0101⟩
        initial_point = np.array([0.1] * len(ansatz_circuit.parameters))
    
    # Perform the minimization
    result = minimizer(cost_function, initial_point)
    
    return result
