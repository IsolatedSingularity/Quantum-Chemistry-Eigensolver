from typing import Callable, Tuple, Union

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.providers import Backend

from quantum_chemistry.estimation import estimate_observable_expectation_value
from quantum_chemistry.pauli import Operator


def h2_ansatz_circuit() -> QuantumCircuit:
    """Build the H2 ansatz circuit with 1 parameter.

    Returns:
        QuantumCircuit: Circuit that can generate all states in the subspace.
    """
    # Create a 4-qubit circuit with one parameter
    varform = QuantumCircuit(4)
    theta = Parameter('theta')
    
    # Prepare |0101⟩ state (reverse order, so qubits 1 and 3)
    varform.x([1, 3])
    
    # Apply parameterized rotations to create superposition between |0101⟩ and |1010⟩
    varform.ry(theta, 0)
    varform.ry(theta, 1)
    varform.ry(-theta, 2)
    varform.ry(-theta, 3)
    
    # Add CNOT gates for entanglement
    varform.cx(0, 1)
    varform.cx(2, 3)
    
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
