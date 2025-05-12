from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

from quantum_chemistry.pauli import Operator, PauliString


def bitstring_to_bits(bit_string: str) -> NDArray[np.bool_]:
    """
    Convert a bitstring (str) into a numpy.ndarray of bools

    Args:
        bit_string (str): String of '0' and '1'. Little endian assumed.

    Returns:
        NDArray[np.bool_]: Array of bits as bools
    """

    bits = np.array([int(bitchar) for bitchar in reversed(bit_string)], dtype=np.bool_)

    return bits


def diagonal_pauli_with_circuit(pauli_string: PauliString) -> Tuple[PauliString, QuantumCircuit]:
    """
    Builds the QuantumCircuit that transforms the input PauliString into a diagonal one, all Z and I.

    Args:
        pauli_string (PauliString): The Pauli String to be diagonalized

    Returns:
        Tuple[PauliString, QuantumCircuit]: A tuple containing the Diagonal Pauli String and the transformation circuit.
    """
    # Create a circuit for the transformation
    n = len(pauli_string)
    diag_circuit = QuantumCircuit(n)
    
    # Create a diagonal Pauli string (all Z or I)
    diagonal_z_bits = np.logical_or(pauli_string.z_bits, pauli_string.x_bits)
    diagonal_x_bits = np.zeros_like(diagonal_z_bits, dtype=bool)
    
    # Apply rotations based on the original Pauli operators
    for i in range(n):
        # X operator: needs Hadamard to convert to Z-basis
        if pauli_string.x_bits[i] and not pauli_string.z_bits[i]:
            diag_circuit.h(i)
        # Y operator: needs S† then Hadamard to convert to Z-basis
        elif pauli_string.x_bits[i] and pauli_string.z_bits[i]:
            diag_circuit.sdg(i)
            diag_circuit.h(i)
    
    return PauliString(diagonal_z_bits, diagonal_x_bits), diag_circuit


def diagonal_pauli_eigenvalue(pauli: PauliString, bits: NDArray[np.bool_]) -> float:
    """
    Computes the eigenvalue of a bitstring for a given Pauli string

    Args:
        pauli (PauliString): A diagonal pauli string
        bits (NDArray[np.bool_]): Basis state bitstring (ex : '1100')

    Returns:
        float: The eigenvalue corresponding to eigenvector `bits`
    """    
    assert np.all(pauli.x_bits == 0)
    
    # For diagonal Pauli strings (only Z and I), the eigenvalue is determined by 
    # counting the parity of '1' bits that match with Z positions
    # If even number of matching 1s: eigenvalue = +1
    # If odd number of matching 1s: eigenvalue = -1
    
    # Logical AND gives positions where both pauli.z_bits and bits are 1
    matches = np.logical_and(pauli.z_bits, bits)
    
    # Count the number of matches
    num_matches = np.sum(matches)
    
    # Determine eigenvalue based on parity
    eigenvalue = 1 if num_matches % 2 == 0 else -1
    
    return eigenvalue


def diagonal_pauli_expectation_value(pauli: PauliString, counts: dict) -> float:
    """
    Computes the expectation value of a digaonal Pauli string based on counts.

    Args:
        pauli (PauliString): A diagonal Pauli string
        counts (dict): Keys : Basis state bitstring (ex : '1100'),
                       Values : Number of times this state was obtained

    Returns:
        float: The expectation value
    """

    assert np.all(~pauli.x_bits)  # is diagonal
    
    # Calculate the total number of measurements
    total_counts = sum(counts.values())
    
    # Initialize the expectation value
    expectation_value = 0.0
    
    # Iterate through all measured bit strings
    for bit_string, count in counts.items():
        # Convert the bitstring to bits array
        bits = bitstring_to_bits(bit_string)
        
        # Calculate the eigenvalue for this bitstring
        eigenvalue = diagonal_pauli_eigenvalue(pauli, bits)
        
        # Add the contribution to the expectation value
        expectation_value += (count / total_counts) * eigenvalue
    
    return expectation_value


def prepare_estimation_circuits_and_diagonal_paulis(
    paulis: List[PauliString], state_circuit: QuantumCircuit
) -> Tuple[List[QuantumCircuit], List[PauliString]]:
    """
    Assemble the quantum circuit to be executed to compute the expectation values of all the Pauli string in paulis. Also returns the diagonal Paulis required to compute the expectation values.

    Args:
        paulis (List[PauliString]): An ensemble on Pauli string
        state_circuit (QuantumCircuit): A quantum circuit which prepare a quantum state

    Returns:
        List[QuantumCircuit]: The quantum circuits which allow to compute the expectation values of the Paulis
        List[PauliString]: The diagonal Paulis required to compute the expectation values
    """
    diagonal_paulis = []
    estimation_circuits = []
    
    for pauli in paulis:
        # Get the diagonal Pauli and the diagonalization circuit
        diagonal_pauli, diag_circuit = diagonal_pauli_with_circuit(pauli)
        diagonal_paulis.append(diagonal_pauli)
        
        # Create a copy of the state preparation circuit
        full_circuit = state_circuit.copy()
        
        # Append the diagonalization circuit
        full_circuit.compose(diag_circuit, inplace=True)
        
        # Add measurement for all qubits
        full_circuit.measure_all(add_bits=True)
        
        estimation_circuits.append(full_circuit)
    
    return estimation_circuits, diagonal_paulis


def estimate_paulis_expectation_values(
    paulis: List[PauliString], state_circuit: QuantumCircuit, backend: Backend
) -> NDArray[np.float64]:
    """
    Estimates the expectation values for an ensemble of Pauli strings (paulis) for a given quantum state (state_circuit) using a given backend.

    Args:
        paulis (List[PauliString]): An ensemble on Pauli string
        state_circuit (QuantumCircuit): A quantum circuit which prepare a quantum state
        backend (Backend): The backend on which the circuits will be executed

    Returns:
        NDArray[np.float64]: The estimated expectation values
    """
    # Prepare the circuits and get the diagonal Paulis
    estimation_circuits, diagonal_paulis = prepare_estimation_circuits_and_diagonal_paulis(paulis, state_circuit)
    
    # Set up the sampler and execute the circuits
    sampler = Sampler(mode=backend)
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuits = pass_manager.run(estimation_circuits)
    
    # Run the circuits
    job = sampler.run(isa_circuits)
    results = job.result()
    
    # Calculate expectation values from the results
    expectation_values = np.zeros(len(paulis))
    
    for i, diagonal_pauli in enumerate(diagonal_paulis):
        # Get the counts for the current circuit
        counts = results[i].data.meas.get_counts()
        # Calculate the expectation value
        expectation_values[i] = diagonal_pauli_expectation_value(diagonal_pauli, counts)
    
    return expectation_values


def estimate_observable_expectation_value(
    observable: Operator, state_circuit: QuantumCircuit, backend: Backend
) -> float:
    """Estimates the expectation values of an operator for a given quantum state

    Args:
        observable (Operator): Operator from which to take the expectation
        state_circuit (QuantumCircuit): Circuit to prepare the state in which to take the expectation
        backend (Backend): The backend on which the circuits will be executed

    Returns:
        float: The estimated expectation value
    """
    # Extract Pauli strings and coefficients from the observable
    paulis = observable.paulis
    coefs = observable.coefs
    
    # Estimate expectation values for all Pauli strings
    expectation_values = estimate_paulis_expectation_values(paulis, state_circuit, backend)
    
    # Compute the weighted sum using the coefficients
    observable_expectation = np.sum(coefs * expectation_values)
    
    return observable_expectation
