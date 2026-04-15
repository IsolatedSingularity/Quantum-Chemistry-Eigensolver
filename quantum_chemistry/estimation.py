from typing import Dict, List, Tuple

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


def qubitwise_commutes(p1: PauliString, p2: PauliString) -> bool:
    """Check whether two Pauli strings commute qubitwise (QWC).

    Two Pauli strings are QWC if, at every qubit position, their single-qubit
    Pauli operators commute. Since any single-qubit Pauli commutes with itself
    and with the identity, the only conflicts are between distinct non-identity
    Paulis (e.g. X vs Z on the same qubit).

    Args:
        p1 (PauliString): First Pauli string.
        p2 (PauliString): Second Pauli string.

    Returns:
        bool: True if p1 and p2 qubitwise commute.
    """
    for i in range(len(p1)):
        op1 = 2 * int(p1.x_bits[i]) + int(p1.z_bits[i])
        op2 = 2 * int(p2.x_bits[i]) + int(p2.z_bits[i])
        if op1 != 0 and op2 != 0 and op1 != op2:
            return False
    return True


def group_paulis_qwc(paulis: List[PauliString]) -> List[List[int]]:
    """Group Pauli strings into qubitwise-commuting (QWC) sets.

    Uses a greedy graph-coloring approach: iterate through the Pauli strings
    and assign each one to the first existing group where it qubitwise commutes
    with every member, or create a new group otherwise.

    Args:
        paulis (List[PauliString]): Pauli strings to group.

    Returns:
        List[List[int]]: Each inner list contains the indices into *paulis*
            that belong to the same QWC group.
    """
    groups: List[List[int]] = []
    for idx, pauli in enumerate(paulis):
        placed = False
        for group in groups:
            if all(qubitwise_commutes(pauli, paulis[g]) for g in group):
                group.append(idx)
                placed = True
                break
        if not placed:
            groups.append([idx])
    return groups


def qwc_measurement_basis(group_paulis: List[PauliString]) -> Tuple[PauliString, QuantumCircuit]:
    """Build a single measurement circuit for a QWC group.

    For each qubit, the measurement basis is determined by the non-identity
    Pauli present in any member of the group (all members agree because they
    are QWC).

    Args:
        group_paulis (List[PauliString]): QWC-compatible Pauli strings.

    Returns:
        Tuple containing the diagonal representation and the basis-rotation circuit.
    """
    n = len(group_paulis[0])
    basis_z = np.zeros(n, dtype=bool)
    basis_x = np.zeros(n, dtype=bool)

    for pauli in group_paulis:
        for i in range(n):
            op = 2 * int(pauli.x_bits[i]) + int(pauli.z_bits[i])
            if op != 0:
                basis_z[i] = pauli.z_bits[i]
                basis_x[i] = pauli.x_bits[i]

    combined = PauliString(basis_z, basis_x)
    _, diag_circuit = diagonal_pauli_with_circuit(combined)
    return combined, diag_circuit


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
    Estimates the expectation values for an ensemble of Pauli strings using
    qubitwise-commuting (QWC) grouping to reduce the number of circuits.

    Pauli strings that share a common measurement basis are grouped and
    measured with a single circuit. Each group's counts are reused to
    compute the expectation value of every member in that group.

    Args:
        paulis (List[PauliString]): An ensemble of Pauli strings
        state_circuit (QuantumCircuit): A quantum circuit which prepares a quantum state
        backend (Backend): The backend on which the circuits will be executed

    Returns:
        NDArray[np.float64]: The estimated expectation values
    """
    pauli_list = list(paulis)

    # Group Pauli strings by QWC compatibility
    groups = group_paulis_qwc(pauli_list)

    # Build one circuit per group
    group_circuits: List[QuantumCircuit] = []
    for group in groups:
        group_members = [pauli_list[i] for i in group]
        _, diag_circuit = qwc_measurement_basis(group_members)

        full_circuit = state_circuit.copy()
        full_circuit.compose(diag_circuit, inplace=True)
        full_circuit.measure_all()
        group_circuits.append(full_circuit)

    # Transpile and execute all group circuits at once
    sampler = Sampler(mode=backend)
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuits = pass_manager.run(group_circuits)

    job = sampler.run(isa_circuits)
    results = job.result()

    # Extract per-Pauli expectation values from the shared counts
    expectation_values = np.zeros(len(pauli_list))

    for g_idx, group in enumerate(groups):
        counts = results[g_idx].data.meas.get_counts()
        for pauli_idx in group:
            diag_pauli, _ = diagonal_pauli_with_circuit(pauli_list[pauli_idx])
            expectation_values[pauli_idx] = diagonal_pauli_expectation_value(
                diag_pauli, counts,
            )

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
