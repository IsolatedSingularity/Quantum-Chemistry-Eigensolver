from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from quantum_chemistry.pauli import Operator, PauliString


def creation_annihilation_operators_with_jordan_wigner(num_states: int) -> Tuple[List[Operator], List[Operator]]:
    """
    Builds the annihilation operators as sum of two Pauli Strings for given number of fermionic states using the Jordan Wigner mapping.

    Args:
        num_states (int): Number of fermionic states.

    Returns:
        Tuple[List[Operator], List[Operator]]: The creation and annihilation operators
    """

    creation_operators = []
    annihilation_operators = []

    for p in range(num_states):
        # Create Z string with Z operators for all qubits before position p
        z_bits = np.zeros(num_states, dtype=bool)
        for i in range(p):
            z_bits[i] = True
            
        # Create X and Y operators at position p
        x_bits_for_x_term = np.zeros(num_states, dtype=bool)
        x_bits_for_x_term[p] = True
        
        x_bits_for_y_term = np.zeros(num_states, dtype=bool)
        x_bits_for_y_term[p] = True
        
        z_bits_for_y_term = np.copy(z_bits)
        z_bits_for_y_term[p] = True
        
        # Creation operator: a†_p = (X_p - iY_p)Z_{p-1}...Z_0 / 2
        x_term = 0.5 * PauliString(z_bits, x_bits_for_x_term)
        y_term = -0.5j * PauliString(z_bits_for_y_term, x_bits_for_y_term)
        creation_op = x_term + y_term
        creation_operators.append(creation_op)
        
        # Annihilation operator: a_p = (X_p + iY_p)Z_{p-1}...Z_0 / 2
        x_term = 0.5 * PauliString(z_bits, x_bits_for_x_term)
        y_term = 0.5j * PauliString(z_bits_for_y_term, x_bits_for_y_term)
        annihilation_op = x_term + y_term
        annihilation_operators.append(annihilation_op)

    return creation_operators, annihilation_operators


def build_one_body_qubit_hamiltonian(
    one_body: NDArray[np.complex128],
    creation_operators: List[Operator],
    annihilation_operators: List[Operator],
) -> Operator:
    """
    Convert a one body fermionic Hamiltonian (square matrix) into a qubit Hamiltonian (Operator) given the annihilation and creation operators (Operator)

    Args:
        one_body (NDArray[np.complex128]): The matrix for the one body Hamiltonian
        creation_operators (List[Operator]): List of creation operators
        annihilation_operators (List[Operator]): List of annihilation operators

    Returns:
        Operator: The one body Hamiltonian as a sum of Pauli strings
    """
    # Create empty arrays for initial Operator
    empty_coefs = np.array([], dtype=complex)
    empty_paulis = np.array([], dtype=object)  # Use object dtype for PauliString array
    hamiltonian = Operator(empty_coefs, empty_paulis)
    
    # Get number of operators
    n = len(creation_operators)
    
    # Create Hamiltonian term by term
    for i in range(n):
        for j in range(n):
            val = one_body[i, j]
            # Skip near-zero terms
            if abs(val) > 1e-10:
                # Multiply creation_i and annihilation_j
                term = creation_operators[i] * annihilation_operators[j]
                # Convert to native Python complex for correct __rmul__ dispatch
                coef = val.item()
                term = coef * term
                # Add to Hamiltonian
                if len(hamiltonian.coefs) == 0:
                    hamiltonian = term
                else:
                    hamiltonian = hamiltonian + term
    
    return hamiltonian


def build_two_body_qubit_hamiltonian(
    two_body: NDArray[np.complex128],
    creation_operators: List[Operator],
    annihilation_operators: List[Operator],
) -> Operator:
    """
    Convert a two body fermionic Hamiltonian (four dimensions square array) into a qubit Hamiltonian (Operator) given the annihilation and creation operators (Operator)

    Args:
        two_body (NDArray[np.complex128]): The array for the two body Hamiltonian
        creation_operators (List[Operator]): List of creation operators
        annihilation_operators (List[Operator]): List of annihilation operators

    Returns:
        Operator: The two body Hamiltonian as a sum of Pauli strings
    """
    # Create empty arrays for initial Operator
    empty_coefs = np.array([], dtype=complex)
    empty_paulis = np.array([], dtype=object)  # Use object dtype for PauliString array
    hamiltonian = Operator(empty_coefs, empty_paulis)
    
    # Get number of operators
    n = len(creation_operators)
    
    # Create Hamiltonian term by term
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    value = two_body[p, q, r, s]
                    # Skip near-zero terms
                    if abs(value) > 1e-10:
                        # Convert to native Python complex and include 1/2 factor
                        coef = (0.5 * value).item()
                        # Compute operator product
                        term = creation_operators[p] * creation_operators[q] * annihilation_operators[r] * annihilation_operators[s]
                        term = coef * term
                        
                        # Add to Hamiltonian
                        if len(hamiltonian.coefs) == 0:
                            hamiltonian = term
                        else:
                            hamiltonian = hamiltonian + term
    
    return hamiltonian


def build_qubit_hamiltonian(
    one_body: NDArray[np.complex128],
    two_body: NDArray[np.complex128],
    creation_operators: List[Operator],
    annihilation_operators: List[Operator],
) -> Operator:
    """
    Build a qubit Hamiltonian from the one body and two body fermionic Hamiltonians.

    Args:
        one_body (NDArray[np.complex128]): The matrix for the one body Hamiltonian
        two_body (NDArray[np.complex128]): The array for the two body Hamiltonian
        creation_operators (List[Operator]): List of creation operators
        annihilation_operators (List[Operator]): List of annihilation operators

    Returns:
        Operator: The total Hamiltonian as a sum of Pauli strings
    """
    # Build the one-body and two-body Hamiltonians separately
    one_body_hamiltonian = build_one_body_qubit_hamiltonian(one_body, creation_operators, annihilation_operators)
    two_body_hamiltonian = build_two_body_qubit_hamiltonian(two_body, creation_operators, annihilation_operators)
    
    # Combine them and simplify
    if len(one_body_hamiltonian.coefs) == 0:
        total_hamiltonian = two_body_hamiltonian
    elif len(two_body_hamiltonian.coefs) == 0:
        total_hamiltonian = one_body_hamiltonian
    else:
        total_hamiltonian = one_body_hamiltonian + two_body_hamiltonian
    
    # Simplify the Hamiltonian by combining like terms and applying a threshold
    total_hamiltonian = total_hamiltonian.combine().apply_threshold().sort()
    
    return total_hamiltonian
