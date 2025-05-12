from numbers import Number
from typing import Self, Union

import numpy as np
from numpy.typing import NDArray

PAULI_LABELS = np.array(["I", "Z", "X", "Y"])

PAULI_I = np.array([[1, 0], [0, 1]], dtype=np.complex128)

PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


class PauliString:
    def __init__(self, z_bits: NDArray[np.bool], x_bits: NDArray[np.bool]):
        """
        Describes a Pauli string. It is encoded into 2 arrays of booleans using the equation

            (-1j)**(z_bits*x_bits) Z**z_bits X**x_bits.

        Args:
            z_bits (np.ndarray<bool>): True where a Z Pauli is applied.
            x_bits (np.ndarray<bool>): True where a X Pauli is applied.
        """

        self.z_bits = z_bits
        self.x_bits = x_bits

    def __str__(self):
        """
        String representation of the PauliString.

        Returns:
            str: String of I, Z, X and Y.
        """

        pauli_choices = np.flip(self.z_bits + 2 * self.x_bits)

        return "".join(PAULI_LABELS[pauli_choices])

    def __len__(self):
        """
        Number of Pauli or Identity in the PauliString.
        Also the number of qubits.

        Returns:
            int: Length of the PauliString, also number of qubits.
        """

        return len(self.z_bits)

    def __mul__(self, other: Union[Self, Number]):
        """
        Allow the use of '*' with other PauliString or with a coef (numeric).

        Args:
            other (PauliString): Will compute the composition between two PauliString
            or
            other (float): Will return an Operator

        Returns:
            PauliString, complex: When other is a PauliString
            or
            Operator : When other is numeric
        """

        if isinstance(other, PauliString):
            return self.mul_pauli_string(other)
        else:
            return self.mul_coef(other)

    def __rmul__(self, other: Union[Self, Number]):
        """
        Same as __mul__. Allow the use of '*' with a preceding coef (numeric) Like in 0.5 * PauliString
        """

        return self.__mul__(other)

    def to_zx_bits(self):
        """
        Return the zx_bits representation of the PauliString.

        ! Useful to compare PauliString together.

        Returns:
            np.array<bool>: zx_bits representation of the PauliString of length 2n
        """

        stringBits = np.concatenate([self.z_bits, self.x_bits])

        return stringBits


    def to_xz_bits(self):
        """
        Return the xz_bits representation of the PauliString.

        ! Useful to check commutativity.

        Returns:
            np.array<bool>: xz_bits representation of the PauliString of length 2n
        """
        # Concatenate in x,z order instead of z,x for commutativity checking
        stringBits = np.concatenate([self.x_bits, self.z_bits])
        
        return stringBits

    def ids(self):
        """
        Position of Identity in the PauliString.

        Returns:
            np.array<bool>: True where both z_bits and x_bits are False.
        """
        # Identity operators are positions where neither Z nor X is applied
        return ~(self.z_bits | self.x_bits)

    def mul_pauli_string(self, other: Self):
        """
        Product with an 'other' PauliString.

        Args:
            other (PauliString): An other PauliString.

        Raises:
            ValueError: If the other PauliString is not of the same length.

        Returns:
            PauliString, complex: The resulting PauliString and the product phase.
        """
        if len(self) != len(other):
            raise ValueError("Cannot multiply PauliStrings of different lengths")
        
        # XOR the bits to get the new Pauli operator type at each position
        new_z_bits = np.logical_xor(self.z_bits, other.z_bits)
        new_x_bits = np.logical_xor(self.x_bits, other.x_bits)
        
        # Calculate phase
        # For each qubit where both operators are non-identity,
        # we need to compute the phase contribution
        phase = 1.0
        for i in range(len(self)):
            # Get operator types from the boolean arrays
            self_op = 2*int(self.x_bits[i]) + int(self.z_bits[i])
            other_op = 2*int(other.x_bits[i]) + int(other.z_bits[i])
            
            # Skip if either operator is Identity (0)
            if self_op == 0 or other_op == 0:
                continue
            
            # If same operator, no phase change
            if self_op == other_op:
                continue
                
            # Compute phase for XY, YZ, ZX and their inverses
            if (self_op == 1 and other_op == 2) or (self_op == 2 and other_op == 3) or (self_op == 3 and other_op == 1):
                phase *= 1j  # X*Y=iZ, Y*Z=iX, Z*X=iY
            elif (self_op == 2 and other_op == 1) or (self_op == 3 and other_op == 2) or (self_op == 1 and other_op == 3):
                phase *= -1j  # Y*X=-iZ, Z*Y=-iX, X*Z=-iY
        
        return PauliString(new_z_bits, new_x_bits), phase

    def mul_coef(self, other: Number):
        """
        Build an Operator from a PauliString (self) and a numeric (coef).

        Args:
            coef (int, float or complex): A numeric coefficient.

        Returns:
            Operator: A Operator with only one PauliString and coef.
        """
        # Create a new Operator with just this PauliString and the given coefficient
        return Operator(np.array([other]), np.array([self]))

    def to_matrix(self):
        """
        Build the matrix representation of the PauliString using the Kroenecker product.

        Returns:
            np.array<complex>: A 2**n side square matrix.
        """
        # Start with 1×1 identity matrix
        matrix = np.array([[1]], dtype=np.complex128)
        
        # For each qubit, tensor product with the appropriate Pauli matrix
        for i in range(len(self)):
            if self.x_bits[i] and self.z_bits[i]:  # Y operator
                pauli = PAULI_Y
            elif self.x_bits[i]:  # X operator
                pauli = PAULI_X
            elif self.z_bits[i]:  # Z operator
                pauli = PAULI_Z
            else:  # Identity operator
                pauli = PAULI_I
            
            matrix = np.kron(matrix, pauli)
        
        return matrix

    @classmethod
    def from_str(cls, pauli_str: str):
        """
        Construct a PauliString from a str (as returned by __str__).

        Args:
            pauli_str (str): String of length n made of 'I', 'X', 'Y' and 'Z'.

        Returns:
            PauliString: The Pauli string specified by the 'pauli_str'.
        """
        n = len(pauli_str)
        z_bits = np.zeros(n, dtype=bool)
        x_bits = np.zeros(n, dtype=bool)
        
        # Parse the string representation into bit arrays
        # Note: string is read from most significant bit to least significant
        for i, letter in enumerate(reversed(pauli_str)):
            if letter == 'Z':
                z_bits[i] = True
            elif letter == 'X':
                x_bits[i] = True
            elif letter == 'Y':  # Y = XZ with phase
                z_bits[i] = True
                x_bits[i] = True
            elif letter != 'I':
                raise ValueError(f"Invalid Pauli character: {letter}")
        
        return cls(z_bits, x_bits)


class Operator:
    def __init__(self, coefs: NDArray[np.complex128], paulis: NDArray[PauliString]):
        """
        Describes an Operator as a Linear Combinaison of Pauli Strings.

        Args:
            coefs (np.array): Coefficients multiplying the respective PauliStrings.
            pauli_strings (np.array<PauliString>): PauliStrings.

        Raises:
            ValueError: If the number of coefs is different from the number of PauliStrings.
            ValueError: If all PauliStrings are not of the same length.
        """

        self.coefs = np.array(coefs)
        self.paulis = np.array(paulis)
        assert len(self.coefs) == len(self.paulis)

    def __str__(self):
        """
        String representation of the Operator.

        Returns:
            str: Descriptive string.
        """

        if len(self.coefs) > 6:
            return "\n".join([f"({coef:.2f})*{pauli}" for coef, pauli in zip(self.coefs, self.paulis)])
        else:
            return " + ".join([f"({coef:.2f})*{pauli}" for coef, pauli in zip(self.coefs, self.paulis)])

    def __len__(self) -> int:
        """
        Number of PauliStrings in the Operator.

        Returns:
            int: Number of PauliStrings/coefs.
        """

        return len(self.paulis)

    def __getitem__(self, key):
        """
        Return a subset of the Operator array-like.

        Args:
            key (int or slice): Elements to be returned.

        Returns:
            Operator: Operator with the element specified in key.
        """

        coefs = [self.coefs[key]] if isinstance(key, int) else self.coefs[key]
        paulis = [self.paulis[key]] if isinstance(key, int) else self.paulis[key]
        return Operator(coefs, paulis)

    def __add__(self, other: Self):
        """
        Allow the use of + to add two Operator together.

        Args:
            other (Operator): An other Operator.

        Returns:
            Operator: New Operator of len = len(self) + len(other).
        """

        if isinstance(other, Operator):
            return self.add_operator(other)
        else:
            raise ValueError("Can only add an other Operator")

    def __sub__(self, other: Self):
        """
        Allow the use of - to substract an Operator from Self.

        Args:
            other (Operator): An other Operator.

        Returns:
            Operator: New Operator of len = len(self) + len(other).
        """

        if isinstance(other, Operator):
            return self.add_operator(-1 * other)
        else:
            raise ValueError("Can only add an other Operator")

    def __mul__(self, other: Union[Self, Number]):
        """
        Allow the use of * with other Operator or numeric value(s)

        Args:
            other (Operator): An other Operator

        Returns:
            Operator: New Operator of len = len(self) * len(other)
            or
            Operator: New Operator of same length with modified coefs
        """

        if isinstance(other, Operator):
            return self.mul_operator(other)
        else:
            return self.mul_coef(other)

    def __rmul__(self, other: Number):
        """
        Same as __mul__.
        Allow the use of '*' with a preceding coef (numeric).
        Like in 0.5 * Operator.

        Args:
            other (Operator): An other Operator.

        Returns:
            Operator: New Operator of len = len(self) * len(other)
            or
            Operator: New Operator of same length with modified coefs
        """

        return self.__mul__(other)

    def add_operator(self, other: Self):
        """
        Adding with an other Operator. Merging the coefs and PauliStrings arrays.

        Args:
            other (Operator): An other Operator.

        Raises:
            ValueError: If other is not an Operator.
            ValueError: If the other Operator has not the same number of qubits.

        Returns:
            Operator: New Operator of len = len(self) + len(other).
        """
        # Check if operators can be added (same number of qubits)
        if len(self.paulis) > 0 and len(other.paulis) > 0:
            if len(self.paulis[0]) != len(other.paulis[0]):
                raise ValueError("Cannot add operators with different number of qubits")
        
        # Concatenate coefficients and Pauli strings
        new_coefs = np.concatenate([self.coefs, other.coefs])
        new_pauli_strings = np.concatenate([self.paulis, other.paulis])
        
        return Operator(new_coefs, new_pauli_strings)

    def mul_operator(self, other: Self):
        """
        Multiply (compose) with an other Operator.

        Args:
            other (Operator): An other Operator.

        Raises:
            ValueError: If other is not an Operator.
            ValueError: If the other Operator has not the same number of qubits.

        Returns:
            Operator: New Operator of len = len(self) * len(other).
        """
        # Check if operators can be multiplied (same number of qubits)
        if len(self.paulis) > 0 and len(other.paulis) > 0:
            if len(self.paulis[0]) != len(other.paulis[0]):
                raise ValueError("Cannot multiply operators with different number of qubits")
        
        new_coefs = []
        new_pauli_strings = []
        
        # Distribute multiplication over all pairs of terms
        for coef1, pauli1 in zip(self.coefs, self.paulis):
            for coef2, pauli2 in zip(other.coefs, other.paulis):
                # Multiply the Pauli strings and get the resulting phase
                new_pauli, phase = pauli1.mul_pauli_string(pauli2)
                # Multiply all coefficients including the phase
                new_coef = coef1 * coef2 * phase
                
                new_coefs.append(new_coef)
                new_pauli_strings.append(new_pauli)
        
        return Operator(np.array(new_coefs), np.array(new_pauli_strings))

    def mul_coef(self, other: Number):
        """
        Multiply the Operator by a coef (numeric) or an array of the same length.

        Args:
            other (float, complex or np.array): One numeric factor or one factor per PauliString.

        Raises:
            ValueError: If other is np.array should be of the same length as the Operator.

        Returns:
            Operator: New Operator of len = len(self).
        """
        return Operator(other * self.coefs, self.paulis)

    def to_zx_bits(self):
        """
        Build an array that contains all the zx_bits for each PauliString.

        Returns:
            np.array<bool>: A 2d array of booleans where each line is the zx_bits of a PauliString.
        """
        # Stack all Pauli strings' zx_bits representations
        return np.stack([pauli.to_zx_bits() for pauli in self.paulis])

    def to_xz_bits(self):
        """
        Build an array that contains all the xz_bits for each PauliString.

        Returns:
            np.array<bool>: A 2d array of booleans where each line is the xz_bits of a PauliString.
        """
        # Stack all Pauli strings' xz_bits representations
        return np.stack([pauli.to_xz_bits() for pauli in self.paulis])

    def ids(self):  # remove?
        """
        Build an array that identifies the position of all the I for each PauliString.

        Returns:
            np.array<bool>: A 2d array of booleans where each line identifies the I on a PauliString.
        """

        return np.stack([pauli.ids() for pauli in self.paulis])

    def combine(self):
        """
        Finds unique PauliStrings in the Operator and combines the coefs of identical PauliStrings.
        Reduces the length of the Operator.

        Returns:
            Operator: Operator with combined coefficients.
        """
        if len(self.paulis) == 0:
            return self
        
        # Use zx_bits to find unique Pauli strings
        zx_bits = self.to_zx_bits()
        unique_zx_bits, indices = np.unique(zx_bits, axis=0, return_inverse=True)
        
        # Combine coefficients for identical Pauli strings
        new_coefs = np.zeros(len(unique_zx_bits), dtype=complex)
        for i, idx in enumerate(indices):
            new_coefs[idx] += self.coefs[i]
        
        # Reconstruct Pauli strings from unique zx_bits
        new_pauli_strings = []
        n = len(self.paulis[0])
        for bits in unique_zx_bits:
            z_bits = bits[:n]
            x_bits = bits[n:]
            new_pauli_strings.append(PauliString(z_bits, x_bits))
        
        return Operator(new_coefs, np.array(new_pauli_strings))

    def apply_threshold(self, threshold: float = 1e-9):
        """
        Remove PauliStrings with coefficients smaller than threshold.

        Args:
            threshold (float, optional): PauliStrings with coef smaller than 'threshold' will be removed.
                                         Defaults to 1e-9.

        Returns:
            Operator: Operator without coefficients smaller than threshold.
        """
        # Create mask for terms with significant coefficients
        mask = np.abs(self.coefs) >= threshold
        
        # Keep only terms with coefficients above threshold
        return Operator(self.coefs[mask], self.paulis[mask])

    def simplify(self, threshold: float = 1e-9):
        """
        Simplify the Operator by applying `combine` and `apply_threshold`.

        Args:
            threshold (float, optional): PauliStrings with coef smaller than 'threshold' will be removed.
                                         Defaults to 1e-0.

        Returns:
            Operator: Operator without coefficients smaller than threshold.
        """

        return self.combine().apply_threshold(threshold)

    def sort(self):
        """
        Sort the PauliStrings with largest (absolute) weights first.

        Returns:
            Operator: Sorted.
        """

        idx = np.argsort(-np.abs(self.coefs))
        return Operator(self.coefs[idx], self.paulis[idx])

    def to_matrix(self):
        """
        Build the total matrix representation of the Operator.

        Returns:
            np.array<complex>: A 2**n side square matrix.
        """
        if len(self.paulis) == 0:
            return None
        
        n = len(self.paulis[0])
        matrix_size = 2**n
        result = np.zeros((matrix_size, matrix_size), dtype=complex)
        
        # Sum all the individual Pauli matrices with their coefficients
        for coef, pauli in zip(self.coefs, self.paulis):
            result += coef * pauli.to_matrix()
        
        return result

    def adjoint(self):
        """
        Returns the adjoint of the Operator

        Returns:
            Operator: Adjoint of the Operator
        """

        return Operator(self.coefs.conj(), self.paulis)
