from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from quantum_chemistry.molecule.linalg import simultaneous_eig


class HFResult(object):
    def __init__(
        self,
        success,
        energy,
        occupations,
        eig_energies,
        eig_states,
        density_matrix,
        number_of_iterations,
        molecular_symmetries,
    ):

        self.success = success
        self.energy = energy
        self.occupations = np.array(occupations)
        self.eig_energies = eig_energies
        self.eig_states = eig_states
        self.transformation = eig_states
        self.density_matrix = density_matrix
        self.number_of_iterations = number_of_iterations
        self.molecular_symmetries = molecular_symmetries

    def __str__(self):

        out = f"Success : {self.success}\n"
        out += f"Iterations : {self.number_of_iterations}\n"
        out += f"Final energy : {self.energy}\n"
        out += f"Occupation vector : " + str(self.occupations.astype(int)) + "\n"
        out += f"State energies : " + str(self.eig_energies) + "\n"

        return out


class HartreeFockSolver(object):
    def __init__(self, convergeance_criteria: float = 1e-12, maximum_iterations: int = 100, callback: Callable = None):

        self.convergeance_criteria = convergeance_criteria
        self.maximum_iterations = maximum_iterations
        self.callback = callback

        self.last_total_energy = 0
        self.last_density_matrix = 0
        self.last_eig_energies = 0

        self.result = None

    def update_options(self, **kwargs):

        possible_options = ["convergeance_criteria", "maximum_iterations", "callback"]

        for option in possible_options:
            if option in kwargs:
                setattr(self, option, kwargs[option])

    def solve(self, one_body: NDArray, two_body: NDArray, num_electrons: int, ao_symmetries: list = []) -> HFResult:

        success = False

        num_orbitals = one_body.shape[0]
        density_matrix = self.density_matrix(np.eye(num_orbitals), num_electrons)

        for i in range(self.maximum_iterations):

            eff_one_body = self.effective_one_body(one_body, two_body, density_matrix)

            eig_energies, eig_states = np.linalg.eigh(eff_one_body)
            eig_energies, eig_states = self.sort_eig_energies(eig_energies, eig_states)

            total_energy = self.total_energy(eig_energies, num_electrons)
            density_matrix = self.density_matrix(eig_states, num_electrons)

            if self.callback:
                self.callback(i, total_energy)

            if i > 0:
                if self.test_convergeance(total_energy, eig_energies):
                    success = True
                    break

            self.last_total_energy = total_energy + 0
            self.last_density_matrix = density_matrix + 0
            self.last_eig_energies = eig_energies + 0

        mo_symmetries = list()
        if ao_symmetries:
            eig_values, eig_states = simultaneous_eig(
                [
                    eff_one_body,
                ]
                + ao_symmetries
            )
            eig_energies = eig_values[0, :]
            eig_energies, eig_states = self.sort_eig_energies(eig_energies, eig_states)

            total_energy = self.total_energy(eig_energies, num_electrons)
            density_matrix = self.density_matrix(eig_states, num_electrons)

            for sym_mat in ao_symmetries:
                mo_symmetries.append(np.einsum("mn,mi,nj->ij", sym_mat, np.conj(eig_states), eig_states))

        eig_states = self.straigten_eig_states(eig_states)

        if np.all(np.isclose(np.imag(eig_energies), 0)):
            eig_energies = np.real(eig_energies)

        if np.all(np.isclose(np.imag(eig_states), 0)):
            eig_states = np.real(eig_states)

        if np.all(np.isclose(np.imag(density_matrix), 0)):
            density_matrix = np.real(density_matrix)

        result = HFResult(
            success,
            total_energy,
            self.starting_occupation_vector(num_orbitals, num_electrons),
            eig_energies,
            eig_states,
            density_matrix,
            i,
            mo_symmetries,
        )

        self.result = result

        return result

    def test_convergeance(self, total_energy: float, eig_energies: NDArray) -> bool:

        energy_crit = np.abs(total_energy - self.last_total_energy) < self.convergeance_criteria
        density_crit = np.all(np.abs(eig_energies - self.last_eig_energies) < self.convergeance_criteria)

        return energy_crit and density_crit

    def effective_one_body(self, one_body: NDArray, two_body: NDArray, density_matrix: NDArray) -> NDArray:

        raise NotImplementedError()

    @staticmethod
    def starting_occupation_vector(num_orbitals: int, num_electrons: int) -> NDArray:

        raise NotImplementedError()

    @classmethod
    def total_energy(cls, eig_energies: NDArray, num_electrons: int) -> float:

        num_orbitals = len(eig_energies)
        starting_occupation_vector = cls.starting_occupation_vector(num_orbitals, num_electrons)

        return np.dot(eig_energies, starting_occupation_vector)

    @classmethod
    def density_matrix(cls, eig_states: NDArray, num_electrons: int) -> NDArray:

        num_orbitals = eig_states.shape[0]
        starting_density_matrix = np.diag(cls.starting_occupation_vector(num_orbitals, num_electrons))

        density_matrix = np.einsum(
            "mn,im,jn->ji", starting_density_matrix, np.conj(eig_states), np.conj(np.conj(eig_states))
        )
        return density_matrix

    @staticmethod
    def sort_eig_energies(eig_energies: NDArray, eig_vectors: NDArray) -> Tuple[NDArray, NDArray]:

        eig_order = np.argsort(eig_energies)
        eig_energies = eig_energies[eig_order]
        eig_vectors = eig_vectors[:, eig_order]

        return eig_energies, eig_vectors

    @staticmethod
    def straigten_eig_states(eig_states: NDArray) -> NDArray:

        max_rows = np.argmax(np.abs(eig_states), axis=0)
        max_cols = np.arange(len(max_rows))

        if np.iscomplexobj(eig_states):
            eig_states_angles = np.angle(eig_states[(max_rows, max_cols)])
            eig_states *= np.exp(-1j * eig_states_angles)
        else:
            eig_states_sign = np.sign(eig_states[(max_rows, max_cols)])
            eig_states *= eig_states_sign

        return eig_states


class RHFSolver(HartreeFockSolver):
    def __init__(self, *vargs, **kwargs):
        super().__init__(*vargs, **kwargs)

    def effective_one_body(self, one_body: NDArray, two_body: NDArray, density_matrix: NDArray) -> NDArray:

        mod_two_body = 2 * two_body - np.einsum("ijkl->ijlk", two_body)
        one_body_corr = np.einsum("ijkl,jk->il", mod_two_body, density_matrix)

        return 2 * one_body + one_body_corr

    @staticmethod
    def starting_occupation_vector(num_orbitals: int, num_electrons: int) -> NDArray:

        assert num_electrons % 2 == 0

        starting_occupation_vector = np.zeros((num_orbitals,))
        starting_occupation_vector[: num_electrons // 2] = 1

        return starting_occupation_vector


class UHFSolver(HartreeFockSolver):
    def __init__(self, *vargs, **kwargs):
        super().__init__(*vargs, **kwargs)

    def effective_one_body(self, one_body: NDArray, two_body: NDArray, density_matrix: NDArray) -> NDArray:

        mod_two_body = two_body - np.einsum("ijkl->jikl", two_body)
        one_body_corr = 0.5 * np.einsum("ijkl,jk->il", mod_two_body, density_matrix)

        return one_body + one_body_corr

    @staticmethod
    def starting_occupation_vector(num_orbitals: int, num_electrons: int) -> NDArray:

        starting_occupation_vector = np.zeros((num_orbitals,))
        starting_occupation_vector[:num_electrons] = 1

        return starting_occupation_vector
