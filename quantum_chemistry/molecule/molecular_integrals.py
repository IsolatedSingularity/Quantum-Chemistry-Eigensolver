from typing import List, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from pyscf import gto

from quantum_chemistry.molecule.hartree_fock import RHFSolver

SPIN_ORDER_CHOICE = Literal["group spin", "group orbital"]


def get_molecular_spin_orbital_integrals(molecule: gto.Mole, symmetries_ao=[]):

    # one-electron atomic integrals
    one_body_kin_ao = molecule.intor("int1e_kin")
    one_body_nuc_ao = molecule.intor("int1e_nuc")
    one_body_ao = one_body_kin_ao + one_body_nuc_ao

    # two-electron atomic integrals
    two_body_ao_chemist = molecule.intor("int2e")
    two_body_ao = np.einsum("ijkl->iklj", two_body_ao_chemist)

    # one-electron atomic overlap
    ovlp_ao = molecule.intor("int1e_ovlp")

    num_electrons = molecule.tot_electrons()

    one_body_pao, two_body_pao, ovlp_pao = from_ao_to_pao(one_body_ao, two_body_ao, ovlp_ao)

    one_body_mo, two_body_mo, ovlp_mo, num_electrons, symmetries_mo = from_pao_to_mo(
        one_body_pao, two_body_pao, ovlp_pao, num_electrons, symmetries_ao
    )

    one_body_so, two_body_so, ovlp_so, symmetries_so = to_so(one_body_mo, two_body_mo, ovlp_mo, symmetries_mo)

    return one_body_so, two_body_so, ovlp_so, symmetries_so


def assemble_atoms(atom_labels: List[str], positions):
    atoms = list()
    for label, position in zip(atom_labels, positions):
        atoms.append((label,) + tuple(position))

    return atoms


def ao2pao_from_ovlp(ovlp_ao):
    _, eigvectors = np.linalg.eigh(ovlp_ao)

    ovlp_eigvalues = np.einsum("mn,mi,nj->ij", ovlp_ao, np.conj(eigvectors), eigvectors).diagonal()
    ao2pao = np.einsum("im,jm,m->ij", eigvectors, np.conj(eigvectors), 1 / np.sqrt(ovlp_eigvalues))

    return ao2pao


def pao2mo_from_hf(one_body_pao, two_body_pao, num_electrons: int, symmetries: List[NDArray] = []):

    hf_solver = RHFSolver()
    hf_result = hf_solver.solve(one_body_pao, two_body_pao, num_electrons, symmetries)

    return hf_result.transformation


def transform_one_body(matrix, transformation):

    tmp_t = transformation
    tmp_c = np.conj(transformation)

    return np.einsum("mn,mi,nj->ij", matrix, tmp_c, tmp_t)


def transform_two_body(matrix, transformation):
    tmp_t = transformation
    tmp_c = np.conj(transformation)

    return np.einsum("mnop,mi,nj,ok,pl->ijkl", matrix, tmp_c, tmp_c, tmp_t, tmp_t)


def from_ao_to_pao(one_body_ao, two_body_ao, ovlp_ao):

    ao2pao = ao2pao_from_ovlp(ovlp_ao)

    one_body_pao = transform_one_body(one_body_ao, ao2pao)
    two_body_pao = transform_two_body(two_body_ao, ao2pao)
    ovlp_pao = transform_one_body(ovlp_ao, ao2pao)

    return one_body_pao, two_body_pao, ovlp_pao


def from_pao_to_mo(one_body_pao, two_body_pao, ovlp_pao, num_electrons, symmetries_pao):

    pao2mo = pao2mo_from_hf(one_body_pao, two_body_pao, num_electrons, symmetries_pao)

    one_body_mo = transform_one_body(one_body_pao, pao2mo)
    two_body_mo = transform_two_body(two_body_pao, pao2mo)
    ovlp_mo = transform_one_body(ovlp_pao, pao2mo)
    symmetries_mo = [transform_one_body(sym, pao2mo) for sym in symmetries_pao]

    return one_body_mo, two_body_mo, ovlp_mo, num_electrons, symmetries_mo


def to_so(one_body, two_body, ovlp, symmetries, orbital_order: SPIN_ORDER_CHOICE = "group spin"):

    one_body_so = tensor_spin_one_body(one_body, orbital_order)
    two_body_so = tensor_spin_two_body(two_body, orbital_order)
    ovlp_so = tensor_spin_one_body(ovlp, orbital_order)
    symmetries_so = [tensor_spin_one_body(sym, orbital_order) for sym in symmetries]

    return one_body_so, two_body_so, ovlp_so, symmetries_so


def tensor_spin_one_body(matrix, orbital_order: SPIN_ORDER_CHOICE = "group spin"):
    spin_tensor = np.eye(2)
    return _tensor_spin(matrix, spin_tensor, orbital_order)


def tensor_spin_two_body(matrix, orbital_order: SPIN_ORDER_CHOICE = "group spin"):
    spin_tensor = np.kron(np.eye(2)[:, None, None, :], np.eye(2)[None, :, :, None])  # Physicist ordering
    return _tensor_spin(matrix, spin_tensor, orbital_order)


def _tensor_spin(matrix, spin_tensor, orbital_order: SPIN_ORDER_CHOICE = "group spin"):
    if orbital_order == "group spin":
        new_matrix = np.kron(spin_tensor, matrix)
    elif orbital_order == "group orbital":
        new_matrix = np.kron(matrix, spin_tensor)
    else:
        raise ValueError("Unknown value for orbital_order")

    return new_matrix
