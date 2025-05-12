import os

import numpy as np


def molecule_h2(distance_hh: float):

    from pyscf import gto

    from quantum_chemistry.molecule.molecular_integrals import assemble_atoms

    atom_labels = ["H", "H"]
    positions = np.array([[0, 0, -distance_hh / 2], [0, 0, distance_hh / 2]])

    atoms = assemble_atoms(atom_labels, positions)

    pyscf_mole = gto.M(atom=atoms)

    return pyscf_mole


def get_h2_spin_orbital_integrals(distance_hh: float):

    from quantum_chemistry.molecule.molecular_integrals import get_molecular_spin_orbital_integrals

    molecule = molecule_h2(distance_hh)
    symmetries_ao = [
        np.array([[0, 1], [1, 0]]),
    ]

    nuclear_repulsion_energy = molecule.energy_nuc()

    one_body_so, two_body_so, ovlp_so, symmetries_so = get_molecular_spin_orbital_integrals(molecule, symmetries_ao)

    return one_body_so, two_body_so, nuclear_repulsion_energy


def generate_and_save_h2_spin_orbital_integrals(hh_distances, data_path):

    for distance in hh_distances:

        one_body, two_body, nuc_eneg = get_h2_spin_orbital_integrals(distance)

        filename = f"h2_mo_integrals_d_{int(distance*1000):04d}.npz"

        print(f"Generating {filename}")

        file = os.path.join(data_path, filename)

        with open(file, "wb") as f:
            np.save(f, distance)
            np.save(f, one_body)
            np.save(f, two_body)
            np.save(f, nuc_eneg)


def load_h2_spin_orbital_integral(data_path, filename):
    print(f"Loading {filename}")
    file = os.path.join(data_path, filename)

    with open(file, "rb") as f:
        distance = np.load(f)
        one_body = np.load(f)
        two_body = np.load(f)
        nuc_eneg = np.load(f)

    return distance, one_body, two_body, nuc_eneg


def load_h2_spin_orbital_integrals(data_path):

    distances = []
    molecule_datas = []

    for filename in os.listdir(data_path):
        _, ext = os.path.splitext(filename)
        if ext != ".npz":
            print(f"Skipping {filename}")
            continue

        distance, one_body, two_body, nuc_eneg = load_h2_spin_orbital_integral(data_path, filename)

        distances.append(float(distance))
        molecule_datas.append((one_body, two_body, nuc_eneg))

    order = np.argsort(distances)

    distances = [distances[o] for o in order]
    molecule_datas = [molecule_datas[o] for o in order]

    return distances, molecule_datas
