# %% Imports


import numpy as np
from matplotlib import pyplot as plt
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from quantum_chemistry.mapping import build_qubit_hamiltonian, creation_annihilation_operators_with_jordan_wigner
from quantum_chemistry.molecule.h2_molecule import load_h2_spin_orbital_integrals
from quantum_chemistry.vqe import h2_ansatz_circuit, minimize_expectation_value

# %%


data_path = "../h2_data"

ansatz_circuit = h2_ansatz_circuit()
backend = AerSimulator(shots=10000)


minimizer = lambda fct, start_param_values: minimize(
    fct,
    start_param_values,
    method="SLSQP",
    options={"maxiter": 5, "eps": 1e-1, "ftol": 1e-4, "disp": True, "iprint": 2},
)


distances, molecule_datas = load_h2_spin_orbital_integrals(data_path)


energies = list()
last_param = 0
for distance, molecule_data in zip(distances, molecule_datas):

    one_body, two_body, nuc_eneg = molecule_data

    num_orbs = one_body.shape[0]

    creation_operators, annihilation_operators = creation_annihilation_operators_with_jordan_wigner(num_orbs)

    qubit_hamiltonian = build_qubit_hamiltonian(
        one_body, two_body, creation_operators, annihilation_operators
    ).simplify()

    res = minimize_expectation_value(qubit_hamiltonian, ansatz_circuit, backend, minimizer, [last_param])
    last_param = res.x[0]

    energies.append(res.fun + nuc_eneg)

distances = np.array(distances)
energies = np.array(energies)

# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(distances, energies, ".")

# %%
