# %%

import numpy as np

from quantum_chemistry.molecule.h2_molecule import (
    generate_and_save_h2_spin_orbital_integrals,
    load_h2_spin_orbital_integrals,
)

hh_distances = np.round(np.arange(0.3, 2, 0.05), 2)
data_path = "../h2_data"

generate_and_save_h2_spin_orbital_integrals(hh_distances, data_path)

# %%

distances, molecule_datas = load_h2_spin_orbital_integrals(data_path)

print(distances)

# %%
