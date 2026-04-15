"""
VQE Energy Curves Visualization

Creates a static visualization of the VQE energy landscape using real
quantum chemistry data computed from the Jordan-Wigner Hamiltonian and the
number-preserving ansatz.

Left panel: 1D energy landscape (energy vs ansatz parameter theta) for
    several representative bond distances.
Right panel: exact dissociation curve vs VQE optimum at each distance.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from quantum_chemistry.mapping import (  # noqa: E402
    build_qubit_hamiltonian,
    creation_annihilation_operators_with_jordan_wigner,
)
from quantum_chemistry.molecule.h2_molecule import load_h2_spin_orbital_integrals  # noqa: E402
from quantum_chemistry.vqe import h2_ansatz_circuit  # noqa: E402


def _energy_vs_theta(h_matrix, ansatz, thetas):
    """Compute exact expectation value of *h_matrix* for each theta."""
    from qiskit.quantum_info import Statevector

    param = list(ansatz.parameters)[0]
    energies = np.empty(len(thetas))
    for i, t in enumerate(thetas):
        bound = ansatz.assign_parameters({param: t})
        sv = Statevector.from_instruction(bound).data
        energies[i] = np.real(sv.conj() @ h_matrix @ sv)
    return energies


def create_vqe_energy_curves_visualization(save_path):
    """
    Build the VQE energy curves figure from real Hamiltonian data.

    Left panel: energy vs theta for a selection of bond distances
    Right panel: exact ground state energy and VQE minimum across all distances
    """
    print("Creating VQE energy curves visualization...")

    data_path = os.path.join(parent_dir, "h2_data")
    distances, molecule_data = load_h2_spin_orbital_integrals(data_path)
    ansatz = h2_ansatz_circuit()

    thetas = np.linspace(-np.pi, np.pi, 200)

    # Color palettes
    dist_cmap = sns.cubehelix_palette(
        start=2,
        rot=0,
        dark=0.15,
        light=0.85,
        reverse=True,
        as_cmap=True,
    )
    iter_cmap = sns.color_palette("mako", as_cmap=True)

    # Precompute Hamiltonians and energies
    exact_energies = []
    vqe_energies = []
    hamiltonians = []

    for d, (one_body, two_body, nuc) in zip(distances, molecule_data):
        c_ops, a_ops = creation_annihilation_operators_with_jordan_wigner(one_body.shape[0])
        h = build_qubit_hamiltonian(one_body, two_body, c_ops, a_ops)
        h_mat = h.to_matrix()
        hamiltonians.append((h_mat, float(nuc)))

        eig_vals = np.linalg.eigvalsh(h_mat)
        exact_energies.append(float(np.min(eig_vals) + nuc))

        curve = _energy_vs_theta(h_mat, ansatz, thetas) + nuc
        vqe_energies.append(float(np.min(curve)))

    # ── Figure ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )
    fig.suptitle("VQE Energy Landscape (Real Data)", fontsize=18)

    # Left panel: energy vs theta for selected distances
    sample_indices = np.linspace(0, len(distances) - 1, 8, dtype=int)
    norm = plt.Normalize(distances[sample_indices[0]], distances[sample_indices[-1]])

    for idx in sample_indices:
        h_mat, nuc = hamiltonians[idx]
        curve = _energy_vs_theta(h_mat, ansatz, thetas) + nuc
        color = dist_cmap(norm(distances[idx]))
        ax1.plot(thetas, curve, color=color, linewidth=2)

    sm = plt.cm.ScalarMappable(cmap=dist_cmap, norm=norm)
    sm.set_array([])
    cbar1 = fig.colorbar(sm, ax=ax1)
    cbar1.set_label("Bond Distance (Å)", fontsize=12)

    ax1.set_title("Energy vs Ansatz Parameter", fontsize=14)
    ax1.set_xlabel("Parameter θ", fontsize=12)
    ax1.set_ylabel("Energy (Hartree)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Right panel: dissociation curves
    ax2.plot(distances, exact_energies, "k-", linewidth=2, label="Exact (diag)")
    ax2.plot(distances, vqe_energies, "o", color=iter_cmap(0.5), markersize=5, label="VQE minimum")

    min_idx = int(np.argmin(exact_energies))
    ax2.plot(
        distances[min_idx],
        exact_energies[min_idx],
        "*",
        color=iter_cmap(0.85),
        markersize=14,
        markeredgecolor="black",
        label="Equilibrium",
    )
    ax2.axvline(x=distances[min_idx], color="gray", linestyle="--", alpha=0.5)

    ax2.set_title("H₂ Dissociation Curve", fontsize=14)
    ax2.set_xlabel("Bond Distance (Å)", fontsize=12)
    ax2.set_ylabel("Total Energy (Hartree)", fontsize=12)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.15)
    fig.text(
        0.5,
        0.03,
        "Left: energy landscape for 8 bond distances.  Right: exact diagonalization vs single-parameter VQE.",
        fontsize=11,
        ha="center",
        va="center",
        bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.5", edgecolor="gray"),
    )

    print(f"Saving VQE energy curves visualization to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print("VQE energy curves visualization completed")


if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    create_vqe_energy_curves_visualization(os.path.join(output_dir, "vqe_energy_curves.png"))
