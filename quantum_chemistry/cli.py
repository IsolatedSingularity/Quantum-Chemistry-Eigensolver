"""Command-line entry points for the quantum chemistry eigensolver."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Return the project root (one level above quantum_chemistry/)."""
    return Path(__file__).resolve().parent.parent


def _default_data_path() -> Path:
    return _project_root() / "h2_data"


# ---------------------------------------------------------------------------
# qce-dissociation
# ---------------------------------------------------------------------------

def dissociation() -> None:
    """Compute and plot the H₂ dissociation curve using VQE."""
    parser = argparse.ArgumentParser(
        prog="qce-dissociation",
        description="Run VQE across H₂ bond distances and plot the dissociation curve.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=_default_data_path(),
        help="Directory containing h2_mo_integrals_d_*.npz files.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=10_000,
        help="Number of measurement shots per circuit (default: 10 000).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Maximum SLSQP iterations per distance (default: 5).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save the plot image (default: display interactively).",
    )
    args = parser.parse_args()

    from matplotlib import pyplot as plt
    from qiskit_aer import AerSimulator
    from scipy.optimize import minimize

    from quantum_chemistry.mapping import (
        build_qubit_hamiltonian,
        creation_annihilation_operators_with_jordan_wigner,
    )
    from quantum_chemistry.molecule.h2_molecule import load_h2_spin_orbital_integrals
    from quantum_chemistry.vqe import h2_ansatz_circuit, minimize_expectation_value

    data_path = str(args.data_path)
    distances, molecule_datas = load_h2_spin_orbital_integrals(data_path)

    ansatz_circuit = h2_ansatz_circuit()
    backend = AerSimulator(shots=args.shots)
    max_iter = args.max_iter

    minimizer = lambda fct, start: minimize(
        fct,
        start,
        method="SLSQP",
        options={"maxiter": max_iter, "eps": 1e-1, "ftol": 1e-4, "disp": True, "iprint": 2},
    )

    energies: list[float] = []
    last_param = 0.0
    for distance, molecule_data in zip(distances, molecule_datas):
        one_body, two_body, nuc_eneg = molecule_data
        num_orbs = one_body.shape[0]

        creation_ops, annihilation_ops = creation_annihilation_operators_with_jordan_wigner(num_orbs)
        qubit_hamiltonian = build_qubit_hamiltonian(
            one_body, two_body, creation_ops, annihilation_ops,
        ).simplify()

        res = minimize_expectation_value(
            qubit_hamiltonian, ansatz_circuit, backend, minimizer, [last_param],
        )
        last_param = res.x[0]
        energies.append(res.fun + nuc_eneg)
        print(f"d = {distance:.2f} Å  →  E = {energies[-1]:.6f} Ha")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(distances, energies, "o-")
    ax.set_xlabel("Bond Distance (Å)")
    ax.set_ylabel("Total Energy (Hartree)")
    ax.set_title("H₂ Dissociation Curve (VQE)")
    ax.grid(True, alpha=0.3)

    if args.output:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# qce-visualize
# ---------------------------------------------------------------------------

def visualize() -> None:
    """Generate all static visualizations for the project."""
    parser = argparse.ArgumentParser(
        prog="qce-visualize",
        description="Generate project visualizations (molecular orbitals, VQE curves, dissociation).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=_project_root() / "visualization",
        help="Directory to write images to (default: visualization/).",
    )
    parser.add_argument(
        "--which",
        nargs="*",
        choices=["orbitals", "vqe", "dissociation", "all"],
        default=["all"],
        help="Which visualizations to generate (default: all).",
    )
    args = parser.parse_args()

    output_dir = args.which and args.output_dir or _project_root() / "visualization"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = set(args.which)
    run_all = "all" in targets

    if run_all or "orbitals" in targets:
        from visualization.h2_molecular_orbitals import create_molecular_orbital_visualization
        create_molecular_orbital_visualization(str(output_dir / "h2_molecular_orbitals.png"))

    if run_all or "vqe" in targets:
        from visualization.vqe_energy_curves import create_vqe_energy_curves_visualization
        create_vqe_energy_curves_visualization(str(output_dir / "vqe_energy_curves.png"))

    if run_all or "dissociation" in targets:
        from visualization.masterVisualization import animate_h2_dissociation
        animate_h2_dissociation(str(output_dir / "h2_dissociation.gif"))

    print("Done.")
