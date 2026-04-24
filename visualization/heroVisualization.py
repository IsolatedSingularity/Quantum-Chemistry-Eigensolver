"""
VQE Hero Visualization

Hero plot for the Quantum Chemistry Eigensolver.

Left panel: blurry 2D potential energy landscape over the (theta, bond_distance)
parameter space. Color encodes total energy. The VQE optimization path at each
bond distance is drawn as a bright trail showing how the variational ansatz
finds the ground state.

Right panel: energy vs. ansatz parameter step during the optimization at the
equilibrium bond distance, showing convergence of the variational algorithm.
"""

import os
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from quantum_chemistry.mapping import (  # noqa: E402
    build_qubit_hamiltonian,
    creation_annihilation_operators_with_jordan_wigner,
)
from quantum_chemistry.molecule.h2_molecule import load_h2_spin_orbital_integrals  # noqa: E402
from quantum_chemistry.vqe import h2_ansatz_circuit  # noqa: E402


# ── Tokyo Night Storm palette ──────────────────────────────────────────────────
PALETTE = {
    "bg": "#1a1b26",
    "panel": "#24283b",
    "fg": "#c0caf5",
    "muted": "#a9b1d6",
    "subtle": "#565f89",
    "blue": "#7aa2f7",
    "cyan": "#7dcfff",
    "purple": "#bb9af7",
    "red": "#f7768e",
    "green": "#9ece6a",
    "yellow": "#e0af68",
    "orange": "#ff9e64",
}


def applyTokyoNight() -> None:
    """Apply Tokyo Night Storm rcParams to matplotlib."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["bg"],
            "savefig.facecolor": PALETTE["bg"],
            "axes.edgecolor": PALETTE["subtle"],
            "axes.labelcolor": PALETTE["fg"],
            "axes.titlecolor": PALETTE["fg"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "text.color": PALETTE["fg"],
            "grid.color": PALETTE["subtle"],
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "axes.prop_cycle": cycler(
                color=[PALETTE["blue"], PALETTE["cyan"], PALETTE["purple"], PALETTE["red"]]
            ),
            "legend.facecolor": PALETTE["panel"],
            "legend.edgecolor": PALETTE["subtle"],
            "legend.labelcolor": PALETTE["fg"],
            "font.family": "sans-serif",
            "font.size": 10,
        }
    )



# ── Energy grid computation ────────────────────────────────────────────────────
def computeEnergyGrid(thetas, distances, molecule_data, ansatz):
    """
    Compute total energy E(theta, distance) for every (theta, distance) pair.

    Returns energy grid of shape (len(distances), len(thetas)).
    """
    from qiskit.quantum_info import Statevector

    param = list(ansatz.parameters)[0]
    energyGrid = np.zeros((len(distances), len(thetas)))

    for dIdx, (_, (one_body, two_body, nuc)) in enumerate(zip(distances, molecule_data)):
        c_ops, a_ops = creation_annihilation_operators_with_jordan_wigner(one_body.shape[0])
        h = build_qubit_hamiltonian(one_body, two_body, c_ops, a_ops)
        h_mat = h.to_matrix()

        for tIdx, theta in enumerate(thetas):
            bound = ansatz.assign_parameters({param: theta})
            sv = Statevector.from_instruction(bound).data
            electronic = float(np.real(sv.conj() @ h_mat @ sv))
            energyGrid[dIdx, tIdx] = electronic + float(nuc)

    return energyGrid


def simulateVqeSteps(h_mat, nuc, ansatz, thetaInit=-0.5, nIterSteps=40):
    """
    Simulate VQE optimization via simple gradient descent, recording each step.

    Returns arrays of theta values and energies at each step.
    """
    from qiskit.quantum_info import Statevector

    param = list(ansatz.parameters)[0]

    def energy(theta):
        bound = ansatz.assign_parameters({param: float(theta)})
        sv = Statevector.from_instruction(bound).data
        return float(np.real(sv.conj() @ h_mat @ sv)) + nuc

    thetaPath = [thetaInit]
    energyPath = [energy(thetaInit)]
    stepSize = 0.15
    for _ in range(nIterSteps):
        t = thetaPath[-1]
        grad = (energy(t + 1e-4) - energy(t - 1e-4)) / 2e-4
        tNew = t - stepSize * grad
        thetaPath.append(tNew)
        energyPath.append(energy(tNew))

    return np.array(thetaPath), np.array(energyPath)


# ── Hero figure ────────────────────────────────────────────────────────────────
def createHeroVisualization(savePath):
    """
    Build the two-panel hero figure.

    Left:  blurry 2D PES (bond distance vs theta) with VQE path overlay.
    Right: energy vs VQE step at equilibrium distance.
    """
    print("Loading H2 data...")
    data_path = os.path.join(parent_dir, "h2_data")
    distances, molecule_data = load_h2_spin_orbital_integrals(data_path)
    ansatz = h2_ansatz_circuit()

    thetas = np.linspace(-np.pi, np.pi, 180)

    print("Computing energy grid (this may take a minute)...")
    energyGrid = computeEnergyGrid(thetas, distances, molecule_data, ansatz)

    # Apply Gaussian blur for the blurry landscape aesthetic
    blurredGrid = gaussian_filter(energyGrid, sigma=2.5)

    # Find optimal theta at each distance (VQE path across the landscape)
    optThetaIdx = np.argmin(blurredGrid, axis=1)
    optTheta = thetas[optThetaIdx]
    optEnergy = blurredGrid[np.arange(len(distances)), optThetaIdx]

    # Equilibrium distance = minimum of total energy vs distance
    eqDistIdx = int(np.argmin(optEnergy))
    eqDist = distances[eqDistIdx]
    print(f"Equilibrium distance: {eqDist:.3f} Ang (index {eqDistIdx})")

    # Simulate step-by-step VQE optimization at equilibrium distance
    print("Simulating VQE steps at equilibrium...")
    one_body_eq, two_body_eq, nuc_eq = molecule_data[eqDistIdx]
    c_ops, a_ops = creation_annihilation_operators_with_jordan_wigner(one_body_eq.shape[0])
    h_eq = build_qubit_hamiltonian(one_body_eq, two_body_eq, c_ops, a_ops)
    h_mat_eq = h_eq.to_matrix()
    _, energySteps = simulateVqeSteps(h_mat_eq, float(nuc_eq), ansatz, thetaInit=-1.2, nIterSteps=45)

    # Custom PES colormap: deep blue to purple to yellow
    pesCmap = LinearSegmentedColormap.from_list(
        "pes_hero",
        ["#1a1b26", "#292e42", "#bb9af7", "#7aa2f7", "#7dcfff", "#e0af68"],
        N=512,
    )

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.45, 1], wspace=0.32)

    axLeft = fig.add_subplot(gs[0])
    axRight = fig.add_subplot(gs[1])

    fig.suptitle(
        "VQE Potential Energy Landscape  \u00b7  H\u2082 Dissociation",
        fontsize=16,
        color=PALETTE["fg"],
        y=0.97,
    )

    # ── Left: 2D blurry PES ────────────────────────────────────────────────────
    im = axLeft.imshow(
        blurredGrid,
        aspect="auto",
        origin="lower",
        extent=[thetas[0], thetas[-1], distances[0], distances[-1]],
        cmap=pesCmap,
        interpolation="bilinear",
    )

    # Subtle iso-energy contour lines
    axLeft.contour(
        thetas,
        distances,
        blurredGrid,
        levels=12,
        colors="white",
        alpha=0.18,
        linewidths=0.7,
    )

    # VQE path: optimal theta at each bond distance
    axLeft.plot(
        optTheta,
        distances,
        color=PALETTE["red"],
        linewidth=2.5,
        zorder=4,
        label="VQE path  (optimal \u03b8 per distance)",
    )

    # Mark the equilibrium point
    axLeft.scatter(
        [optTheta[eqDistIdx]],
        [eqDist],
        s=100,
        color=PALETTE["yellow"],
        edgecolors=PALETTE["bg"],
        linewidths=1.5,
        zorder=5,
        label=f"Equilibrium  $R_{{eq}} = {eqDist:.2f}$ \u00c5",
    )

    cbar = fig.colorbar(im, ax=axLeft, fraction=0.046, pad=0.02)
    cbar.set_label("Total Energy (Hartree)", color=PALETTE["fg"], fontsize=11)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["muted"])

    axLeft.set_xlabel("Ansatz Parameter  \u03b8  (rad)", color=PALETTE["fg"], fontsize=12)
    axLeft.set_ylabel("Bond Distance  R  (\u00c5)", color=PALETTE["fg"], fontsize=12)
    axLeft.set_title("Blurred Potential Energy Surface", color=PALETTE["fg"], fontsize=13)
    axLeft.legend(
        loc="upper right",
        fontsize=9,
        facecolor=PALETTE["panel"],
        edgecolor=PALETTE["subtle"],
        labelcolor=PALETTE["fg"],
    )

    # ── Right: energy vs VQE step ──────────────────────────────────────────────
    steps = np.arange(len(energySteps))
    groundTruth = float(np.min(energySteps))

    axRight.plot(steps, energySteps, color=PALETTE["blue"], linewidth=2.5, label="VQE energy")
    axRight.scatter(steps, energySteps, color=PALETTE["cyan"], s=30, zorder=4, alpha=0.8)
    axRight.axhline(
        groundTruth,
        color=PALETTE["yellow"],
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
        label=f"Ground state  {groundTruth:.4f} Ha",
    )

    convergedIdx = np.where(np.abs(energySteps - groundTruth) < 2e-3)[0]
    if len(convergedIdx):
        axRight.axvspan(convergedIdx[0], steps[-1], alpha=0.08, color=PALETTE["green"], label="Converged (<2 mHa)")

    axRight.set_xlabel("VQE Iteration", color=PALETTE["fg"], fontsize=12)
    axRight.set_ylabel("Total Energy (Hartree)", color=PALETTE["fg"], fontsize=12)
    axRight.set_title(
        f"Energy Convergence at $R = {eqDist:.2f}$ \u00c5",
        color=PALETTE["fg"],
        fontsize=13,
    )
    axRight.legend(
        fontsize=9,
        facecolor=PALETTE["panel"],
        edgecolor=PALETTE["subtle"],
        labelcolor=PALETTE["fg"],
    )
    axRight.grid(True)

    print(f"Saving hero plot to {savePath}...")
    plt.savefig(savePath, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("Hero visualization complete.")


if __name__ == "__main__":
    outputDir = os.path.dirname(os.path.abspath(__file__))
    applyTokyoNight()
    createHeroVisualization(os.path.join(outputDir, "vqe_hero.png"))
