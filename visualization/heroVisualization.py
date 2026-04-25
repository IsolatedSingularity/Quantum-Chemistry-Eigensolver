"""
VQE Hero Visualization

Hero plot for the Quantum Chemistry Eigensolver.

Left panel: filled contour of E(theta, R) across all 34 bond distances.
Five warm-start gradient-descent trajectories (different initial parameters)
are traced through the (theta, R) landscape, converging toward the minimum
energy valley as the algorithm finds the ground state.

Right panel: energy error |E - E_0| (log scale) vs VQE iteration at three
bond distances (compressed, equilibrium, stretched), demonstrating convergence
of the parameter-shift gradient descent at different molecular geometries.
"""

import os
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
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


# ── Circuit helpers ────────────────────────────────────────────────────────────
def _energyAt(h_mat, nuc, ansatz, theta):
    """Exact expectation value for a single theta."""
    from qiskit.quantum_info import Statevector

    param = list(ansatz.parameters)[0]
    bound = ansatz.assign_parameters({param: float(theta)})
    sv = Statevector.from_instruction(bound).data
    return float(np.real(sv.conj() @ h_mat @ sv)) + float(nuc)


def _psrGradient(h_mat, nuc, ansatz, theta):
    """Exact parameter-shift rule gradient at theta."""
    shift = np.pi / 2.0
    return (_energyAt(h_mat, nuc, ansatz, theta + shift) - _energyAt(h_mat, nuc, ansatz, theta - shift)) / 2.0


def _energyRow(h_mat, nuc, ansatz, thetas):
    """E(theta) for an array of thetas."""
    from qiskit.quantum_info import Statevector

    param = list(ansatz.parameters)[0]
    out = np.empty(len(thetas))
    for i, t in enumerate(thetas):
        bound = ansatz.assign_parameters({param: float(t)})
        sv = Statevector.from_instruction(bound).data
        out[i] = float(np.real(sv.conj() @ h_mat @ sv))
    return out + float(nuc)


# ── Hero figure ────────────────────────────────────────────────────────────────
def createHeroVisualization(savePath):
    """
    Build the two-panel light-theme hero figure.

    Left:  filled contour of E(theta, R) across all bond distances.
           Five warm-start gradient-descent trajectories from different
           initial parameters converge toward the ground-state valley.
    Right: energy error |E - E_0| (log scale) vs VQE iteration at three
           bond distances (compressed, equilibrium, stretched).
    """
    print("Loading H2 data...")
    data_path = os.path.join(parent_dir, "h2_data")
    distances, molecule_data = load_h2_spin_orbital_integrals(data_path)
    ansatz = h2_ansatz_circuit()

    n_thetas = 160
    thetas = np.linspace(-np.pi, np.pi, n_thetas)

    # ── Precompute Hamiltonians ────────────────────────────────────────────────
    hMats = []
    nucs = []
    for one_body, two_body, nuc in molecule_data:
        c_ops, a_ops = creation_annihilation_operators_with_jordan_wigner(one_body.shape[0])
        h = build_qubit_hamiltonian(one_body, two_body, c_ops, a_ops)
        hMats.append(h.to_matrix())
        nucs.append(float(nuc))

    # ── Energy grid for contour plot ───────────────────────────────────────────
    print("Computing energy grid...")
    energyGrid = np.zeros((len(distances), n_thetas))
    for dIdx, (h_mat, nuc) in enumerate(zip(hMats, nucs)):
        energyGrid[dIdx] = _energyRow(h_mat, nuc, ansatz, thetas)

    # ── Warm-start gradient-descent trajectories ───────────────────────────────
    print("Computing VQE trajectories...")
    initialThetas = np.array([-2.5, -1.2, 0.0, 1.2, 2.5])
    lr = 0.30
    nStepsPerDist = 4
    nStarts = len(initialThetas)
    trajTheta = np.zeros((nStarts, len(distances)))

    for s, theta0 in enumerate(initialThetas):
        t = float(theta0)
        for dIdx, (h_mat, nuc) in enumerate(zip(hMats, nucs)):
            for _ in range(nStepsPerDist):
                g = _psrGradient(h_mat, nuc, ansatz, t)
                t = t - lr * g
                t = float(((t + np.pi) % (2.0 * np.pi)) - np.pi)
            trajTheta[s, dIdx] = t

    # ── Convergence curves at 3 bond distances ─────────────────────────────────
    print("Computing convergence curves...")
    eqDistIdx = int(np.argmin(np.min(energyGrid, axis=1)))
    convIndices = [5, eqDistIdx, len(distances) - 5]
    convEnergy = []
    convGround = []

    for dIdx in convIndices:
        h_mat, nuc = hMats[dIdx], nucs[dIdx]
        ground = float(np.min(energyGrid[dIdx]))
        convGround.append(ground)
        t = -1.8
        steps = [_energyAt(h_mat, nuc, ansatz, t)]
        for _ in range(38):
            g = _psrGradient(h_mat, nuc, ansatz, t)
            t = t - lr * g
            t = float(((t + np.pi) % (2.0 * np.pi)) - np.pi)
            steps.append(_energyAt(h_mat, nuc, ansatz, t))
        convEnergy.append(np.array(steps))

    # ── Color palettes matching vqe_energy_curves.py style ────────────────────
    pesCmap = sns.cubehelix_palette(start=1.5, rot=-0.5, dark=0.05, light=0.95, reverse=False, as_cmap=True)
    trajPalette = sns.color_palette("Set2", nStarts)
    convPalette = sns.color_palette("mako", 3)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.45, 1], wspace=0.30)
    axLeft = fig.add_subplot(gs[0])
    axRight = fig.add_subplot(gs[1])

    fig.suptitle("VQE Potential Energy Surface  \u00b7  H\u2082 Dissociation (34 Bond Distances)", fontsize=15)

    # ── Left: filled contour PES + trajectories ───────────────────────────────
    THETA, DIST = np.meshgrid(thetas, distances)
    cf = axLeft.contourf(THETA, DIST, energyGrid, levels=30, cmap=pesCmap)
    axLeft.contour(THETA, DIST, energyGrid, levels=30, colors="k", alpha=0.10, linewidths=0.4)
    cbar = fig.colorbar(cf, ax=axLeft, fraction=0.035, pad=0.02)
    cbar.set_label("Total Energy (Hartree)", fontsize=11)

    for s in range(nStarts):
        axLeft.plot(
            trajTheta[s],
            distances,
            color=trajPalette[s],
            linewidth=2.2,
            alpha=0.88,
            label=f"$\\theta_0 = {initialThetas[s]:.1f}$",
            zorder=3,
        )
        mid = len(distances) // 2
        axLeft.annotate(
            "",
            xy=(trajTheta[s, mid + 2], distances[mid + 2]),
            xytext=(trajTheta[s, mid - 1], distances[mid - 1]),
            arrowprops=dict(arrowstyle="-|>", color=trajPalette[s], lw=1.6),
            zorder=4,
        )

    axLeft.set_xlabel("Ansatz Parameter \u03b8", fontsize=12)
    axLeft.set_ylabel("Bond Distance (\u00c5)", fontsize=12)
    axLeft.set_title("Warm-Start Gradient Descent on PES", fontsize=13)
    axLeft.legend(fontsize=9, loc="upper right", framealpha=0.85, ncol=2)

    # ── Right: convergence curves (log scale) ─────────────────────────────────
    distLabels = [f"d = {distances[i]:.2f} \u00c5" for i in convIndices]
    for eSteps, gnd, label, c in zip(convEnergy, convGround, distLabels, convPalette):
        err = np.maximum(np.abs(eSteps - gnd), 1e-10)
        axRight.plot(err, color=c, linewidth=2.2, label=label)

    axRight.set_yscale("log")
    axRight.set_xlabel("VQE Iteration", fontsize=12)
    axRight.set_ylabel("|E \u2212 E\u2080| (Hartree)", fontsize=12)
    axRight.set_title("Convergence at Selected Geometries", fontsize=13)
    axRight.legend(fontsize=10)
    axRight.grid(True, alpha=0.3)

    plt.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.10)

    print(f"Saving hero plot to {savePath}...")
    plt.savefig(savePath, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("Hero visualization complete.")


if __name__ == "__main__":
    outputDir = os.path.dirname(os.path.abspath(__file__))
    createHeroVisualization(os.path.join(outputDir, "vqe_hero.png"))
