"""
VQE Hero Visualization

Hero plot for the Quantum Chemistry Eigensolver.

Left panel: filled contour of E(theta, R) - E_min(R) across all 34 bond
distances, revealing how the well depth and landscape shape vary with geometry.
Gradient-descent trajectories are shown at 8 sampled bond distances, starting
from theta=pi/2. Convergence is fastest near equilibrium (deep well, large
gradient) and slowest at dissociation (shallow well, small coupling) -- a
physically meaningful non-trivial pattern. Color scheme matches
vqe_energy_curves.py: cubehelix palette for bond-distance trajectory coloring,
mako colormap for the PES background.

Right panel: energy error |E - E_0| (log scale) vs VQE iteration at three
geometries (compressed, equilibrium, stretched). Uses mako palette.
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

    # ── Energy grid ───────────────────────────────────────────────────────────
    print("Computing energy grid...")
    energyGrid = np.zeros((len(distances), n_thetas))
    for dIdx, (h_mat, nuc) in enumerate(zip(hMats, nucs)):
        energyGrid[dIdx] = _energyRow(h_mat, nuc, ansatz, thetas)

    # Relative landscape: E(theta, R) - E_min(R)  highlights how well-depth
    # varies with bond distance (deep at equilibrium, shallow at dissociation)
    relGrid = energyGrid - energyGrid.min(axis=1, keepdims=True)

    # ── VQE trajectories at 8 sampled bond distances ──────────────────────────
    # Each trajectory starts at theta=pi/2 (maximum gradient, clearly away from
    # both the minimum and the saddle). Convergence rate is proportional to the
    # coupling |B(R)|, which peaks at equilibrium and decays toward dissociation,
    # so paths are short near R_eq and long near R_max -- physically revealing.
    print("Computing VQE trajectories...")
    sampleIndices = np.linspace(0, len(distances) - 1, 8, dtype=int)
    lr = 0.06
    nSteps = 28
    trajData = []  # list of (theta_path, R)

    for idx in sampleIndices:
        h_mat, nuc = hMats[idx], nucs[idx]
        t = np.pi / 2.0
        tPath = [t]
        for _ in range(nSteps):
            g = _psrGradient(h_mat, nuc, ansatz, t)
            t = t - lr * g
            t = float(((t + np.pi) % (2.0 * np.pi)) - np.pi)
            tPath.append(t)
        trajData.append((np.array(tPath), distances[idx]))

    # ── Convergence curves at 3 representative geometries ─────────────────────
    print("Computing convergence curves...")
    eqDistIdx = int(np.argmin(energyGrid.min(axis=1)))
    convIndices = [sampleIndices[0], eqDistIdx, sampleIndices[-1]]
    convEnergy = []
    convGround = []

    for dIdx in convIndices:
        h_mat, nuc = hMats[dIdx], nucs[dIdx]
        ground = float(np.min(energyGrid[dIdx]))
        convGround.append(ground)
        t = np.pi / 2.0
        steps = [_energyAt(h_mat, nuc, ansatz, t)]
        for _ in range(38):
            g = _psrGradient(h_mat, nuc, ansatz, t)
            t = t - lr * g
            t = float(((t + np.pi) % (2.0 * np.pi)) - np.pi)
            steps.append(_energyAt(h_mat, nuc, ansatz, t))
        convEnergy.append(np.array(steps))

    # ── Color palettes -- exact match to vqe_energy_curves.py ─────────────────
    distCmap = sns.cubehelix_palette(start=2, rot=0, dark=0.15, light=0.85, reverse=True, as_cmap=True)
    pesCmap = sns.color_palette("mako", as_cmap=True)
    convPalette = sns.color_palette("mako", n_colors=3)
    norm = plt.Normalize(distances[0], distances[-1])

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.45, 1], wspace=0.30)
    axLeft = fig.add_subplot(gs[0])
    axRight = fig.add_subplot(gs[1])

    fig.suptitle("VQE Gradient Descent on H\u2082 Potential Energy Surface", fontsize=15)

    # ── Left: relative PES + per-distance trajectories ────────────────────────
    THETA, DIST = np.meshgrid(thetas, distances)
    cf = axLeft.contourf(THETA, DIST, relGrid, levels=30, cmap=pesCmap)
    axLeft.contour(THETA, DIST, relGrid, levels=30, colors="k", alpha=0.08, linewidths=0.4)
    cbar = fig.colorbar(cf, ax=axLeft, fraction=0.035, pad=0.02)
    cbar.set_label("E \u2212 E\u2090\u2091\u2099(R)  (Hartree)", fontsize=11)

    for tPath, R in trajData:
        lineColor = distCmap(norm(R))
        axLeft.plot(tPath, [R] * len(tPath), color=lineColor, linewidth=2.0, alpha=0.85, zorder=3)
        # Arrow at 60% along the path to show direction of descent
        arrowIdx = int(0.60 * len(tPath))
        axLeft.annotate(
            "",
            xy=(tPath[arrowIdx + 1], R),
            xytext=(tPath[arrowIdx], R),
            arrowprops=dict(arrowstyle="-|>", color=lineColor, lw=1.8),
            zorder=4,
        )

    sm = plt.cm.ScalarMappable(cmap=distCmap, norm=norm)
    sm.set_array([])

    axLeft.set_xlabel("Ansatz Parameter \u03b8", fontsize=12)
    axLeft.set_ylabel("Bond Distance (\u00c5)", fontsize=12)
    axLeft.set_title("Energy Above Minimum vs. Ansatz Parameter", fontsize=13)

    # ── Right: convergence curves (log scale) ─────────────────────────────────
    distLabels = [f"d = {distances[i]:.2f} \u00c5" for i in convIndices]
    for eSteps, gnd, label, c in zip(convEnergy, convGround, distLabels, convPalette):
        err = np.maximum(np.abs(eSteps - gnd), 1e-10)
        axRight.plot(err, color=c, linewidth=2.2, label=label)

    axRight.set_yscale("log")
    axRight.set_xlabel("VQE Iteration", fontsize=12)
    axRight.set_ylabel("|E \u2212 E\u2080|  (Hartree)", fontsize=12)
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
