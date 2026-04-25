"""
VQE Hero Visualization

Hero plot for the Quantum Chemistry Eigensolver.

Left panel: filled contour of E(theta, R) - E_0 across all 34 bond distances
(absolute energy shifted to the global minimum), revealing a single 2D well
near (theta_eq, R_eq ≈ 0.74 Å). A single 2D gradient-descent trajectory is
overlaid, with each iteration drawn as its own arrow segment so the descent
is visually explicit. Color: mako (matches vqe_energy_curves.py and
h2_molecular_orbitals.py).

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
from scipy.interpolate import RegularGridInterpolator

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

    # Absolute landscape: E(theta, R) shifted by the global minimum. Unlike
    # E - E_min(R) (which produces 34 stacked horizontal wells), this gives a
    # true 2D potential with a single global basin near (theta_eq, R_eq) and
    # nuclear-repulsion + dissociation cliffs at the edges -- a non-trivial
    # surface for the trajectory to navigate.
    absMin = float(energyGrid.min())
    pesGrid = energyGrid - absMin

    # ── 2D gradient-descent trajectory (single non-trivial path) ─────────────
    # Numerical gradients over the energy grid let the path move in BOTH theta
    # AND R, tracing a curve from a stretched, off-equilibrium start toward
    # the global minimum at (theta_eq, R_eq ≈ 0.74 Å).
    print("Computing 2D gradient-descent trajectory...")
    lr = 0.06  # kept for convergence section

    dEdTheta2d = np.gradient(energyGrid, thetas, axis=1)
    dEdR2d = np.gradient(energyGrid, distances, axis=0)
    gradThetaFn = RegularGridInterpolator(
        (distances, thetas), dEdTheta2d,
        method="linear", bounds_error=False, fill_value=None,
    )
    gradRFn = RegularGridInterpolator(
        (distances, thetas), dEdR2d,
        method="linear", bounds_error=False, fill_value=None,
    )

    # Single trajectory: start at compressed bond + off-axis theta. Both
    # components of the gradient are large here, so the path sweeps clearly
    # diagonally across the well toward (theta_eq, R_eq ≈ 0.74 Å).
    t0, R0 = np.pi * 0.75, distances[1]  # ≈ (2.36, 0.35 Å)
    lrTheta, lrR, nSteps2d = 0.30, 0.060, 28
    t, R = float(t0), float(R0)
    tPath, RPath = [t], [R]
    for _ in range(nSteps2d):
        gt = float(gradThetaFn([[R, t]]).item())
        gR = float(gradRFn([[R, t]]).item())
        t = float(((t - lrTheta * gt + np.pi) % (2.0 * np.pi)) - np.pi)
        R = float(np.clip(R - lrR * gR, distances[0], distances[-1]))
        tPath.append(t)
        RPath.append(R)
    tPath = np.array(tPath)
    RPath = np.array(RPath)

    # ── Convergence curves at 3 representative geometries ─────────────────────
    print("Computing convergence curves...")
    # 5 evenly spaced geometries -- avoids duplicates.
    convIndices = list(np.linspace(0, len(distances) - 1, 5, dtype=int))
    convEnergy = []
    convGround = []

    for dIdx in convIndices:
        h_mat, nuc = hMats[dIdx], nucs[dIdx]
        ground = float(np.min(energyGrid[dIdx]))
        convGround.append(ground)
        t = np.pi / 2.0
        steps = [_energyAt(h_mat, nuc, ansatz, t)]
        for _ in range(299):
            g = _psrGradient(h_mat, nuc, ansatz, t)
            t = t - lr * g
            t = float(((t + np.pi) % (2.0 * np.pi)) - np.pi)
            steps.append(_energyAt(h_mat, nuc, ansatz, t))
        convEnergy.append(np.array(steps))

    # ── Color palettes -- match vqe_energy_curves.py / h2_molecular_orbitals.py ──
    pesCmap = sns.color_palette("mako", as_cmap=True)
    convPalette = sns.color_palette("mako", n_colors=5)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.45, 1], wspace=0.30)
    axLeft = fig.add_subplot(gs[0])
    axRight = fig.add_subplot(gs[1])

    fig.suptitle("VQE Gradient Descent on H\u2082 Potential Energy Surface", fontsize=15)

    # ── Left: absolute PES + single trajectory ────────────────────────────────
    THETA, DIST = np.meshgrid(thetas, distances)
    # Linear levels on the shifted absolute energy: produces a single deep
    # mako-navy basin near (theta_eq, R_eq) that smoothly brightens through
    # teal to pale green at the dissociation/repulsion cliffs -- a true 2D
    # well, not stacked horizontal bands. Clip vmax to 1.0 Hartree so the
    # colour range is dominated by the chemically interesting region (±100 mHa
    # around the well), not the nuclear-repulsion blow-up at small R.
    pesVmax = min(float(pesGrid.max()), 1.0)
    pesGridClip = np.clip(pesGrid, 0.0, pesVmax)
    pesLevels = np.linspace(0.0, pesVmax, 40)
    cf = axLeft.contourf(THETA, DIST, pesGridClip, levels=pesLevels, cmap=pesCmap, extend="max")
    # Thin contour lines highlight the well structure (non-trivial pattern)
    axLeft.contour(THETA, DIST, pesGridClip, levels=pesLevels[::4], colors="white", alpha=0.18, linewidths=0.5)
    cbar = fig.colorbar(cf, ax=axLeft, fraction=0.035, pad=0.02)
    cbar.set_label("E \u2212 E\u2080  (Hartree)", fontsize=11)

    # Single trajectory: each step rendered as its own arrow segment so the
    # iterative descent is visually explicit.
    for i in range(len(tPath) - 1):
        if abs(tPath[i + 1] - tPath[i]) > np.pi:  # skip wrap-around
            continue
        axLeft.annotate(
            "",
            xy=(tPath[i + 1], RPath[i + 1]),
            xytext=(tPath[i], RPath[i]),
            arrowprops=dict(
                arrowstyle="-|>", color="white", lw=1.8, mutation_scale=12,
                alpha=0.95,
            ),
            zorder=4,
        )
    # Vertex dots so segment joins read crisply
    axLeft.scatter(tPath, RPath, color="white", s=18, zorder=5, alpha=0.95, linewidths=0)
    # Larger start marker
    axLeft.scatter(
        [tPath[0]], [RPath[0]], color="white", s=80, zorder=6,
        edgecolors="black", linewidths=1.2, marker="o",
    )

    axLeft.set_xlabel("Ansatz Parameter \u03b8", fontsize=12)
    axLeft.set_ylabel("Bond Distance (\u00c5)", fontsize=12)
    axLeft.set_title("Energy Above Global Minimum", fontsize=13)

    # ── Right: convergence curves (log scale) ─────────────────────────────────
    distLabels = [f"d = {distances[i]:.2f} \u00c5" for i in convIndices]
    for eSteps, gnd, label, c in zip(convEnergy, convGround, distLabels, convPalette):
        err = np.maximum(np.abs(eSteps - gnd), 1e-10)
        axRight.plot(err, color=c, linewidth=2.2, label=label)

    axRight.set_yscale("log")
    axRight.set_xlim(0, 300)
    axRight.set_ylim(bottom=1e-9, top=10.0)
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
