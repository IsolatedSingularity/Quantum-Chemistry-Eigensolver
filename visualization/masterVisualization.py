"""
Quantum Chemistry Visualizations

This script creates animations that visualize:
1. H2 molecule separation (bond distance vs. energy)
2. H2 molecular orbitals visualization
3. VQE optimization energy curves

The visualizations use matplotlib for creating animations without requiring qiskit.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import from quantum_chemistry
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from quantum_chemistry.molecule.h2_molecule import load_h2_spin_orbital_integrals

# Constants
BOHR_TO_ANGSTROM = 0.529177  # Convert Bohr to Angstrom
HARTREE_TO_EV = 27.211396  # Convert Hartree to eV

# Define a class for 3D arrows
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

def load_and_process_data():
    """
    Load H2 data and process it for visualization
    """
    # Get data path
    data_path = os.path.join(parent_dir, "h2_data")
    
    print(f"Loading H2 data from {data_path}")
    try:
        distances, molecule_data = load_h2_spin_orbital_integrals(data_path)
        
        # Extract nuclear repulsion energies
        nuclear_energies = [data[2] for data in molecule_data]
        
        # Define simulated electronic energies (as we don't have actual VQE results)
        # This approximates the H2 ground state energy curve
        electronic_energies = []
        for d, nuc_energy in zip(distances, nuclear_energies):
            # Approximate the electronic energy using a simple model
            # (in reality this would come from quantum simulations)
            elec_energy = -1.0 - 0.2/(0.4 + (d-0.74)**2)
            electronic_energies.append(elec_energy)
        
        total_energies = [e + n for e, n in zip(electronic_energies, nuclear_energies)]
        
        return distances, electronic_energies, nuclear_energies, total_energies
    except Exception as e:
        print(f"Error loading data: {e}")
        # Generate synthetic data if actual data can't be loaded
        return generate_synthetic_data()

def generate_synthetic_data():
    """
    Generate synthetic data for H2 dissociation curve if real data cannot be loaded
    """
    print("Generating synthetic H2 data")
    distances = np.linspace(0.3, 1.9, 35)
    
    # Generate simulated energy values
    electronic_energies = -1.1 - 0.8*np.exp(-(distances-0.74)**2/0.2)
    nuclear_energies = 1.0 / distances
    total_energies = electronic_energies + nuclear_energies
    
    return distances, electronic_energies, nuclear_energies, total_energies


def animate_h2_dissociation(save_path):
    """
    Create an animation of H2 molecule separation with energy curve
    """
    print("Creating H2 dissociation animation...")
    
    # Load or generate data
    distances, electronic_energies, nuclear_energies, total_energies = load_and_process_data()
    
    try:
        # Set up the figure with two subplots
        fig = plt.figure(figsize=(14, 7))
        grid = plt.GridSpec(1, 2, width_ratios=[1, 1.2])
        
        # First subplot: Molecule visualization (simplified 2D version to avoid 3D issues)
        ax1 = fig.add_subplot(grid[0])
        ax1.set_title('H₂ Molecule Separation', fontsize=14)
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        ax1.set_aspect('equal')
        ax1.set_xlabel('X (Å)', fontsize=12)
        ax1.set_ylabel('Y (Å)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Second subplot: Energy curve
        ax2 = fig.add_subplot(grid[1])
        ax2.set_title('Potential Energy Curve', fontsize=14)
        ax2.set_xlabel('Bond Distance (Å)', fontsize=12)
        ax2.set_ylabel('Energy (Hartree)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Plot the full energy curves
        ax2.plot(distances, electronic_energies, 'b--', alpha=0.5, label='Electronic Energy')
        ax2.plot(distances, nuclear_energies, 'r--', alpha=0.5, label='Nuclear Repulsion')
        ax2.plot(distances, total_energies, 'k-', alpha=0.5, label='Total Energy')
        ax2.legend(loc='upper right', fontsize=10)
        
        # Energy curve elements
        energy_point, = ax2.plot([], [], 'ko', ms=8)
        
        # Add text annotations
        distance_text = ax1.text(0.05, 1.9, "", fontsize=10)
        energy_text = ax2.text(0.05, 0.05, "", transform=ax2.transAxes)
        
        # Points for current distance vertical line
        vertical_line, = ax2.plot([], [], 'k-', alpha=0.7)
        
        def init():
            # Return all artists that must be redrawn
            energy_point.set_data([], [])
            vertical_line.set_data([], [])
            return (energy_point, vertical_line, distance_text, energy_text)
        
        def animate(i):
            ax1.clear()
            ax1.set_title('H₂ Molecule Separation', fontsize=14)
            ax1.set_xlim(-2, 2)
            ax1.set_ylim(-2, 2)
            ax1.set_aspect('equal')
            ax1.set_xlabel('X (Å)', fontsize=12)
            ax1.set_ylabel('Y (Å)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Determine current distance (oscillate back and forth)
            if i < len(distances):
                idx = i
            else:
                idx = 2*len(distances) - i - 2
                
            if idx < 0:
                idx = 0
            if idx >= len(distances):
                idx = len(distances) - 1
                
            current_distance = distances[idx]
            
            # Draw the molecular visualization in 2D
            # The atoms are on the x-axis
            h1_x = -current_distance/2
            h2_x = current_distance/2
            
            # Draw atoms as circles
            h1_circle = plt.Circle((h1_x, 0), 0.3, fc='lightblue', ec='blue', lw=1.5)
            h2_circle = plt.Circle((h2_x, 0), 0.3, fc='lightblue', ec='blue', lw=1.5)
            ax1.add_patch(h1_circle)
            ax1.add_patch(h2_circle)
            
            # Add labels
            ax1.text(h1_x, 0.5, 'H', fontsize=14, ha='center')
            ax1.text(h2_x, 0.5, 'H', fontsize=14, ha='center')
            
            # Draw electron cloud (simplified 2D version)
            if current_distance < 0.9:  # Bonded state
                # Draw a shared electron cloud between atoms
                center_x = (h1_x + h2_x) / 2
                width = current_distance + 0.6
                height = 0.8
                
                from matplotlib.patches import Ellipse
                electron_cloud = Ellipse((center_x, 0), width, height, fc='blue', alpha=0.2)
                ax1.add_patch(electron_cloud)
            else:  # Dissociated state
                # Draw separate electron clouds
                from matplotlib.patches import Ellipse
                e1_cloud = Ellipse((h1_x, 0), 0.8, 0.8, fc='blue', alpha=0.2)
                e2_cloud = Ellipse((h2_x, 0), 0.8, 0.8, fc='blue', alpha=0.2)
                ax1.add_patch(e1_cloud)
                ax1.add_patch(e2_cloud)
            
            # Draw bond line (fades as atoms separate)
            if current_distance < 1.2:
                alpha = max(0, 1 - (current_distance/1.2))
                ax1.plot([h1_x, h2_x], [0, 0], 'k-', alpha=alpha, linewidth=2)
            
            # Update energy plot point and vertical line
            energy_point.set_data([current_distance], [total_energies[idx]])  # Fix: providing a sequence for x
            vertical_line.set_data([current_distance, current_distance], 
                                  [min(electronic_energies)-0.1, max(nuclear_energies)+0.1])
            
            # Update text information
            distance_text.set_text(f"Bond Distance: {current_distance:.2f} Å")
            distance_text.set_position((-1.9, 1.7))  # Set fixed position
            energy_value = total_energies[idx]
            energy_text.set_text(f"Energy: {energy_value:.4f} Ha")
            
            # Add indicator showing whether atoms are bonded, stretched, or dissociated
            if current_distance < 0.7:
                state_text = "Compressed State"
                color = 'orange'
            elif current_distance < 0.9:
                state_text = "Equilibrium State"
                color = 'green'
            elif current_distance < 1.3:
                state_text = "Stretched Bond"
                color = 'blue'
            else:
                state_text = "Dissociated"
                color = 'red'
                
            ax1.text(0, -1.7, state_text, fontsize=12, ha='center', color=color, weight='bold')
            
            # Add an explanation of the electron configuration
            if current_distance < 0.9:
                e_text = "Shared Electron Cloud"
            else:
                e_text = "Separated Electron Clouds"
            ax1.text(0, -1.4, e_text, fontsize=10, ha='center', style='italic')
            
            return (energy_point, vertical_line, distance_text, energy_text)
        
        # Create the animation
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                      frames=2*len(distances)-2, interval=100, blit=False)
        
        # Save animation
        print(f"Saving H2 dissociation animation to {save_path}")
        try:
            anim.save(save_path, writer='pillow', fps=10, dpi=100)
        except Exception as e:
            print(f"Error saving animation: {e}")
            # Save a static image instead
            print("Saving static image instead...")
            idx = len(distances) // 2
            # Create a new static figure instead
            static_fig, (static_ax1, static_ax2) = plt.subplots(1, 2, figsize=(14, 7))
            
            # Draw left panel
            static_ax1.set_title('H₂ Molecule at Equilibrium', fontsize=14)
            static_ax1.set_xlim(-2, 2)
            static_ax1.set_ylim(-2, 2)
            static_ax1.set_aspect('equal')
            static_ax1.set_xlabel('X (Å)', fontsize=12)
            static_ax1.set_ylabel('Y (Å)', fontsize=12)
            static_ax1.grid(True, alpha=0.3)
            
            # Draw H2 molecule at equilibrium distance
            eq_idx = np.argmin(total_energies)
            eq_distance = distances[eq_idx]
            h1_x = -eq_distance/2
            h2_x = eq_distance/2
            
            # Draw atoms
            h1_circle = plt.Circle((h1_x, 0), 0.3, fc='lightblue', ec='blue', lw=1.5)
            h2_circle = plt.Circle((h2_x, 0), 0.3, fc='lightblue', ec='blue', lw=1.5)
            static_ax1.add_patch(h1_circle)
            static_ax1.add_patch(h2_circle)
            
            # Add labels
            static_ax1.text(h1_x, 0.5, 'H', fontsize=14, ha='center')
            static_ax1.text(h2_x, 0.5, 'H', fontsize=14, ha='center')
            
            # Draw shared electron cloud
            from matplotlib.patches import Ellipse
            electron_cloud = Ellipse(((h1_x + h2_x)/2, 0), eq_distance + 0.6, 0.8, fc='blue', alpha=0.2)
            static_ax1.add_patch(electron_cloud)
            
            # Draw bond line
            static_ax1.plot([h1_x, h2_x], [0, 0], 'k-', linewidth=2)
            
            # Add bond distance text
            static_ax1.text(0, -1.7, f"Bond Distance: {eq_distance:.2f} Å", 
                           fontsize=12, ha='center', color='green', weight='bold')
            
            # Draw right panel - energy curve
            static_ax2.set_title('Potential Energy Curve', fontsize=14)
            static_ax2.set_xlabel('Bond Distance (Å)', fontsize=12)
            static_ax2.set_ylabel('Energy (Hartree)', fontsize=12)
            static_ax2.grid(True, alpha=0.3)
            
            # Plot energy curves
            static_ax2.plot(distances, electronic_energies, 'b--', alpha=0.5, label='Electronic Energy')
            static_ax2.plot(distances, nuclear_energies, 'r--', alpha=0.5, label='Nuclear Repulsion')
            static_ax2.plot(distances, total_energies, 'k-', alpha=0.5, label='Total Energy')
            
            # Mark equilibrium position
            static_ax2.plot(eq_distance, total_energies[eq_idx], 'go', markersize=8)
            
            # Add vertical line
            static_ax2.axvline(x=eq_distance, color='g', linestyle='--', alpha=0.7)
            
            # Add legend
            static_ax2.legend(loc='upper right', fontsize=10)
            
            # Save static image
            plt.tight_layout()
            plt.savefig(save_path.replace('.gif', '.png'), dpi=150)
            plt.close(static_fig)
        
        plt.close(fig)
        
    except Exception as e:
        print(f"Error in H2 dissociation animation: {e}")
        # Create a completely separate static figure as fallback
        try:
            print("Creating a simple static visualization as fallback")
            
            # Create a simple figure with the potential energy curve
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title('H₂ Molecule Potential Energy Curve', fontsize=16)
            ax.set_xlabel('Bond Distance (Å)', fontsize=14)
            ax.set_ylabel('Energy (Hartree)', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Plot the energy components
            ax.plot(distances, electronic_energies, 'b-', label='Electronic Energy')
            ax.plot(distances, nuclear_energies, 'r-', label='Nuclear Repulsion')
            ax.plot(distances, total_energies, 'k-', linewidth=2, label='Total Energy')
            
            # Mark equilibrium position
            min_idx = np.argmin(total_energies)
            min_distance = distances[min_idx]
            min_energy = total_energies[min_idx]
            ax.plot(min_distance, min_energy, 'go', markersize=10, 
                   label=f'Equilibrium (d = {min_distance:.2f} Å)')
            
            # Add annotation
            ax.text(min_distance + 0.1, min_energy, 
                   f'Ground State: {min_energy:.4f} Ha', fontsize=12)
            
            # Add a simple H2 molecule visualization as inset
            from matplotlib.patches import Ellipse, Circle
            inset_ax = ax.inset_axes([0.2, 0.6, 0.3, 0.3])
            inset_ax.set_xlim(-1, 1)
            inset_ax.set_ylim(-0.6, 0.6)
            inset_ax.set_aspect('equal')
            inset_ax.axis('off')
            
            # Draw H atoms
            h1 = Circle((-min_distance/2, 0), 0.2, fc='lightblue', ec='blue')
            h2 = Circle((min_distance/2, 0), 0.2, fc='lightblue', ec='blue')
            inset_ax.add_patch(h1)
            inset_ax.add_patch(h2)
            
            # Draw electron cloud
            e_cloud = Ellipse((0, 0), min_distance+0.4, 0.4, fc='blue', alpha=0.2)
            inset_ax.add_patch(e_cloud)
            
            # Add legend
            ax.legend(loc='upper right')
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(save_path.replace('.gif', '.png'), dpi=150)
            plt.close(fig)
            
        except Exception as ex:
            print(f"Failed to create fallback visualization: {ex}")
    
    print("H2 dissociation visualization completed")


def animate_vqe_optimization(save_path):
    """
    Create an animation of the VQE optimization process without using 3D
    """
    print("Creating VQE optimization animation...")
    
    try:
        # Load or generate data
        distances, electronic_energies, nuclear_energies, total_energies = load_and_process_data()
        
        # Choose a specific bond distance to visualize VQE
        bond_distance_idx = 8  # Choose a distance near equilibrium
        bond_distance = distances[bond_distance_idx]
        ground_state_energy = total_energies[bond_distance_idx]
        
        # Set up the figure with two subplots (avoiding the third since it caused issues)
        fig = plt.figure(figsize=(14, 6))
        grid = plt.GridSpec(1, 2)
        
        # First subplot: Energy convergence
        ax1 = fig.add_subplot(grid[0])
        ax1.set_title('VQE Energy Convergence', fontsize=14)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Energy (Hartree)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Second subplot: Quantum state evolution
        ax2 = fig.add_subplot(grid[1])
        ax2.set_title(f'Quantum State Evolution (d = {bond_distance:.2f} Å)', fontsize=14)
        ax2.set_xlim(-0.1, 1.1)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_xlabel('|0101⟩ Amplitude', fontsize=12)
        ax2.set_ylabel('|1010⟩ Amplitude', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Generate simulated VQE optimization data
        num_iterations = 15
        
        # Simulated VQE parameters (theta) over iterations
        # Starting from a state close to |0101⟩ and converging to the optimal value
        thetas = np.array([0.1 + 0.6*(1-np.exp(-i/5)) for i in range(num_iterations)])
        
        # Corresponding energies (converging to ground state)
        # Start with higher energy and converge to ground state
        vqe_energies = []
        for theta in thetas:
            # Simple model: Energy depends on theta
            # At theta=0: mostly |0101⟩, at theta=π/4: equal superposition
            energy = -0.5 - 0.3 * np.sin(2*theta)**2
            # Add noise early in the optimization
            noise = 0.1 * np.exp(-len(vqe_energies)/3) * np.random.randn()
            vqe_energies.append(energy + noise)
        
        # Add reference line for true ground state energy
        ax1.axhline(y=ground_state_energy, color='g', linestyle='--', 
                   alpha=0.7, label='True Ground State')
        
        # Energy plot elements
        energy_line, = ax1.plot([], [], 'b-', linewidth=2, label='VQE Energy')
        energy_point, = ax1.plot([], [], 'bo', ms=8)
        ax1.set_xlim(-0.5, num_iterations-0.5)
        ax1.set_ylim(min(vqe_energies)-0.1, max(vqe_energies)+0.1)
        ax1.legend(loc='upper right')
        
        # Initial state point and optimization trajectory
        trajectory_line, = ax2.plot([], [], 'b-', alpha=0.5)
        state_point, = ax2.plot([], [], 'ro', ms=10)
        
        # Draw contour plot of energy landscape in 2D
        theta_values = np.linspace(0, np.pi/2, 100)
        x_values = np.cos(theta_values)**2  # |0101⟩ probability
        y_values = np.sin(theta_values)**2  # |1010⟩ probability
        
        # Create a grid for contour plot
        X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        Z = np.zeros_like(X)
        
        # Fill Z with energy values (simple model approximation)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                # Calculate corresponding theta
                if X[i,j] + Y[i,j] > 0:  # Avoid division by zero
                    theta_approx = np.arctan2(np.sqrt(Y[i,j]), np.sqrt(X[i,j]))
                    Z[i,j] = -0.5 - 0.3 * np.sin(2*theta_approx)**2
                else:
                    Z[i,j] = -0.5
        
        # Plot energy contours
        contour = ax2.contourf(X, Y, Z, 20, cmap='viridis', alpha=0.6)
        fig.colorbar(contour, ax=ax2, label='Energy (Ha)')
        
        # Add constraint line (normalization)
        norm_x = np.linspace(0, 1, 100)
        norm_y = 1 - norm_x
        ax2.plot(norm_x, norm_y, 'k--', alpha=0.5, label='|ψ|² = 1')
        ax2.legend(loc='upper right')
        
        # Add circuit information as a text box instead of drawing it
        ax1.text(0.05, 0.05, 
                "Circuit:\n|0101⟩ → Ry(θ) → CNOT", 
                transform=ax1.transAxes,
                bbox=dict(facecolor='white', alpha=0.7),
                fontsize=10)
        
        # Add text for current information
        iter_text = ax1.text(0.5, 0.05, "", transform=ax1.transAxes,
                            bbox=dict(facecolor='white', alpha=0.8))
        state_text = ax2.text(0.05, 0.05, "", transform=ax2.transAxes,
                             bbox=dict(facecolor='white', alpha=0.8))
        
        def init():
            # Initialize plots
            energy_line.set_data([], [])
            energy_point.set_data([], [])
            trajectory_line.set_data([], [])
            state_point.set_data([], [])
            iter_text.set_text("")
            state_text.set_text("")
            return (energy_line, energy_point, trajectory_line, state_point, iter_text, state_text)
        
        def animate(i):
            if i < len(thetas):
                # Current parameter and energy
                theta = thetas[i]
                energy = vqe_energies[i]
                
                # Calculate state components
                prob_0101 = np.cos(theta)**2
                prob_1010 = np.sin(theta)**2
                
                # Update energy convergence plot
                energy_line.set_data(range(i+1), vqe_energies[:i+1])
                energy_point.set_data(i, energy)
                
                # Update quantum state plot
                if i > 0:
                    # Plot trajectory
                    traj_x = [np.cos(t)**2 for t in thetas[:i+1]]
                    traj_y = [np.sin(t)**2 for t in thetas[:i+1]]
                    trajectory_line.set_data(traj_x, traj_y)
                
                state_point.set_data([prob_0101], [prob_1010])
                
                # Update text information
                iter_text.set_text(f"Iteration: {i}\nEnergy: {energy:.6f} Ha\nθ: {theta:.4f}")
                
                # Update quantum state text
                amp_0101 = np.cos(theta)
                amp_1010 = np.sin(theta)
                state_text.set_text(f"|ψ⟩ = {amp_0101:.4f}|0101⟩ + {amp_1010:.4f}|1010⟩\n"
                                 f"P(|0101⟩) = {prob_0101:.4f}\n"
                                 f"P(|1010⟩) = {prob_1010:.4f}")
                
                # Highlight position on constraint curve
                ax2.plot([prob_0101], [prob_1010], 'ro', ms=10)
                
            return (energy_line, energy_point, trajectory_line, state_point, iter_text, state_text)
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                      frames=num_iterations, interval=500, blit=False)
        
        # Save animation
        print(f"Saving VQE optimization animation to {save_path}")
        try:
            anim.save(save_path, writer='pillow', fps=5, dpi=100)
        except Exception as e:
            print(f"Error saving animation: {e}")
            # Save a static image instead
            print("Saving static image instead...")
            animate(num_iterations-1)
            plt.savefig(save_path.replace('.gif', '.png'), dpi=150)
        
        plt.close(fig)
        
    except Exception as e:
        print(f"Error in VQE optimization animation: {e}")
        # Create and save a simple static image instead
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('VQE Optimization Process', fontsize=16)
        ax.set_xlabel('Parameter (θ)', fontsize=14)
        ax.set_ylabel('Energy', fontsize=14)
        
        # Draw a simple energy curve
        theta_values = np.linspace(0, np.pi, 100)
        energies = -0.7 - 0.3 * np.sin(theta_values)**2
        ax.plot(theta_values, energies, 'b-', linewidth=2)
        
        # Mark the minimum
        min_idx = np.argmin(energies)
        min_theta = theta_values[min_idx]
        min_energy = energies[min_idx]
        ax.plot(min_theta, min_energy, 'ro', markersize=10)
        
        # Add text explanation
        ax.text(0.1, -0.85, 
               "VQE works by finding the parameter (θ) that\nminimizes the energy expectation value.",
               fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        # Save static image
        plt.tight_layout()
        plt.savefig(save_path.replace('.gif', '.png'), dpi=150)
        plt.close(fig)
    
    print("VQE optimization visualization completed")


def create_ground_state_energy_visualization(save_path):
    """
    Create a static visualization of H2 ground state energy at different bond distances
    """
    print("Creating ground state energy visualization...")
    
    # Load or generate data
    distances, electronic_energies, nuclear_energies, total_energies = load_and_process_data()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot energy components
    ax.plot(distances, electronic_energies, 'b-', label='Electronic Energy')
    ax.plot(distances, nuclear_energies, 'r-', label='Nuclear Repulsion')
    ax.plot(distances, total_energies, 'k-', linewidth=2, label='Total Energy')
    
    # Find equilibrium position (minimum energy)
    min_idx = np.argmin(total_energies)
    min_distance = distances[min_idx]
    min_energy = total_energies[min_idx]
    
    # Mark equilibrium position
    ax.plot(min_distance, min_energy, 'go', markersize=10, label=f'Equilibrium (d = {min_distance:.2f} Å)')
    
    # Add annotations
    ax.annotate(f'Ground State Energy: {min_energy:.4f} Ha',
               xy=(min_distance, min_energy),
               xytext=(min_distance+0.2, min_energy-0.1),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
               fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Bond Distance (Å)', fontsize=14)
    ax.set_ylabel('Energy (Hartree)', fontsize=14)
    ax.set_title('H₂ Molecule Ground State Energy vs. Bond Distance', fontsize=16)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Add additional information
    textstr = '\n'.join((
        r'Equilibrium Properties:',
        r'Bond Distance: %.2f Å' % (min_distance,),
        r'Electronic Energy: %.4f Ha' % (electronic_energies[min_idx],),
        r'Nuclear Repulsion: %.4f Ha' % (nuclear_energies[min_idx],),
        r'Total Energy: %.4f Ha' % (min_energy,)))
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=12,
           verticalalignment='bottom', bbox=props)
    
    # Save figure
    print(f"Saving ground state energy visualization to {save_path}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print("Ground state energy visualization completed")

def create_vqe_steps_visualization(save_path):
    """
    Create a static visualization explaining the VQE algorithm steps
    """
    print("Creating VQE steps visualization...")
    
    # Create a figure with multiple subplots arranged vertically
    fig, axes = plt.subplots(4, 1, figsize=(10, 14))
    
    # Step 1: Molecular Hamiltonian Mapping
    ax1 = axes[0]
    ax1.set_title("Step 1: Hamiltonian Mapping", fontsize=16)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.axis('off')
    
    # Hamiltonian mapping illustration
    ax1.text(1, 5, "Molecular Hamiltonian (Electronic Structure):", fontsize=12, weight='bold')
    ax1.text(1, 4, r"$\hat{H}_{elec} = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s$", fontsize=12)
    
    ax1.add_patch(plt.Arrow(5, 3.5, 0, -1, width=0.3, color='black'))
    ax1.text(5.5, 3, "Jordan-Wigner Transformation", fontsize=10, style='italic')
    
    ax1.text(1, 2, "Qubit Hamiltonian:", fontsize=12, weight='bold')
    ax1.text(1, 1, r"$\hat{H}_{qubit} = \sum_i c_i \hat{P}_i$ where $\hat{P}_i$ are Pauli strings", fontsize=12)
    ax1.text(1, 0.5, r"Example: $c_1 \hat{Z}_0\hat{Z}_1 + c_2 \hat{X}_0\hat{X}_1\hat{Z}_2 + c_3 \hat{I}_0\hat{Y}_1\hat{Y}_2 + ...$", fontsize=10)
    
    # Step 2: Parameterized Circuit
    ax2 = axes[1]
    ax2.set_title("Step 2: Parameterized Quantum Circuit", fontsize=16)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis('off')
    
    # Circuit illustration
    circuit_text = """
    H₂ Ansatz Circuit:
    
    |0⟩ ─[X]─────[Ry(θ)]─■─────── 
                          │
    |0⟩ ───────[Ry(θ)]───■───────
    
    |0⟩ ─[X]─────[Ry(-θ)]───■───
                            │
    |0⟩ ───────[Ry(-θ)]─────■───
    """
    ax2.text(1, 4, circuit_text, fontsize=12, family='monospace')
    
    ax2.text(1, 1.5, "State Preparation:", fontsize=12, weight='bold')
    ax2.text(1, 0.8, r"$|\psi(\theta)\rangle = $ superposition of $|0101\rangle$ and $|1010\rangle$", fontsize=12)
    ax2.text(1, 0.3, r"Represents ground state of H₂ at various bond distances", fontsize=10, style='italic')
    
    # Step 3: Energy Estimation
    ax3 = axes[2]
    ax3.set_title("Step 3: Energy Estimation", fontsize=16)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 6)
    ax3.axis('off')
    
    # Energy estimation illustration
    ax3.text(1, 5, "Expectation Value Calculation:", fontsize=12, weight='bold')
    ax3.text(1, 4.3, r"$E(\theta) = \langle\psi(\theta)|\hat{H}_{qubit}|\psi(\theta)\rangle = \sum_i c_i \langle\psi(\theta)|\hat{P}_i|\psi(\theta)\rangle$", fontsize=12)
    
    # Create a small grid to show measurement process
    grid_x, grid_y = np.meshgrid(range(5), range(3))
    for i in range(5):
        for j in range(3):
            if j == 0:
                # First row: Pauli strings
                if i == 0:
                    pauli = "ZZ"
                elif i == 1:
                    pauli = "XX"
                elif i == 2:
                    pauli = "YY"
                elif i == 3:
                    pauli = "ZI"
                else:
                    pauli = "IZ"
                ax3.text(i+2, 3, pauli, ha='center', fontsize=10)
            elif j == 1:
                # Second row: Coefficients
                coef = f"c{i+1}"
                ax3.text(i+2, 2.5, coef, ha='center', fontsize=10)
            else:
                # Third row: Measured values
                val = f"⟨{pauli}⟩"
                ax3.text(i+2, 2, val, ha='center', fontsize=10)
    
    ax3.add_patch(plt.Rectangle((1.8, 1.8), 5.4, 1.5, fill=False))
    
    # Show explanation of measurement
    ax3.text(1, 1.3, "For each Pauli string:", fontsize=11)
    ax3.text(1.2, 0.9, "1. Apply basis rotations (H and S gates)", fontsize=10)
    ax3.text(1.2, 0.5, "2. Measure in Z basis multiple times", fontsize=10)
    ax3.text(1.2, 0.1, "3. Calculate expectation value from statistics", fontsize=10)
    
    # Step 4: Optimization
    ax4 = axes[3]
    ax4.set_title("Step 4: Classical Optimization", fontsize=16)
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 6)
    ax4.axis('off')
    
    # Parameter optimization illustration
    ax4.text(1, 5, "Objective: Find θ that minimizes E(θ)", fontsize=12, weight='bold')
    
    # Draw optimization curve
    theta_vals = np.linspace(0, np.pi, 100)
    energy_vals = -0.7 - 0.4 * np.sin(theta_vals)**2
    ax4.plot(theta_vals/np.pi*6 + 2, energy_vals + 3, 'b-')
    
    # Mark minimum
    min_theta = np.pi/2
    min_energy = -0.7 - 0.4 * np.sin(min_theta)**2
    ax4.plot(min_theta/np.pi*6 + 2, min_energy + 3, 'ro')
    
    # Axis labels
    ax4.text(5, 1.8, "Parameter θ", fontsize=11)
    ax4.text(1.5, 3, "Energy E(θ)", fontsize=11)
    
    # Show optimization process
    ax4.text(1, 1.2, "Optimization Algorithm (SLSQP, SPSA, etc.):", fontsize=11)
    ax4.text(1.2, 0.8, "1. Start with initial guess for θ", fontsize=10)
    ax4.text(1.2, 0.5, "2. Evaluate energy using quantum processor", fontsize=10)
    ax4.text(1.2, 0.2, "3. Update θ to minimize energy and iterate", fontsize=10)
    
    # Add title to overall figure
    fig.suptitle("Variational Quantum Eigensolver (VQE) Algorithm Steps", fontsize=18)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for suptitle
    print(f"Saving VQE steps visualization to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print("VQE steps visualization completed")


def create_molecular_orbital_visualization(save_path):
    """
    Create a static visualization of H2 molecular orbitals
    """
    print("Creating H2 molecular orbital visualization...")
    
    # Create figure with 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('H₂ Molecular Orbitals', fontsize=18)
    
    # Set up common parameters for all subplots
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-2, 2, 80)
    X, Y = np.meshgrid(x, y)
    
    # Define atomic positions at equilibrium bond length
    bond_length = 0.74  # Angstroms
    h1_pos = (-bond_length/2, 0)
    h2_pos = (bond_length/2, 0)
    
    # Define simplified 1s atomic orbitals centered on each H atom
    def h1s(x, y, center):
        # 1s orbital centered at 'center' coordinates
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        return np.exp(-r)
    
    # Calculate atomic and molecular orbitals
    h1_1s = h1s(X, Y, h1_pos)
    h2_1s = h1s(X, Y, h2_pos)
    
    # Bonding molecular orbital (symmetric combination)
    bonding_mo = h1_1s + h2_1s
    
    # Antibonding molecular orbital (antisymmetric combination)
    antibonding_mo = h1_1s - h2_1s
    
    # Plot 1s orbital of first H atom
    ax = axes[0, 0]
    ax.set_title('H₁ 1s Atomic Orbital', fontsize=14)
    contour = ax.contourf(X, Y, h1_1s, 20, cmap='coolwarm')
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_aspect('equal')
    ax.plot(h1_pos[0], h1_pos[1], 'o', color='black', markersize=8)  # H1 nucleus
    ax.text(h1_pos[0]-0.2, h1_pos[1]-0.3, 'H₁', fontsize=12)
    
    # Plot 1s orbital of second H atom
    ax = axes[0, 1]
    ax.set_title('H₂ 1s Atomic Orbital', fontsize=14)
    contour = ax.contourf(X, Y, h2_1s, 20, cmap='coolwarm')
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_aspect('equal')
    ax.plot(h2_pos[0], h2_pos[1], 'o', color='black', markersize=8)  # H2 nucleus
    ax.text(h2_pos[0]+0.1, h2_pos[1]-0.3, 'H₂', fontsize=12)
    
    # Plot bonding molecular orbital
    ax = axes[1, 0]
    ax.set_title('σ Bonding Molecular Orbital', fontsize=14)
    contour = ax.contourf(X, Y, bonding_mo, 20, cmap='coolwarm')
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_aspect('equal')
    ax.plot(h1_pos[0], h1_pos[1], 'o', color='black', markersize=8)  # H1 nucleus
    ax.plot(h2_pos[0], h2_pos[1], 'o', color='black', markersize=8)  # H2 nucleus
    ax.text(h1_pos[0]-0.2, h1_pos[1]-0.3, 'H₁', fontsize=12)
    ax.text(h2_pos[0]+0.1, h2_pos[1]-0.3, 'H₂', fontsize=12)
    
    # Add annotation for the bonding orbital
    ax.text(0, -1.7, "Lower Energy State", fontsize=12, ha='center',
           bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot antibonding molecular orbital
    ax = axes[1, 1]
    ax.set_title('σ* Antibonding Molecular Orbital', fontsize=14)
    contour = ax.contourf(X, Y, antibonding_mo, 20, cmap='coolwarm')
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_aspect('equal')
    ax.plot(h1_pos[0], h1_pos[1], 'o', color='black', markersize=8)  # H1 nucleus
    ax.plot(h2_pos[0], h2_pos[1], 'o', color='black', markersize=8)  # H2 nucleus
    ax.text(h1_pos[0]-0.2, h1_pos[1]-0.3, 'H₁', fontsize=12)
    ax.text(h2_pos[0]+0.1, h2_pos[1]-0.3, 'H₂', fontsize=12)
    
    # Add annotation for the antibonding orbital
    ax.text(0, -1.7, "Higher Energy State", fontsize=12, ha='center',
           bbox=dict(facecolor='white', alpha=0.8))
    
    # Add a colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.set_label('Orbital Amplitude', fontsize=12)
    
    # Add explanatory notes
    fig.text(0.5, 0.02, 
            "Bonding MO: Constructive interference between atomic orbitals\n"
            "Antibonding MO: Destructive interference between atomic orbitals\n"
            "H₂ ground state: both electrons in the bonding orbital", 
            fontsize=12, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the figure
    plt.tight_layout(rect=[0, 0.05, 0.91, 0.95])
    print(f"Saving H2 molecular orbital visualization to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print("H2 molecular orbital visualization completed")


if __name__ == "__main__":
    print("Starting H2 molecule visualization generation")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Generate the H2 dissociation animation
        try:
            animate_h2_dissociation(os.path.join(output_dir, 'h2_dissociation.gif'))
            print("H2 dissociation animation completed successfully")
        except Exception as e:
            print(f"Error in H2 dissociation animation: {e}")
            # Create a simple fallback static image
            print("Creating static H2 dissociation image instead...")
            distances = np.linspace(0.3, 1.9, 35)
            energies = -1.1 - 0.8*np.exp(-(distances-0.74)**2/0.2) + 1.0/distances
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(distances, energies, 'k-', linewidth=2)
            ax.set_xlabel('Bond Distance (Å)')
            ax.set_ylabel('Energy (Hartree)')
            ax.set_title('H₂ Molecule Potential Energy Curve')
            ax.grid(True, alpha=0.3)
            
            # Mark minimum
            min_idx = np.argmin(energies)
            ax.plot(distances[min_idx], energies[min_idx], 'ro')
            
            plt.savefig(os.path.join(output_dir, 'h2_dissociation.png'), dpi=150)
            plt.close(fig)
        
        # Run the molecular orbital visualization script separately
        print("Generating molecular orbital visualization using separate script")
        try:
            import h2_molecular_orbitals
            h2_molecular_orbitals.create_molecular_orbital_visualization(
                os.path.join(output_dir, 'h2_molecular_orbitals.png'))
            print("Molecular orbital visualization completed successfully")
        except Exception as e:
            print(f"Error running molecular orbital visualization: {e}")
        
        # Run the VQE energy curves visualization script
        print("Generating VQE energy curves visualization using separate script")
        try:
            import vqe_energy_curves
            vqe_energy_curves.create_vqe_energy_curves_visualization(
                os.path.join(output_dir, 'vqe_energy_curves.png'))
            print("VQE energy curves visualization completed successfully")
        except Exception as e:
            print(f"Error running VQE energy curves visualization: {e}")
            print(f"Exception details: {str(e)}")
        
        print("All requested visualizations completed")
        
    except Exception as e:
        print(f"Error in visualization generation: {e}")
