"""
VQE Energy Curves Visualization

This script creates a static visualization of the VQE optimization process showing multiple energy curves 
using a color palette to represent the progress of the optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.cm import viridis

def create_vqe_energy_curves_visualization(save_path):
    """
    Create a static visualization of the VQE optimization process showing multiple energy curves
    using the viridis color palette.
    
    Parameters:
    -----------
    save_path : str
        Path where the visualization will be saved
    """
    print("Creating VQE energy curves visualization...")
      # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [2, 1]})
    fig.suptitle('VQE Optimization Energy Landscape', fontsize=18)
    
    # Custom color palettes using seaborn
    # For the energy landscape (cubehelix)
    energy_cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
    # For the iterations (mako)
    iter_cmap = sns.color_palette("mako", as_cmap=True)
    
    # Parameters for the energy function
    bond_distance = 0.74  # Equilibrium bond distance in Angstroms

    # Create a grid of parameter values
    theta1_vals = np.linspace(-np.pi, np.pi, 100)
    theta2_vals = np.linspace(-np.pi, np.pi, 100)
    T1, T2 = np.meshgrid(theta1_vals, theta2_vals)
    
    # Define a simple energy function that depends on two parameters
    # This simulates the energy landscape for a VQE optimization
    def energy_function(theta1, theta2):
        # Simplified energy function with a minimum at (0, 0)
        return -1.2 - 0.5 * np.exp(-0.5 * (theta1**2 + theta2**2)) + \
               0.1 * (np.sin(2*theta1)**2 + np.sin(2*theta2)**2)
    
    # Calculate energy values for the grid
    energy_grid = energy_function(T1, T2)
    
    # Define optimization trajectory (simulated)
    # Starting from a random initial point and converging to the minimum
    num_iterations = 10
    np.random.seed(42)  # For reproducibility
    
    # Generate a trajectory with some randomness to simulate optimization path
    start_point = (-2, 1.5)  # Starting point for the optimization
    end_point = (0, 0)  # Global minimum
    
    # Create trajectory points with some noise
    trajectory_x = np.zeros(num_iterations)
    trajectory_y = np.zeros(num_iterations)
    
    # Linear path from start to end with decreasing noise
    for i in range(num_iterations):
        t = i / (num_iterations - 1)  # Interpolation parameter [0, 1]
        noise_scale = 0.5 * (1 - t)  # Noise decreases as we approach the minimum
        
        # Calculate the position with noise
        trajectory_x[i] = start_point[0] * (1 - t) + end_point[0] * t + noise_scale * np.random.randn()
        trajectory_y[i] = start_point[1] * (1 - t) + end_point[1] * t + noise_scale * np.random.randn()
    
    # Get energy values along the trajectory
    trajectory_energies = [energy_function(x, y) for x, y in zip(trajectory_x, trajectory_y)]    # Plot the energy landscape as a contour plot in the first subplot with cubehelix colormap
    contour = ax1.contourf(T1, T2, energy_grid, 50, cmap=energy_cmap)
    ax1.set_title('VQE Parameter Space Energy Landscape', fontsize=14)
    ax1.set_xlabel('Parameter θ₁', fontsize=12)
    ax1.set_ylabel('Parameter θ₂', fontsize=12)
    
    # Add colorbar for the contour plot
    cbar = fig.colorbar(contour, ax=ax1)
    cbar.set_label('Energy (Hartree)', fontsize=12)
    
    # Plot the optimization trajectory
    # Use mako colormap to color the trajectory points by iteration
    colors = iter_cmap(np.linspace(0, 1, num_iterations))
    
    # Plot the trajectory with changing colors to show progress
    for i in range(num_iterations-1):
        ax1.plot(trajectory_x[i:i+2], trajectory_y[i:i+2], 'o-', color=colors[i], linewidth=2, markersize=8)
      # Mark the start and end points
    ax1.plot(trajectory_x[0], trajectory_y[0], 'o', color='white', markersize=10, markeredgecolor='black', markeredgewidth=1.5, label='Initial Parameters')
    ax1.plot(trajectory_x[-1], trajectory_y[-1], '*', color=colors[-1], markersize=15, markeredgecolor='black', markeredgewidth=1.5, label='Final Parameters')
    
    # Add a legend
    ax1.legend(fontsize=10, loc='upper right')
    
    # Plot the energy curves for different iterations in the second subplot
    
    # Define bond distances for which we'll plot energy curves
    bond_distances = np.linspace(0.3, 2.0, 100)
    
    # For each iteration, we'll simulate how the energy vs. bond distance curve would look
    # with the parameters at that iteration
    
    # Function to generate an energy curve for a given set of parameters
    def get_energy_curve(theta1, theta2, bond_distances):
        # Baseline curve (correct ground state)
        baseline = -1.1 - 0.8 * np.exp(-(bond_distances-0.74)**2/0.2) + 1.0/bond_distances
        
        # Deviation depends on how far the parameters are from optimal
        param_error = np.sqrt(theta1**2 + theta2**2)
        deviation = 0.5 * param_error * np.exp(-(bond_distances-1.0)**2/0.5)
        
        return baseline + deviation
    
    # Plot multiple energy curves with viridis colormap to show iteration progress
    for i in range(num_iterations):
        energy_curve = get_energy_curve(trajectory_x[i], trajectory_y[i], bond_distances)
        ax2.plot(bond_distances, energy_curve, '-', color=colors[i], linewidth=2, alpha=0.8)
      # Add colorbar for the iterations
    sm = plt.cm.ScalarMappable(cmap=iter_cmap, norm=plt.Normalize(0, num_iterations-1))
    sm.set_array([])
    cbar2 = fig.colorbar(sm, ax=ax2)
    cbar2.set_label('Optimization Iteration', fontsize=12)
    
    # Set title and labels for the energy curves plot
    ax2.set_title('Energy Curves During Optimization', fontsize=14)
    ax2.set_xlabel('Bond Distance (Å)', fontsize=12)
    ax2.set_ylabel('Energy (Hartree)', fontsize=12)
    ax2.grid(True, alpha=0.3)
      # Mark the equilibrium bond distance
    ax2.axvline(x=0.74, color='gray', linestyle='--', alpha=0.5, label='Equilibrium Bond Distance')
    
    # Plot the final (best) energy curve with a thicker line
    final_energy_curve = get_energy_curve(trajectory_x[-1], trajectory_y[-1], bond_distances)
    final_color = colors[-1]  # Use the last color from the mako palette
    ax2.plot(bond_distances, final_energy_curve, '-', color=final_color, linewidth=3, label='Final Energy Curve')
      # Mark the minimum point on the final curve
    min_idx = np.argmin(final_energy_curve)
    min_distance = bond_distances[min_idx]
    min_energy = final_energy_curve[min_idx]
    ax2.plot(min_distance, min_energy, '*', color=final_color, markersize=15, markeredgecolor='black')
    
    # Add legend to the second subplot
    ax2.legend(fontsize=10, loc='upper right')
    
    # First adjust the subplot positions to make room for the text
    plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.2)
    
    # Add explanatory text with a clear background box to ensure visibility
    text_box = fig.text(0.5, 0.05, 
            "VQE optimization progressively improves the energy landscape.\n"
            "The colors represent progressive iterations from initial guess (dark blue) to final result (light blue).", 
            fontsize=12, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5', edgecolor='gray'))
    
    # Save the figure
    print(f"Saving VQE energy curves visualization to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print("VQE energy curves visualization completed")

if __name__ == "__main__":
    # Run the visualization directly
    output_dir = os.path.dirname(os.path.abspath(__file__))
    create_vqe_energy_curves_visualization(os.path.join(output_dir, 'vqe_energy_curves.png'))
