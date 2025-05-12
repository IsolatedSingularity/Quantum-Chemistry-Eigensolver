"""
H2 Molecular Orbital Visualization

This script creates a visualization of atomic and molecular orbitals for the H2 molecule.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

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
      # Adjust figure layout to make room for explanatory notes
    plt.tight_layout(rect=[0, 0.15, 0.91, 0.95])
    
    # Add explanatory notes with more vertical space
    fig.text(0.5, 0.04, 
            "Bonding MO: Constructive interference between atomic orbitals\n"
            "Antibonding MO: Destructive interference between atomic orbitals\n"
            "H₂ ground state: both electrons in the bonding orbital", 
            fontsize=12, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Create more space at the bottom of the figure
    fig.subplots_adjust(bottom=0.15)
    
    # Save the figure
    print(f"Saving H2 molecular orbital visualization to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print("H2 molecular orbital visualization completed")

if __name__ == "__main__":
    # Run the visualization directly
    output_dir = os.path.dirname(os.path.abspath(__file__))
    create_molecular_orbital_visualization(os.path.join(output_dir, 'h2_molecular_orbitals.png'))
