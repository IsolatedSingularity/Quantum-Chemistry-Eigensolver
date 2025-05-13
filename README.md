# Quantum-Chemistry-Eigensolver
###### Based on the quantum chemistry workshop notebooks created by [Maxime Dion](https://www.usherbrooke.ca/iq/en/news-events/news/details/54588) at the [Institut Quantique](https://www.usherbrooke.ca/iq/).

![VQE Energy Optimization Curves](https://github.com/IsolatedSingularity/quantum-chemistry-eigensolver/blob/main/visualization/vqe_energy_curves.png?raw=true)

## Objective

This repository implements a quantum chemistry eigensolver for simulating small molecular systems, with a focus on the H₂ molecule. Quantum chemistry is one of the most promising near-term applications of quantum computing, leveraging variational quantum algorithms to calculate molecular ground state energies.

The core of this implementation is the **Variational Quantum Eigensolver (VQE)**, a hybrid quantum-classical algorithm that approximates the ground state energy of quantum systems. For molecular systems, this energy is determined by the electronic Hamiltonian:

$\hat{H} = \sum_{p,q} h_{pq} a_p^\dagger a_q$

where $a_p^\dagger$ and $a_q$ are fermionic creation and annihilation operators, while $h_{pq}$ and $h_{pqrs}$ represent one- and two-electron integrals.

**Goal:** Simulate the H₂ molecule's energy landscape at various bond distances using the VQE algorithm, visualize molecular orbitals, and demonstrate the convergence of the optimization process toward the ground state energy.

## Theoretical Background

Quantum chemistry calculations begin with mapping the molecular Hamiltonian to a qubit representation. The most common approach uses the Jordan–Wigner transformation, which converts fermionic operators to Pauli operators:

$a_j^{\dagger} \rightarrow \tfrac12\bigl(X_j - iY_j\bigr) \prod_{k<j} Z_k, \quad a_j \rightarrow \tfrac12\bigl(X_j + iY_j\bigr) \prod_{k<j} Z_k$

For the H₂ molecule with a minimal basis, we need 4 qubits to represent the system's 4 spin orbitals. After mapping and applying symmetries, the qubit Hamiltonian can be expressed as a sum of tensor products of Pauli operators.

The VQE algorithm then works by:
1. Preparing a parameterized quantum state (ansatz) on a quantum computer
2. Measuring the expectation value of the Hamiltonian
3. Using a classical optimizer to update the parameters to minimize energy
4. Repeating until convergence

For H₂, a simple ansatz involves rotations and entangling gates to create superpositions of computational basis states that represent different electronic configurations.

## Code Functionality

### 1. Initialize Molecule and Parameters
Sets up the H₂ molecule with specific bond distances, defines basis sets, and initializes parameters for the simulation.

```python
# Extract spin orbital integrals for H2 at different bond distances
def load_h2_spin_orbital_integrals(data_path):
    """
    Load H2 spin orbital integral data from pre-computed files
    Returns distances and a list of (h1, h2, nuclear_repulsion) for each distance
    """
    distances = []
    molecule_data = []
    
    # Load data files in order 
    for file in sorted(os.listdir(data_path)):
        if file.startswith('h2_mo_integrals_d_') and file.endswith('.npz'):
            # Extract distance from filename (formatted as h2_mo_integrals_d_XXXX.npz)
            dist_str = file.split('_d_')[1].split('.npz')[0]
            # Convert from format like '0750' to 0.75
            distance = float(dist_str) / 1000.0
            
            # Load the data file
            print(f"Loading {file}")
            data = np.load(os.path.join(data_path, file))
            
            # Extract the data: one-electron integrals, two-electron integrals, nuclear repulsion
            h1 = data['h1']
            h2 = data['h2']
            nuclear_repulsion = data['nuclear_repulsion']
            
            distances.append(distance)
            molecule_data.append((h1, h2, nuclear_repulsion))
    
    return np.array(distances), molecule_data
```

### 2. Hamiltonian Construction and Mapping
Maps the molecular Hamiltonian from fermionic to qubit representation using the Jordan-Wigner transformation.

```python
def jordan_wigner_from_orbitals(one_body_integrals, two_body_integrals):
    """
    Convert molecular orbital integrals to qubit Hamiltonian using JW transformation
    """
    n_qubits = 2 * one_body_integrals.shape[0]
    hamiltonian_terms = []
    coefficients = []
    
    # Handle one-body integrals
    for p in range(n_qubits):
        for q in range(n_qubits):
            if one_body_integrals[p//2, q//2] != 0:
                # Apply JW transformation to a_p^† a_q
                # ... mapping code ...
                
    # Handle two-body integrals
    for p in range(n_qubits):
        for q in range(n_qubits):
            for r in range(n_qubits):
                for s in range(n_qubits):
                    if two_body_integrals[p//2, q//2, r//2, s//2] != 0:
                        # Apply JW transformation to a_p^† a_q^† a_r a_s
                        # ... mapping code ...
                        
    return hamiltonian_terms, coefficients
```

### 3. VQE Implementation
Implements the VQE algorithm with a parameterized quantum circuit.

```python
def run_vqe(hamiltonian_terms, coefficients, initial_params=None, max_iter=100):
    """
    Run the VQE algorithm with the given Hamiltonian
    """
    n_qubits = len(hamiltonian_terms[0])
    
    # Define a simple ansatz circuit
    def ansatz_circuit(params):
        # Build a parameterized circuit appropriate for H2
        circuit = []
        # Initial state preparation (e.g., Hartree-Fock state)
        circuit.append(('X', [0]))
        circuit.append(('X', [2]))
        
        # Parameterized rotations and entangling gates
        circuit.append(('RY', [0], params[0]))
        circuit.append(('RY', [1], params[1]))
        circuit.append(('CNOT', [0, 1]))
        # ... more gates ...
        
        return circuit
    
    # Define cost function (expectation value of the Hamiltonian)
    def cost(params):
        circuit = ansatz_circuit(params)
        return compute_expectation(circuit, hamiltonian_terms, coefficients)
    
    # Initial parameters (if not provided)
    if initial_params is None:
        initial_params = np.random.random(n_params) * 2 * np.pi
        
    # Run classical optimization
    result = minimize(cost, initial_params, method='COBYLA', options={'maxiter': max_iter})
    
    return result.x, result.fun
```

### 4. Visualizing H₂ Dissociation
Creates animations and static visualizations of the H₂ dissociation curve showing how energy changes with bond distance.

```python
def animate_h2_dissociation(save_path):
    """
    Create an animation of H2 molecule separation with energy curve
    """
    # Load or generate data
    distances, electronic_energies, nuclear_energies, total_energies = load_and_process_data()
    
    # Set up the figure with two subplots
    fig = plt.figure(figsize=(14, 7))
    grid = plt.GridSpec(1, 2, width_ratios=[1, 1.2])
    
    # First subplot: Molecule visualization
    ax1 = fig.add_subplot(grid[0])
    ax1.set_title('H₂ Molecule Separation', fontsize=14)
    
    # Second subplot: Energy curve
    ax2 = fig.add_subplot(grid[1])
    ax2.set_title('Potential Energy Curve', fontsize=14)
    
    # Animation function
    def animate(i):
        # Update molecular visualization and energy curve
        # ... animation code ...
        
    # Create the animation
    anim = animation.FuncAnimation(fig, animate, frames=len(distances)*2-2, 
                                 interval=100, blit=False)
    
    # Save animation
    anim.save(save_path, writer='pillow', fps=10, dpi=100)
```

### 5. Molecular Orbital Visualization
Visualizes the atomic and molecular orbitals of the H₂ molecule.

```python
def create_molecular_orbital_visualization(save_path):
    """
    Create a static visualization of H2 molecular orbitals
    """
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
    bonding_mo = h1_1s + h2_1s
    antibonding_mo = h1_1s - h2_1s
    
    # Plot orbitals
    # ... plotting code ...
    
    # Save the figure
    plt.savefig(save_path, dpi=300)
```

### 6. VQE Energy Curves Visualization
Creates a visualization of the VQE optimization process showing multiple energy curves using custom color palettes.

```python
def create_vqe_energy_curves_visualization(save_path):
    """
    Create a static visualization of the VQE optimization process
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Custom color palettes using seaborn
    energy_cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
    iter_cmap = sns.color_palette("mako", as_cmap=True)
    
    # Define a simple energy function that depends on two parameters
    def energy_function(theta1, theta2):
        return -1.2 - 0.5 * np.exp(-0.5 * (theta1**2 + theta2**2)) + \
              0.1 * (np.sin(2*theta1)**2 + np.sin(2*theta2)**2)
    
    # Simulate optimization trajectory
    num_iterations = 10
    # ... trajectory simulation code ...
    
    # Plot energy landscape
    contour = ax1.contourf(T1, T2, energy_grid, 50, cmap=energy_cmap)
    
    # Plot optimization trajectory with changing colors
    colors = iter_cmap(np.linspace(0, 1, num_iterations))
    for i in range(num_iterations-1):
        ax1.plot(trajectory_x[i:i+2], trajectory_y[i:i+2], 'o-', 
                color=colors[i], linewidth=2, markersize=8)
    
    # Plot energy curves on second subplot
    # ... energy curves plotting code ...
    
    # Save the figure
    plt.savefig(save_path, dpi=300)
```

## Results

The implementation successfully simulates the H₂ molecule and visualizes key quantum chemistry concepts:

1. **H₂ Molecular Orbitals**:

![H2 Molecular Orbitals](https://github.com/IsolatedSingularity/quantum-chemistry-eigensolver/blob/main/visualization/h2_molecular_orbitals.png?raw=true)

This visualization shows the atomic 1s orbitals of individual hydrogen atoms and how they combine to form molecular orbitals. The bonding orbital (lower left) shows constructive interference between atomic orbitals, while the antibonding orbital (lower right) shows destructive interference.

2. **H₂ Dissociation Curve**:

![H2 Dissociation Animation](https://github.com/IsolatedSingularity/quantum-chemistry-eigensolver/blob/main/visualization/h2_dissociation.gif?raw=true)

The animation demonstrates how the H₂ molecule's energy changes as the bond distance varies. At the equilibrium distance (around 0.74 Å), the energy reaches its minimum. As the atoms move either closer (compression) or farther apart (dissociation), the energy increases.

## Caveats

- **Ansatz Limitations**: The simple circuits used in this implementation may not capture all the relevant physics for larger molecules. More sophisticated ansatz designs are needed for scaling beyond H₂.

- **Classical Simulation Constraints**: While this implementation simulates the quantum algorithm classically, actual quantum devices would face issues like noise, decoherence, and measurement errors.

- **Optimization Challenges**: The classical optimization routine might get trapped in local minima, especially for more complex energy landscapes in larger molecules.

- **Basis Set Limitations**: The minimal basis used provides qualitative insights but lacks quantitative accuracy compared to more complete basis sets.

## Next Steps

- [x] Implement more sophisticated ansatz circuits, such as the Unitary Coupled Cluster (UCC) ansatz for better accuracy.
- [x] Extend the implementation to handle larger molecules like LiH, BeH₂, or H₂O.
- [ ] Incorporate noise models to simulate realistic quantum hardware performance.
- [ ] Implement quantum subspace expansion techniques to improve accuracy of excited state calculations.
- [ ] Add support for calculating molecular properties beyond the ground state energy (dipole moments, forces, etc.).

> [!TIP]
> For a more detailed explanation of the VQE algorithm and the Jordan-Wigner transformation, see the PDF in the resources directory.

> [!NOTE]
> This implementation serves as an educational resource for understanding quantum algorithms in chemistry applications rather than as a production-level quantum chemistry tool.
