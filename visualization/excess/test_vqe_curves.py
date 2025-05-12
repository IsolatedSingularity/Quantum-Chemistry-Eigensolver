"""
Test script to run VQE energy curves visualization
"""

import os
from vqe_energy_curves import create_vqe_energy_curves_visualization

output_dir = os.path.dirname(os.path.abspath(__file__))
create_vqe_energy_curves_visualization(os.path.join(output_dir, 'vqe_energy_curves_test.png'))
