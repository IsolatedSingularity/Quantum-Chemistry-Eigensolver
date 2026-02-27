"""Tests for quantum_chemistry.molecule.h2_molecule — data loading."""

from __future__ import annotations

import os

import numpy as np
import pytest

from quantum_chemistry.molecule.h2_molecule import (
    load_h2_spin_orbital_integral,
    load_h2_spin_orbital_integrals,
)

DATA_PATH = "h2_data"


@pytest.fixture()
def single_file():
    return load_h2_spin_orbital_integral(DATA_PATH, "h2_mo_integrals_d_0750.npz")


class TestLoadSingle:
    def test_returns_four_items(self, single_file):
        assert len(single_file) == 4

    def test_distance_value(self, single_file):
        distance, *_ = single_file
        assert np.isclose(distance, 0.75, atol=0.01)

    def test_one_body_shape(self, single_file):
        _, one_body, _, _ = single_file
        assert one_body.ndim == 2
        assert one_body.shape[0] == one_body.shape[1]  # square

    def test_two_body_shape(self, single_file):
        _, _, two_body, _ = single_file
        assert two_body.ndim == 4
        n = two_body.shape[0]
        assert two_body.shape == (n, n, n, n)

    def test_nuclear_repulsion_positive(self, single_file):
        _, _, _, nuc = single_file
        assert nuc > 0


class TestLoadAll:
    def test_loads_all_files(self):
        distances, data = load_h2_spin_orbital_integrals(DATA_PATH)
        npz_count = sum(1 for f in os.listdir(DATA_PATH) if f.endswith(".npz"))
        assert len(distances) == npz_count
        assert len(data) == npz_count

    def test_distances_sorted(self):
        distances, _ = load_h2_spin_orbital_integrals(DATA_PATH)
        assert distances == sorted(distances)

    def test_distance_range(self):
        distances, _ = load_h2_spin_orbital_integrals(DATA_PATH)
        assert distances[0] >= 0.2
        assert distances[-1] <= 2.0
