from typing import Tuple

import numpy as np
import rustworkx as rx
from numpy.typing import NDArray


def simultaneous_eig(matrices: Tuple[NDArray[np.complex128], ...], atol: float = 1e-9):
    """
    Simultanously diagonlized matrices. These must commute between themselves for this procedure to have a solution.

    Args:
        matrices (Tuple[NDArray[np.complex128], ...]): _description_
        atol (float, optional): _description_. Defaults to 1e-9.

    Returns:
        _type_: _description_
    """

    num_matrices = len(matrices)
    size = matrices[0].shape[0]

    eig_values = np.zeros((num_matrices, size), dtype=complex)

    eig_values[0, :], eig_vectors = np.linalg.eig(matrices[0])
    eig_vectors, _ = np.linalg.qr(eig_vectors)
    eig_vectors = eig_vectors.astype(complex)
    for i in range(1, num_matrices):
        blocked_matrix = np.linalg.inv(eig_vectors) @ matrices[i] @ eig_vectors
        blocks, indices = block_diagonal_to_blocks(blocked_matrix, threshold=atol)
        for block, ind in zip(blocks, indices):
            if len(ind) > 1:
                _, sub_eig_vectors = np.linalg.eig(block)
                sub_eig_vectors, _ = np.linalg.qr(sub_eig_vectors)
                eig_vectors[:, ind] = eig_vectors[:, ind] @ sub_eig_vectors

        diag_matrix = np.linalg.inv(eig_vectors) @ matrices[i] @ eig_vectors
        eig_values[i, :] = diag_matrix.diagonal()

    if np.all(np.isclose(np.imag(eig_values), 0)):
        eig_values = np.real(eig_values)

    if np.all(np.isclose(np.imag(eig_vectors), 0)):
        eig_vectors = np.real(eig_vectors)

    return eig_values, eig_vectors


def block_diagonal_to_blocks(matrix: NDArray[np.complex128], threshold: float = 1e-9):
    """
    Find disconnected blocks in a square matrix

    Args:
        matrix (NDArray[np.complex128]): _description_
        threshold (float, optional): _description_. Defaults to 1e-9.

    Returns:
        _type_: _description_
    """

    edges = list(zip(*np.where(np.triu(np.abs(matrix) > threshold, k=1))))

    graph = rx.PyGraph()
    graph.add_nodes_from(range(matrix.shape[0]))
    graph.add_edges_from_no_data(edges)

    blocks = list()
    indices = list()
    for component in rx.connected_components(graph):
        index = list(component)
        blocks.append(matrix[np.ix_(index, index)])
        indices.append(index)

    return blocks, indices
