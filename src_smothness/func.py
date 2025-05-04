import numpy as np
import random

from sklearn.metrics.pairwise import cosine_similarity

def add_random_edges(adj_matrix, n):
    adj_matrix = np.array(adj_matrix)
    adj_matrix = adj_matrix.copy()
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    size = adj_matrix.shape[0]
    edges_added = 0

    while edges_added < n:
        # Randomly select two different nodes
        i, j = random.sample(range(size), 2)

        # Check if the edge doesn't already exist
        if adj_matrix[i, j] == 0:
            adj_matrix[i, j] = 1
            edges_added += 1

    return adj_matrix

def get_num_ones(A):
    A = np.array(A)
    num = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] == 1:
                num += 1
    return num

def check_symmetric(A):
    A = np.array(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != A[j,i]:
                return False
    return True

def shuffle(A):
    A = np.array(A)
    ones = get_num_ones(A)
    A = A.copy()
    rang = A.shape[0] * A.shape[1]
    flags = np.random.choice(rang, size=ones,replace=False)
    new = np.zeros(rang,dtype=np.float32)
    new[flags] = 1
    return new.reshape((A.shape[0],A.shape[1]))


def shuffle_symmetric(A):
    A = np.array(A).copy()
    ones = get_num_ones(A) // 2
    rang = A.shape[0] * A.shape[1]
    flags = np.random.choice(rang, size=ones, replace=False)
    new = np.zeros(rang, dtype=np.float32)
    new[flags] = 1
    new = new.reshape((A.shape[0],A.shape[1]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if new[i, j] == 1:
                new[j,i] = 1
    return new
# Instantiate our model and optimiser

def make_r_regular(adj_matrix, r):
    n = len(adj_matrix)

    adj_matrix = adj_matrix.copy()

    if r >= n:
        raise ValueError("r must be less than the number of vertices.")

    adj_matrix = np.array(adj_matrix)

    # Remove self loops if any
    np.fill_diagonal(adj_matrix, 0)

    degrees = adj_matrix.sum(axis=1)

    # Check feasibility
    # if np.any(degrees > r):
    #     raise ValueError("Some vertices already exceed degree r; cannot make r-regular.")
    if (n * r) % 2 != 0:
        raise ValueError("An r-regular graph must satisfy the condition n*r even.")

    # Create edges until the graph is r-regular
    for i in range(n):
        for j in range(i+1, n):
            if degrees[i] < r and degrees[j] < r and adj_matrix[i, j] == 0:
                adj_matrix[i, j] = adj_matrix[j, i] = 1
                degrees[i] += 1
                degrees[j] += 1

    # if not np.all(degrees == r):
    #     raise ValueError("Unable to make the given graph r-regular with the current structure.")

    return adj_matrix


def prune_edges_to_r(adj_matrix, r):
    n = len(adj_matrix)
    adj_matrix = adj_matrix.copy()
    adj_matrix = np.array(adj_matrix)

    # Remove self loops if any
    np.fill_diagonal(adj_matrix, 0)

    degrees = adj_matrix.sum(axis=1,dtype=int)

    # Remove edges from nodes exceeding r edges
    for i in range(n):
        if degrees[i] > r:
            neighbors = np.where(adj_matrix[i] == 1)[0]
            np.random.shuffle(neighbors)
            edges_to_remove = degrees[i] - r
            for neighbor in neighbors[:edges_to_remove]:
                adj_matrix[i, neighbor] = adj_matrix[neighbor, i] = 0
                degrees[i] -= 1
                degrees[neighbor] -= 1

    return adj_matrix


def compute_smoothness_dense(adj, features):
    D = np.diag(np.sum(adj, axis=1))
    L = D - adj

    numerator = np.trace(features.T @ L @ features)
    denominator = np.trace(features.T @ D @ features)

    smoothness = numerator / denominator
    return smoothness




def alter_dense_adjacency(features, adj, threshold=0.8, mode='increase'):
    sim_matrix = cosine_similarity(features)
    np.fill_diagonal(sim_matrix, 0)  # Ignore self-connections

    adj_modified = adj.copy()

    if mode == 'increase':
        # Add edges between nodes with high similarity
        adj_modified[sim_matrix > threshold] = 1
    elif mode == 'decrease':
        # Add edges between nodes with low similarity
        adj_modified[sim_matrix < threshold] = 1

    # Ensure symmetry
    adj_modified = np.maximum(adj_modified, adj_modified.T)

    # Ensure binary adjacency (0/1)
    adj_modified = (adj_modified > 0).astype(int)

    return adj_modified

import networkx as nx
from itertools import combinations
from tqdm import tqdm

def reduce_cov_by_adding_edges(adj, target_cov=0.1, max_edges_to_add=None, verbose=False):
    """
    Adds edges between low-degree nodes to reduce COV of node degrees.

    Args:
        adj (np.ndarray): Symmetric binary adjacency matrix (n x n).
        target_cov (float): Target coefficient of variation (COV) to stop at.
        max_edges_to_add (int): Optional limit on how many edges to add.
        verbose (bool): Whether to print progress.

    Returns:
        new_adj (np.ndarray): Updated adjacency matrix.
        edge_type (np.ndarray): Same shape as adj. 1 for original edges, 0.5 for added.
    """
    n = adj.shape[0]

    G = nx.from_numpy_array(adj)
    degrees = np.array([deg for _, deg in G.degree()])
    edge_type = adj.copy().astype(float)  # 1.0 for original edges

    num_added = 0
    candidates = list(combinations(range(n), 2))
    np.random.shuffle(candidates)

    for u, v in tqdm(candidates, disable=not verbose):
        if not G.has_edge(u, v):
            # Prefer to connect low-degree nodes
            if degrees[u] < np.mean(degrees) or degrees[v] < np.mean(degrees):
                G.add_edge(u, v)
                edge_type[u, v] = edge_type[v, u] = 0.5  # mark added edge
                degrees[u] += 1
                degrees[v] += 1
                num_added += 1

                cov = np.std(degrees) / np.mean(degrees)
                if verbose and num_added % 10 == 0:
                    print(f"Added {num_added} edges, current COV: {cov:.4f}")

                if cov <= target_cov:
                    break
                if max_edges_to_add and num_added >= max_edges_to_add:
                    break

    new_adj = nx.to_numpy_array(G).astype(int)
    return new_adj, edge_type


from scipy.sparse import csr_matrix, isspmatrix

def compute_cov_from_adjacency(adj):
    """
    Compute the coefficient of variation (COV) of node degrees from an adjacency matrix.

    Args:
        adj (np.ndarray or scipy.sparse matrix): The adjacency matrix.

    Returns:
        float: The COV of node degrees.
    """
    if isspmatrix(adj):
        degrees = np.array(adj.sum(axis=1)).flatten()
    else:
        degrees = np.sum(adj, axis=1)

    mean = np.mean(degrees)
    std = np.std(degrees)

    if mean == 0:
        return 0.0

    cov = std / mean
    return cov




def prune_random_edges(adj_matrix, n):
    """
    Removes n random edges from an undirected graph's adjacency matrix.

    Parameters:
    - adj_matrix: 2D numpy array (adjacency matrix of the graph)
    - n: Number of edges to remove

    Returns:
    - pruned adjacency matrix (as a new numpy array)
    """
    # Make a copy to avoid modifying original
    pruned_matrix = np.array(adj_matrix, copy=True)

    # Get list of all existing edges (upper triangle, to avoid duplicates)
    edge_indices = [(i, j) for i in range(len(pruned_matrix))
                            for j in range(i+1, len(pruned_matrix))
                            if pruned_matrix[i][j] != 0]

    # Check if enough edges exist to prune
    if n > len(edge_indices):
        raise ValueError("Not enough edges to prune.")

    # Randomly select edges to remove
    edges_to_remove = random.sample(edge_indices, n)

    # Remove selected edges
    for i, j in edges_to_remove:
        pruned_matrix[i][j] = 0
        pruned_matrix[j][i] = 0  # Ensure symmetry for undirected graph

    return pruned_matrix


def make_graph_r_regular(adj, r, verbose=False):
    """
    Adds edges to make the graph r-regular while preserving all original edges
    and minimizing COV of node degrees.

    Args:
        adj (np.ndarray): Symmetric binary adjacency matrix (n x n).
        r (int): Target regular degree.
        verbose (bool): Print progress.

    Returns:
        new_adj (np.ndarray): Updated adjacency matrix.
        edge_type (np.ndarray): Matrix with 1.0 for original, 0.5 for added edges.
        cov (float): Final coefficient of variation of degrees (should be 0).
    """
    adj = adj.copy()
    n = adj.shape[0]
    assert adj.shape[0] == adj.shape[1], "Adjacency matrix must be square"
    assert (adj == adj.T).all(), "Adjacency matrix must be symmetric"

    # Check degree constraints
    current_degrees = adj.sum(axis=1)
    if np.max(current_degrees) > r:
        raise ValueError(f"Target r={r} is too small. Current max degree is {np.max(current_degrees)}.")

    if (r * n) % 2 != 0:
        raise ValueError("r * n must be even for a simple undirected r-regular graph.")

    G = nx.from_numpy_array(adj)
    degrees = np.array([deg for _, deg in G.degree()])
    edge_type = adj.copy().astype(float)  # original edges marked as 1.0

    # Build list of nodes that need degree increments
    degree_deficit = r - degrees
    to_fill = {i for i in range(n) if degree_deficit[i] > 0}

    # Add edges iteratively
    while to_fill:
        nodes = list(to_fill)
        added = False

        for u, v in combinations(nodes, 2):
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                edge_type[u, v] = edge_type[v, u] = 0.5
                degree_deficit[u] -= 1
                degree_deficit[v] -= 1
                if degree_deficit[u] == 0: to_fill.discard(u)
                if degree_deficit[v] == 0: to_fill.discard(v)
                added = True
                break  # add one edge at a time

        if not added:
            raise RuntimeError("Failed to complete r-regular graph. Try increasing r or relaxing constraints.")

    new_adj = nx.to_numpy_array(G).astype(int)
    final_degrees = np.array([deg for _, deg in G.degree()])
    cov = np.std(final_degrees) / np.mean(final_degrees)

    if verbose:
        print(f"Final COV: {cov:.6f}, all degrees: {final_degrees[0]}")
    return new_adj
    return new_adj, edge_type, cov