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
    if np.any(degrees > r):
        raise ValueError("Some vertices already exceed degree r; cannot make r-regular.")
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



