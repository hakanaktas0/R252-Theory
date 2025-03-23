import numpy as np
import random

import torch


def add_random_edges(adj_matrix, n):
    adj_matrix = np.array(adj_matrix)
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

    return torch.tensor(adj_matrix)

def get_num_ones(A):
    A = A.detach().cpu().numpy()
    num = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] == 1:
                num += 1
    return num

def check_symmetric(A):
    A = A.detach().cpu().numpy()
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != A[j,i]:
                return False
    return True

def shuffle(A):
    ones = get_num_ones(A)
    rang = A.shape[0] * A.shape[1]
    flags = np.random.choice(rang, size=ones,replace=False)
    new = np.zeros(rang,dtype=np.float32)
    new[flags] = 1
    return torch.tensor(new.reshape((A.shape[0],A.shape[1])))


def shuffle_symmetric(A):
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
    return torch.tensor(new)
# Instantiate our model and optimiser

def make_r_regular(adj_matrix, r):
    n = len(adj_matrix)

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

    return torch.tensor(adj_matrix,dtype=torch.float32)


def prune_edges_to_r(adj_matrix, r):
    n = len(adj_matrix)
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

    return torch.tensor(adj_matrix,dtype=torch.float32)


