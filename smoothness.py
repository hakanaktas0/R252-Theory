import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


# Dirichlet Energy
def dirichlet_energy(A, X):
    D = np.diag(A.sum(axis=1))
    L = D - A
    return np.trace(X.T @ L @ X)


# Increase/Decrease smoothness based on Dirichlet Energy
def adjust_smoothness_dirichlet(A, X, increase=True, num_edges=1):
    G = nx.from_numpy_array(A)
    potential_edges = list(nx.non_edges(G)) if increase else list(G.edges())
    best_edges = []

    current_energy = dirichlet_energy(A, X)

    for edge in potential_edges:
        A_mod = A.copy()
        if increase:
            A_mod[edge[0], edge[1]] = A_mod[edge[1], edge[0]] = 1
        else:
            A_mod[edge[0], edge[1]] = A_mod[edge[1], edge[0]] = 0
        energy = dirichlet_energy(A_mod, X)
        if (increase and energy < current_energy) or (not increase and energy > current_energy):
            best_edges.append((edge, energy))

    best_edges = sorted(best_edges, key=lambda x: x[1])[:num_edges]
    for edge, _ in best_edges:
        if increase:
            A[edge[0], edge[1]] = A[edge[1], edge[0]] = 1
        else:
            A[edge[0], edge[1]] = A[edge[1], edge[0]] = 0

    return A


# Feature Homophily Ratio
def feature_homophily_ratio(A, X):
    edges = np.array(A.nonzero()).T
    sim = cosine_similarity(X)
    return sim[edges[:, 0], edges[:, 1]].mean()


# Adjust smoothness based on Feature Homophily
def adjust_smoothness_homophily(A, X, increase=True, num_edges=1):
    A = A.copy()
    sim = cosine_similarity(X)
    G = nx.from_numpy_array(A)

    potential_edges = list(nx.non_edges(G)) if increase else list(G.edges())
    scores = []

    for edge in potential_edges:
        score = sim[edge[0], edge[1]]
        scores.append((edge, score))

    scores = sorted(scores, key=lambda x: -x[1] if increase else x[1])[:num_edges]
    for edge, _ in scores:
        if increase:
            A[edge[0], edge[1]] = A[edge[1], edge[0]] = 1
        else:
            A[edge[0], edge[1]] = A[edge[1], edge[0]] = 0

    return A


# Local Variation Smoothness
def local_variation_smoothness(A, X):
    G = nx.from_numpy_array(A)
    total = 0
    for i in G.nodes():
        neighbors = list(G.neighbors(i))
        if neighbors:
            total += np.mean(np.linalg.norm(X[i] - X[neighbors], axis=1))
    return total / G.number_of_nodes()


# Rayleigh Quotient
def rayleigh_quotient(A, X):
    D = np.diag(A.sum(axis=1))
    L = D - A
    return np.trace(X.T @ L @ X) / np.trace(X.T @ X)


# Average Neighbor Cosine Similarity
def avg_neighbor_cosine_similarity(A, X):
    G = nx.from_numpy_array(A)
    total = 0
    for i in G.nodes():
        neighbors = list(G.neighbors(i))
        if neighbors:
            total += cosine_similarity(X[i].reshape(1, -1), X[neighbors]).mean()
    return total / G.number_of_nodes()


# Label-based Homophily Index
def label_homophily(A, labels):
    edges = np.array(A.nonzero()).T
    same_label = labels[edges[:, 0]] == labels[edges[:, 1]]
    return same_label.mean()


# Adjust smoothness based on Label Homophily
def adjust_label_homophily(A, labels, increase=True, num_edges=1):
    G = nx.from_numpy_array(A)
    potential_edges = list(nx.non_edges(G)) if increase else list(G.edges())
    scores = []

    for edge in potential_edges:
        score = labels[edge[0]] == labels[edge[1]]
        scores.append((edge, score))

    scores = sorted(scores, key=lambda x: -x[1] if increase else x[1])[:num_edges]
    for edge, _ in scores:
        if increase:
            A[edge[0], edge[1]] = A[edge[1], edge[0]] = 1
        else:
            A[edge[0], edge[1]] = A[edge[1], edge[0]] = 0

    return A

def prune_edges_dirichlet(A, X, num_edges=1):
    G = nx.from_numpy_array(A)
    edges = list(G.edges())
    current_energy = dirichlet_energy(A, X)
    scores = []

    for edge in edges:
        A_mod = A.copy()
        A_mod[edge[0], edge[1]] = A_mod[edge[1], edge[0]] = 0
        energy = dirichlet_energy(A_mod, X)
        if energy < current_energy:
            scores.append((edge, energy))

    scores.sort(key=lambda x: x[1])
    for edge, _ in scores[:num_edges]:
        A[edge[0], edge[1]] = A[edge[1], edge[0]] = 0

    return A


def prune_edges_local_variation(A, X, num_edges=1):
    G = nx.from_numpy_array(A)
    edges = list(G.edges())
    current_lv = local_variation_smoothness(A, X)
    scores = []

    for edge in edges:
        A_mod = A.copy()
        A_mod[edge[0], edge[1]] = A_mod[edge[1], edge[0]] = 0
        lv = local_variation_smoothness(A_mod, X)
        if lv < current_lv:
            scores.append((edge, lv))

    scores.sort(key=lambda x: x[1])
    for edge, _ in scores[:num_edges]:
        A[edge[0], edge[1]] = A[edge[1], edge[0]] = 0

    return A


def prune_edges_label_homophily(A, labels, num_edges=1):
    G = nx.from_numpy_array(A)
    edges = list(G.edges())
    scores = [(edge, labels[edge[0]] != labels[edge[1]]) for edge in edges]

    scores.sort(key=lambda x: x[1], reverse=True)  # remove edges between different labels
    for edge, diff in scores[:num_edges]:
        if diff:
            A[edge[0], edge[1]] = A[edge[1], edge[0]] = 0

    return A

def prune_edges_homophily(A, X, num_edges=1):
    sim = cosine_similarity(X)
    G = nx.from_numpy_array(A)
    edges = list(G.edges())
    scores = [(edge, sim[edge[0], edge[1]]) for edge in edges]

    scores.sort(key=lambda x: x[1])  # remove edges with lowest similarity
    for edge, _ in scores[:num_edges]:
        A[edge[0], edge[1]] = A[edge[1], edge[0]] = 0

    return A


def rewire_edges_smoothness(A, X, num_edges=1):
    G = nx.from_numpy_array(A)
    current_energy = dirichlet_energy(A, X)

    for _ in range(num_edges):
        edges = list(G.edges())
        non_edges = list(nx.non_edges(G))
        best_improvement = 0
        best_swap = None

        for edge in edges:
            for non_edge in non_edges:
                A_mod = A.copy()
                A_mod[edge[0], edge[1]] = A_mod[edge[1], edge[0]] = 0
                A_mod[non_edge[0], non_edge[1]] = A_mod[non_edge[1], non_edge[0]] = 1
                new_energy = dirichlet_energy(A_mod, X)

                improvement = current_energy - new_energy
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_swap = (edge, non_edge)

        if best_swap:
            edge, non_edge = best_swap
            A[edge[0], edge[1]] = A[edge[1], edge[0]] = 0
            A[non_edge[0], non_edge[1]] = A[non_edge[1], non_edge[0]] = 1
            G = nx.from_numpy_array(A)
            current_energy -= best_improvement

    return A
