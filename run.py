#@title [RUN] Set random seed for deterministic results
import random
import torch
import numpy as np
def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed(0)
print("All seeds set.")


from dataset import CoraDataset
from models import SimpleGNN
from utils import *

# Lets download our cora dataset and get the splits
cora_data = CoraDataset()
train_x, train_y, valid_x, valid_y, test_x, test_y = cora_data.train_val_test_split()

# Always check and confirm our data shapes match our expectations
print(f"Train shape x: {train_x.shape}, y: {train_y.shape}")
print(f"Val shape x: {valid_x.shape}, y: {valid_y.shape}")
print(f"Test shape x: {test_x.shape}, y: {test_y.shape}")


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




A = cora_data.get_adjacency_matrix()




# A = torch.ones((A.shape),dtype=torch.float32)
print(get_num_ones(A))
print(check_symmetric(A))
# A = shuffle_symmetric(A)


A = prune_edges_to_r(A,2)
print(get_num_ones(A))
A = np.array(A)
np.fill_diagonal(A, 1)
A = torch.tensor(A,dtype=torch.float32)
X = cora_data.get_fullx()
model = SimpleGNN(input_dim=train_x.shape[-1], output_dim=7, A=A, hidden_dim=train_x.shape[-1], num_gcn_layers=1)
train_mask = cora_data.train_mask
valid_mask = cora_data.valid_mask
test_mask = cora_data.test_mask

# Run training loop
train_stats_gnn_cora = train_eval_loop_gnn_cora(model, X, train_y, train_mask,
                                          X, valid_y, valid_mask,
                                          X, test_y, test_mask
                                       )
plot_stats(train_stats_gnn_cora, name="GNN_Cora")