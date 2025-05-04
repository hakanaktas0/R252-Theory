#@title [RUN] Set random seed for deterministic results
import random
import torch
from src_smothness.func import *
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


from src_smothness.dataset import CoraDataset
from src_smothness.models import MaskedCommonWeightSimpleLinearGNN
from src_smothness.utils import *

# Lets download our cora dataset and get the splits
cora_data = CoraDataset()
train_x, train_y, valid_x, valid_y, test_x, test_y = cora_data.train_val_test_split()

# Always check and confirm our data shapes match our expectations
print(f"Train shape x: {train_x.shape}, y: {train_y.shape}")
print(f"Val shape x: {valid_x.shape}, y: {valid_y.shape}")
print(f"Test shape x: {test_x.shape}, y: {test_y.shape}")



A = cora_data.get_adjacency_matrix()




# A = torch.ones((A.shape),dtype=torch.float32)
print(get_num_ones(A))
print(check_symmetric(A))
# A = shuffle_symmetric(A)
# A = np.array(A)
# A_2 = prune_edges_to_r(A,2)
# A_3 = prune_edges_to_r(A,3)
# A_4 = prune_edges_to_r(A,4)
# A_5 = prune_edges_to_r(A,5)
#

# A = prune_edges_to_r(A,2)
# A = make_r_regular(A,5)
# A = add_random_edges(A,5000)
# print(get_num_ones(A))
# A = np.array(A)
# features = cora_data.cora_data.x.detach().cpu().numpy()
# np.fill_diagonal(A, 1)
# compute_smoothness_dense(np.array(alter_dense_adjacency(features,A)),features)
# A = torch.tensor(A,dtype=torch.float32)


X = cora_data.get_fullx()
model = MaskedCommonWeightSimpleLinearGNN(input_dim=train_x.shape[-1], output_dim=7, A=A, hidden_dim=train_x.shape[-1], num_gcn_layers=1)
train_mask = cora_data.train_mask
valid_mask = cora_data.valid_mask
test_mask = cora_data.test_mask

# Run training loop
train_stats_gnn_cora = train_eval_loop_gnn_cora(model, X, train_y, train_mask,
                                          X, valid_y, valid_mask,
                                          X, test_y, test_mask
                                       )
plot_stats(train_stats_gnn_cora, name="GNN_Cora")

# Linear elementwise multiplication to force sparsity.


