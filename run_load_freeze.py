#@title [RUN] Set random seed for deterministic results
import random
import torch
from src.func import *
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


from src.dataset import CoraDataset
from src.models import SimpleGNN, SimpleLinearGNN
from src.utils import *

INITIALIZE_TRUE_ADJ = True
ADJ_POSITIVE = False
WEIGHT_DECAY = 0.001
EXPERIMENT_NAME = 'adj_nonpos_wd0001'

# Lets download our cora dataset and get the splits
cora_data = CoraDataset()
train_x, train_y, valid_x, valid_y, test_x, test_y = cora_data.train_val_test_split()

# Always check and confirm our data shapes match our expectations
print(f"Train shape x: {train_x.shape}, y: {train_y.shape}")
print(f"Val shape x: {valid_x.shape}, y: {valid_y.shape}")
print(f"Test shape x: {test_x.shape}, y: {test_y.shape}")

A = cora_data.get_adjacency_matrix()

X = cora_data.get_fullx()
model = SimpleGNN(input_dim=train_x.shape[-1], output_dim=7, A=A, hidden_dim=train_x.shape[-1], num_gcn_layers=1)
train_mask = cora_data.train_mask
valid_mask = cora_data.valid_mask
test_mask = cora_data.test_mask

# Run training loop
train_stats_gnn_cora = train_eval_loop_gnn_cora(model, X, train_y, train_mask,
                                          X, valid_y, valid_mask,
                                          X, test_y, test_mask, weight_decay=0
                                       )
plot_stats(train_stats_gnn_cora, name="GNN_Cora")

matrix_save_dir = f'adj_matrices/{EXPERIMENT_NAME}'

new_model = SimpleLinearGNN(input_dim=train_x.shape[-1], output_dim=7, A=A, hidden_dim=train_x.shape[-1], num_gcn_layers=1, initialize_true_adj=INITIALIZE_TRUE_ADJ, adj_positive=ADJ_POSITIVE)

new_model.load_weights(model.state_dict(),A)

for name, param in new_model.named_parameters():
    if name == "gcn_layers.0.linear.weight":
        param.requires_grad = False

train_stats_gnn_cora = train_eval_loop_gnn_cora(new_model, X, train_y, train_mask,
                                          X, valid_y, valid_mask,
                                          X, test_y, test_mask, weight_decay=WEIGHT_DECAY
                                       )

new_model.save_adj_matrices(dir=matrix_save_dir)
torch.save(model.gcn_layers[0].linear.state_dict(), f'{matrix_save_dir}/linear_weights.pt')