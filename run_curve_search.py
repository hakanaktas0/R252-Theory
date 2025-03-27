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
from src.models import CurveFinder
from src.utils import *


LEARNED_MATRIX_ID = 'adj_nonpos_wd0001'
INIT_MIDPOINT = True


cora_data = CoraDataset()
train_x, train_y, valid_x, valid_y, test_x, test_y = cora_data.train_val_test_split()

cora_adj = cora_data.get_adjacency_matrix()

learned_adj_matrix = torch.from_numpy(np.load(f'adj_matrices/{LEARNED_MATRIX_ID}/learned_adjacency.npy'))

model = CurveFinder(input_dim=train_x.shape[-1], output_dim=7, adj_1=cora_adj, adj_2=learned_adj_matrix, init_midpoint=INIT_MIDPOINT)

W_params = torch.load(f'adj_matrices/{LEARNED_MATRIX_ID}/linear_weights.pt')

model.linear.load_state_dict(W_params)

X = cora_data.get_fullx()
train_mask = cora_data.train_mask
valid_mask = cora_data.valid_mask
test_mask = cora_data.test_mask

train_stats_gnn_cora = train_eval_loop_gnn_cora(model, X, train_y, train_mask,
                                          X, valid_y, valid_mask,
                                          X, test_y, test_mask, weight_decay=0
                                       )

np.save(f'adj_matrices/{LEARNED_MATRIX_ID}/theta_param.npy', model.theta_param.numpy())