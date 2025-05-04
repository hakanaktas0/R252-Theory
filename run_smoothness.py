#@title [RUN] Set random seed for deterministic results
import random
import torch
from src_smothness.func import *


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

from  smoothness import *
from src_smothness.dataset import CoraDataset
from src_smothness.models import SimpleGNN
from src_smothness.utils import *

# Lets download our cora dataset and get the splits
cora_data = CoraDataset()
train_x, train_y, valid_x, valid_y, test_x, test_y = cora_data.train_val_test_split()

# Always check and confirm our data shapes match our expectations
print(f"Train shape x: {train_x.shape}, y: {train_y.shape}")
print(f"Val shape x: {valid_x.shape}, y: {valid_y.shape}")
print(f"Test shape x: {test_x.shape}, y: {test_y.shape}")

import pickle
#
with open('stats_make_random_r_regular_1_layer.pkl','rb') as f:
    make_random_r_regular_1_layer = pickle.load(f)

for k in make_random_r_regular_1_layer.keys():
    make_random_r_regular_1_layer[k]['av_diric'] = make_random_r_regular_1_layer[k]['dirichlet_energy'] / (make_random_r_regular_1_layer[k]['final_num_edges']/2)
    make_random_r_regular_1_layer[k]['av_ray'] = make_random_r_regular_1_layer[k]['rayleigh_quotient'] / (make_random_r_regular_1_layer[k]['final_num_edges']/2)


with open('stats_prune_edges_to_r_1_layer.pkl','rb') as f:
    prune_edges_to_r_1_layer = pickle.load(f)


for k in prune_edges_to_r_1_layer.keys():
    prune_edges_to_r_1_layer[k]['av_diric'] = prune_edges_to_r_1_layer[k]['dirichlet_energy'] / (prune_edges_to_r_1_layer[k]['final_num_edges']/2)
    prune_edges_to_r_1_layer[k]['av_ray'] = prune_edges_to_r_1_layer[k]['rayleigh_quotient'] / (prune_edges_to_r_1_layer[k]['final_num_edges']/2)


with open('stats_add_random_edges_1_layer.pkl','rb') as f:
    add_random_edges_1_layer = pickle.load(f)


for k in add_random_edges_1_layer.keys():
    add_random_edges_1_layer[k]['av_diric'] = add_random_edges_1_layer[k]['dirichlet_energy'] / (add_random_edges_1_layer[k]['final_num_edges']/2)
    add_random_edges_1_layer[k]['av_ray'] = add_random_edges_1_layer[k]['rayleigh_quotient'] / (add_random_edges_1_layer[k]['final_num_edges']/2)




with open('stats_add_cov_edges_1_layer.pkl','rb') as f:
    add_cov_edges_1_layer = pickle.load(f)


for k in add_cov_edges_1_layer.keys():
    add_cov_edges_1_layer[k]['av_diric'] = add_cov_edges_1_layer[k]['dirichlet_energy'] / (add_cov_edges_1_layer[k]['final_num_edges']/2)
    add_cov_edges_1_layer[k]['av_ray'] = add_cov_edges_1_layer[k]['rayleigh_quotient'] / (add_cov_edges_1_layer[k]['final_num_edges']/2)



with open('stats_prune_make_cov_r_regular_1_layer.pkl','rb') as f:
    stats_prune_make_cov_r_regular_1_layer = pickle.load(f)

for k in stats_prune_make_cov_r_regular_1_layer.keys():
    stats_prune_make_cov_r_regular_1_layer[k]['av_diric'] = stats_prune_make_cov_r_regular_1_layer[k]['dirichlet_energy'] / (2708 * k /2)
    stats_prune_make_cov_r_regular_1_layer[k]['av_ray'] = stats_prune_make_cov_r_regular_1_layer[k]['rayleigh_quotient'] / (2708 * k /2)



with open('stats_add_smooth_edges_1_layer.pkl','rb') as f:
    stats_add_smooth_edges_1_layer = pickle.load(f)
#

for k in stats_add_smooth_edges_1_layer.keys():
    stats_add_smooth_edges_1_layer[k]['av_diric'] = stats_add_smooth_edges_1_layer[k]['dirichlet_energy'] / (stats_add_smooth_edges_1_layer[k]['final_num_edges']/2)
    stats_add_smooth_edges_1_layer[k]['av_ray'] = stats_add_smooth_edges_1_layer[k]['rayleigh_quotient'] / (stats_add_smooth_edges_1_layer[k]['final_num_edges']/2)



with open('stats_prune_smooth_edges_1_layer.pkl','rb') as f:
    stats_prune_smooth_edges_1_layer = pickle.load(f)

for k in stats_prune_smooth_edges_1_layer.keys():
    stats_prune_smooth_edges_1_layer[k]['av_diric'] = stats_prune_smooth_edges_1_layer[k]['dirichlet_energy'] / (stats_prune_smooth_edges_1_layer[k]['final_num_edges']/2)
    stats_prune_smooth_edges_1_layer[k]['av_ray'] = stats_prune_smooth_edges_1_layer[k]['rayleigh_quotient'] / (stats_prune_smooth_edges_1_layer[k]['final_num_edges']/2)


a = 2




A = cora_data.get_adjacency_matrix()

X = cora_data.get_fullx()
# model = SimpleGNN(input_dim=train_x.shape[-1], output_dim=7, A=A, hidden_dim=train_x.shape[-1], num_gcn_layers=1)
train_mask = cora_data.train_mask
valid_mask = cora_data.valid_mask
test_mask = cora_data.test_mask



# A = torch.ones((A.shape),dtype=torch.float32)
print(get_num_ones(A))
print(check_symmetric(A))
# A = shuffle_symmetric(A)
A = np.array(A)
feat = cora_data.cora_data.x.detach().numpy()
stats = {}


for i in range(1,6):
    print('edges before : ', get_num_ones(A))
    init_edges = get_num_ones(A)
    print('COV before : ' ,compute_cov_from_adjacency(A))
    print('energy before : ' , dirichlet_energy(A,X))
    init_energy = dirichlet_energy(A,X)
    init_cov = compute_cov_from_adjacency(A)
    A_new = prune_edges_homophily(A,X,num_edges=i*500)
    # A_new = prune_edges_to_r(A,i)
    # A_new = add_random_edges(A.copy(), 1000 * i)

    # try:
        # A_new = rewire_edges_smoothness(A,X,num_edges=50*i)
        # A_new = adjust_smoothness_homophily(A,X,increase=True,num_edges=1000*i)
        # A_new = make_r_regular(A, i)
        # print('Success : ' , i)
    # except:
    #     print('tried ', i)
    #     continue
    # A_new = prune_edges_to_r(A, i)
    # A_new = make_r_regular(A_new, i)
    # A_new = adjust_smoothness_dirichlet(A,X,increase=True,num_edges=100*i)
    # A_new = add_random_edges(A.copy(),1000 * i)

    # A_new, _  = reduce_cov_by_adding_edges(A,target_cov=compute_cov_from_adjacency(A) - 0.1 * i)


    A_new = A_new.astype(np.float32)
    print('edges after : ', get_num_ones(A_new))
    final_edges = get_num_ones(A)
    print('COV after : ', compute_cov_from_adjacency(A_new))
    print('energy after : ', dirichlet_energy(A_new,X))
    final_energy = dirichlet_energy(A_new,X)
    final_cov = compute_cov_from_adjacency(A_new)


    # print(dirichlet_energy(A_new,feat))
    model = SimpleGNN(input_dim=train_x.shape[-1], output_dim=7, A=torch.tensor(A_new), hidden_dim=train_x.shape[-1], num_gcn_layers=1)
    train_stats_gnn_cora = train_eval_loop_gnn_cora(model, X, train_y, train_mask,
                                                    X, valid_y, valid_mask,
                                                    X, test_y, test_mask
                                                    )
    train_stats_gnn_cora['dirichlet_energy'] = dirichlet_energy(A_new,feat)
    train_stats_gnn_cora['feature_homophily_ratio'] = feature_homophily_ratio(A_new,feat)
    train_stats_gnn_cora['local_variation_smoothness'] = local_variation_smoothness(A_new,feat)
    train_stats_gnn_cora['rayleigh_quotient'] = rayleigh_quotient(A_new,feat)
    train_stats_gnn_cora['avg_neighbor_cosine_similarity'] = avg_neighbor_cosine_similarity(A_new,feat)
    train_stats_gnn_cora['init_num_edges'] = init_edges
    train_stats_gnn_cora['final_num_edges'] = final_edges
    train_stats_gnn_cora['init_energy'] = init_energy
    train_stats_gnn_cora['final_energy'] = final_energy
    train_stats_gnn_cora['init_cov'] = init_cov
    train_stats_gnn_cora['final_cov'] = final_cov
    stats[i] = train_stats_gnn_cora


# print('edges before : ', get_num_ones(A))
# init_edges = get_num_ones(A)
# print('COV before : ' ,compute_cov_from_adjacency(A))
# print('energy before : ' , dirichlet_energy(A,X))
# init_energy = dirichlet_energy(A,X)
# init_cov = compute_cov_from_adjacency(A)
# # A_new = prune_edges_to_r(A,i)
# # A_new = add_random_edges(A.copy(), 1000 * i)
#
# try:
#     A_new = prune_edges_to_r(A, 169)
#     A_new = make_r_regular(A_new, 169)
#     print('Success : ' , 169)
# except:
#     print('tried ', 169)
#     # continue
# # A_new = prune_edges_to_r(A, i)
# # A_new = make_r_regular(A_new, i)
# # A_new = adjust_smoothness_dirichlet(A,X,increase=True,num_edges=100*i)
# # A_new = add_random_edges(A.copy(),1000 * i)
#
# # A_new, _  = reduce_cov_by_adding_edges(A,target_cov=compute_cov_from_adjacency(A) - 0.1 * i)
#
#
# A_new = A_new.astype(np.float32)
# print('edges after : ', get_num_ones(A_new))
# final_edges = get_num_ones(A)
# print('COV after : ', compute_cov_from_adjacency(A_new))
# print('energy after : ', dirichlet_energy(A_new,X))
# final_energy = dirichlet_energy(A_new,X)
# final_cov = compute_cov_from_adjacency(A_new)
#
#
# # print(dirichlet_energy(A_new,feat))
# model = SimpleGNN(input_dim=train_x.shape[-1], output_dim=7, A=torch.tensor(A_new), hidden_dim=train_x.shape[-1], num_gcn_layers=1)
# train_stats_gnn_cora = train_eval_loop_gnn_cora(model, X, train_y, train_mask,
#                                                 X, valid_y, valid_mask,
#                                                 X, test_y, test_mask
#                                                 )
# train_stats_gnn_cora['dirichlet_energy'] = dirichlet_energy(A_new,feat)
# train_stats_gnn_cora['feature_homophily_ratio'] = feature_homophily_ratio(A_new,feat)
# train_stats_gnn_cora['local_variation_smoothness'] = local_variation_smoothness(A_new,feat)
# train_stats_gnn_cora['rayleigh_quotient'] = rayleigh_quotient(A_new,feat)
# train_stats_gnn_cora['avg_neighbor_cosine_similarity'] = avg_neighbor_cosine_similarity(A_new,feat)
# train_stats_gnn_cora['init_num_edges'] = init_edges
# train_stats_gnn_cora['final_num_edges'] = final_edges
# train_stats_gnn_cora['init_energy'] = init_energy
# train_stats_gnn_cora['final_energy'] = final_energy
# train_stats_gnn_cora['init_cov'] = init_cov
# train_stats_gnn_cora['final_cov'] = final_cov
# stats[169] = train_stats_gnn_cora


# with open('stats_prune_smooth_edges_1_layer.pkl', 'wb') as f:
#     pickle.dump(stats, f)
exit()


print(dirichlet_energy(add_random_edges(A.copy(),2000),feat))
print(dirichlet_energy(add_random_edges(A.copy(),3000),feat))
print(dirichlet_energy(add_random_edges(A.copy(),4000),feat))
print(dirichlet_energy(add_random_edges(A.copy(),5000),feat))
print(dirichlet_energy(add_random_edges(A.copy(),6000),feat))
print(dirichlet_energy(add_random_edges(A.copy(),7000),feat))
print(dirichlet_energy(add_random_edges(A.copy(),8000),feat))
print(dirichlet_energy(add_random_edges(A.copy(),9000),feat))
print(dirichlet_energy(add_random_edges(A.copy(),10000),feat))

A_2 = prune_edges_to_r(A,2)
A_3 = prune_edges_to_r(A,3)
A_4 = prune_edges_to_r(A,4)
A_5 = prune_edges_to_r(A,5)


# A = prune_edges_to_r(A,2)
# A = make_r_regular(A,5)
# A = add_random_edges(A,5000)
print(get_num_ones(A))
A = np.array(A)
features = cora_data.cora_data.x.detach().cpu().numpy()
# np.fill_diagonal(A, 1)
compute_smoothness_dense(np.array(alter_dense_adjacency(features,A)),features)
A = torch.tensor(A,dtype=torch.float32)




# Run training loop
# train_stats_gnn_cora = train_eval_loop_gnn_cora(model, X, train_y, train_mask,
#                                           X, valid_y, valid_mask,
#                                           X, test_y, test_mask
#                                        )
# plot_stats(train_stats_gnn_cora, name="GNN_Cora")

# Linear elementwise multiplication to force sparsity.


