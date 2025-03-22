# Fill in the initialisation and forward method the GCNLayer below
from torch.nn import Embedding, Linear, ReLU, BatchNorm1d, Module, ModuleList, Sequential
import torch.nn.functional as F
import torch

class GCNLayer(Module):
    """Graph Convolutional Network layer from Kipf & Welling.

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        A (torch.Tensor): 2-D adjacency matrix
    """
    def __init__(self, input_dim, output_dim, A):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A

        # ============ YOUR CODE HERE ==============
        # Sample answer
        # Compute symmetric norm
        I = torch.eye(A.size(0), device=A.device)
        A_hat = A + I
        D_hat = torch.diag(A_hat.sum(dim=1))
        D_inv_sqrt = torch.linalg.inv(torch.sqrt(D_hat))
        self.adj_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

        # + Simple linear transformation and non-linear activation
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.activation = torch.nn.ReLU()
        # ===========================================

    def forward(self, x):
        """Implements the forward pass for the layer

        Args:
            x (torch.Tensor): input node feature matrix
        """
        # ============ YOUR CODE HERE ==============
        # Sample answer
        x = self.adj_norm @ x
        x = self.linear(x)
        x = self.activation(x)
        # ===========================================
        return x


class LinearLayer(Module):
    def __init__(self, input_dim, output_dim, A):
        super(LinearLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A

        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.left_weights = torch.nn.Parameter(torch.randn(self.A.shape[0], self.A.shape[1]))
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        L_tril = torch.tril(self.L)
        symmetric_matrix = L_tril + L_tril.T - torch.diag(torch.diag(L_tril))
        x = symmetric_matrix @ x
        x = self.linear(x)
        x = self.activation(x)
        return x
    
    def load(self, weights):
        weights = 0


class SimpleGNN(Module):
    """A Simple GNN model using the GCNLayer for node classification

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        A (torch.Tensor): 2-D adjacency matrix
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers, A):
        super(SimpleGNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A

        # Note: if a single layer is used hidden_dim should be the same as input_dim
        if num_gcn_layers > 1:
          self.gcn_layers = [GCNLayer(input_dim, hidden_dim, A)]
          self.gcn_layers += [GCNLayer(hidden_dim, hidden_dim, A) for i in range(num_gcn_layers-2)]
          self.gcn_layers += [GCNLayer(hidden_dim, output_dim, A)]
        else:
          self.gcn_layers = [GCNLayer(input_dim, output_dim, A)]

        self.gcn_layers = ModuleList(self.gcn_layers)
        self.num_gcn_layers = num_gcn_layers

    def forward(self, x):
        """Forward pass through SimpleGNN on input x

        Args:
            x (torch.Tensor): input node features
        """
        for j in range(self.num_gcn_layers-1):
          x = self.gcn_layers[j](x)
          x = F.relu(x)

        x = self.gcn_layers[-1](x)

        y_hat = x
        return y_hat






class SimpleLinearGNN(Module):
    """A Simple GNN model using the GCNLayer for node classification

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        A (torch.Tensor): 2-D adjacency matrix
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers, A):
        super(SimpleLinearGNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A

        # Note: if a single layer is used hidden_dim should be the same as input_dim
        if num_gcn_layers > 1:
          self.gcn_layers = [LinearLayer(input_dim, hidden_dim, A)]
          self.gcn_layers += [LinearLayer(hidden_dim, hidden_dim, A) for i in range(num_gcn_layers-2)]
          self.gcn_layers += [LinearLayer(hidden_dim, output_dim, A)]
        else:
          self.gcn_layers = [LinearLayer(input_dim, output_dim, A)]

        self.gcn_layers = ModuleList(self.gcn_layers)
        self.num_gcn_layers = num_gcn_layers

    def forward(self, x):
        """Forward pass through SimpleGNN on input x

        Args:
            x (torch.Tensor): input node features
        """
        for j in range(self.num_gcn_layers-1):
          x = self.gcn_layers[j](x)
          x = F.relu(x)

        x = self.gcn_layers[-1](x)

        y_hat = x
        return y_hat
    def load_weights(self,state_dict,A):
        with torch.no_grad():
            for name, param in state_dict.items():
                # Split key to find layer index and param type
                splits = name.split('.')

                # Check if key is for a layer in ModuleList
                if splits[0] == 'gcn_layers':
                    layer_idx = int(splits[1])  # Get the layer index
                    param_type = splits[2]

                    # Load the param into the appropriate layer
                    getattr(getattr(self.gcn_layers[layer_idx], param_type),splits[3]).copy_(param)
                    getattr(self.gcn_layers[layer_idx], 'left_weights').copy_(A)
    def freeze_weights(self):
        for layer in self.gcn_layers:
            layer.linear.requires_grad = False
