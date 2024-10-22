

import torch
from torch_geometric.utils import k_hop_subgraph
#from torch_geometric.nn import GCNConv

class GraphMomentAggregator(torch.nn.Module):
    def __init__(self, S=4):
        super(GraphMomentAggregator, self).__init__()
        self.S = S
        
    def forward(self, x, batch_index, apply_abs=True):
        # x has shape [num_nodes, num_features]
        # batch_index has shape [num_nodes], indicating the graph assignment for each node
        # We want output to have shape [num_graphs, num_features, S]
        
        # Get the number of graphs from the batch index
        num_graphs = batch_index.max().item() + 1
        num_nodes, num_features = x.size()
        
        # Initialize the output tensor for storing moments
        graph_moments = torch.zeros(num_graphs, num_features, self.S, device=x.device)
        
        # Compute moments for each graph in the batch
        for g in range(num_graphs):
            # Get node indices for the current graph
            mask = (batch_index == g)
            
            # Extract relevant data for the current graph
            x_graph = x[mask]  # Shape [num_nodes_in_graph, num_features]
            
            for s in range(1, self.S + 1):
                # Compute the s-th moment for the current graph
                graph_moments[g, :, s - 1] = self._get_graph_moment(x_graph, s, apply_abs)
                
        return graph_moments
    
    def _get_graph_moment(self, x, s, apply_abs=True):
        # x has shape [num_nodes_in_graph, num_features]
        # Output should have shape [num_features]
        if apply_abs:
            return x.abs().pow(s).sum(dim=0)
        else:
            return x.pow(s).sum(dim=0)


if __name__ == '__main__':
    # Example usage
    num_outputs = 2
    num_nodes = 10
    K = 3
    M = 5
    num_features = 4
    batch_size = 5
    S = 4

    # Create example input
    x = torch.rand((num_outputs, num_nodes, K, M, num_features))
    edge_index = torch.randint(0, num_nodes, (2, num_nodes))
    batch_index = torch.randint(0, batch_size, (num_nodes,))

    # Initialize the aggregator
    aggregator = GraphMomentAggregator(S=S)

    # Compute the graph moments
    graph_moments = aggregator(x, batch_index)

    print("Graph moments shape:", graph_moments.shape)