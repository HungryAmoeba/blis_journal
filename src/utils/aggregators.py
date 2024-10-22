import torch

class GraphMomentAggregator(torch.nn.Module):
    def __init__(self, S=4):
        super(GraphMomentAggregator, self).__init__()
        self.S = S

    def forward(self, x, edge_index):
        # x has shape [num_outputs, num_nodes, K, M, num_features]
        # want output to have shape [num_outputs, S, K, M, num_features]

        # Compute the moments for each node
        graph_moments = torch.zeros(x.size(0), self.S, x.size(2), x.size(3), x.size(4))
        for s in range(1, self.S+1):
            graph_moments[:, s-1, :, :, :] = self._get_graph_moment(x, s)
        return graph_moments

    def _get_graph_moment(self, x, s):
        # x has shape [num_outputs, num_nodes, K, M, num_features]
        # want output to have shape [num_outputs, K, M, num_features]
        return x.abs().pow(s).sum(dim=1)