import numpy as np
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.nn import Linear
from torch_scatter import scatter_mean
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
import torch_geometric
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn import GCNConv

from lightning import LightningModule
from components.moment_agg import GraphMomentAggregator

class IdentityModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, add_self_loops=False, dtype=None):

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, 1, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    
    return edge_index, deg_inv_sqrt[col] * edge_weight

class LazyLayer(torch.nn.Module):
    
    """ Currently a single elementwise multiplication with one laziness parameter per
    channel. this is run through a softmax so that this is a real laziness parameter
    """

    def __init__(self, n):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.Tensor(2, n))

    def forward(self, x, propogated):
        inp = torch.stack((x, propogated), dim=1)
        s_weights = torch.nn.functional.softmax(self.weights, dim=0)
        return torch.sum(inp * s_weights, dim=-2)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weights)

class Diffuse(MessagePassing):

    """ Implements low pass walk with optional weights
    """

    def __init__(self, in_channels, out_channels, trainable_laziness=False, fixed_weights=True):

        super().__init__(aggr="add",  flow = "target_to_source", node_dim=-3)  # "Add" aggregation.
        assert in_channels == out_channels
        self.trainable_laziness = trainable_laziness
        self.fixed_weights = fixed_weights
        if trainable_laziness:
            self.lazy_layer = LazyLayer(in_channels)
        if not self.fixed_weights:
            self.lin = torch.nn.Linear(in_channels, out_channels)


    def forward(self, x, edge_index, edge_weight=None):

        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 2: Linearly transform node feature matrix.
        # turn off this step for simplicity
        if not self.fixed_weights:
            x = self.lin(x)

        # Step 3: Compute normalization
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(self.node_dim), dtype=x.dtype)

        # Step 4-6: Start propagating messages.
        propogated = self.propagate(
            edge_index, edge_weight=edge_weight, size=None, x=x,
        )
        if not self.trainable_laziness:
            return 0.5 * (x + propogated), edge_index, edge_weight

        return self.lazy_layer(x, propogated), edge_index, edge_weight

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

class Blis(torch.nn.Module):

    def __init__(self, in_channels, trainable_laziness=False, trainable_scales = False, activation = "blis"):

        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.in_channels = in_channels
        self.trainable_laziness = trainable_laziness
        self.diffusion_layer1 = Diffuse(in_channels, in_channels, trainable_laziness)
        # self.diffusion_layer2 = Diffuse(
        #     4 * in_channels, 4 * in_channels, trainable_laziness
        # )
        self.wavelet_constructor = torch.nn.Parameter(torch.tensor([
            [1, -1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ], requires_grad=trainable_scales))

        if activation == "blis":
            self.activations = [lambda x: torch.relu(x), lambda x: torch.relu(-x)]
        elif activation == None:
            self.activations = [lambda x : x]

    def forward(self, data):

        """ This performs  Px with P = 1/2(I + AD^-1) (column stochastic matrix) at the different scales"""

        x, edge_index = data.x, data.edge_index
        if len(x.shape) == 1:
            s0 = x[:, None, None]
        else:
            s0 = x[:,:,None]
        avgs = [s0]
        for i in range(16):
            avgs.append(self.diffusion_layer1(avgs[-1], edge_index)[0])
        for j in range(len(avgs)):
            avgs[j] = avgs[j][None, :, :, :]  # add an extra dimension to each tensor to avoid data loss while concatenating TODO: is there a faster way to do this?
        
        # Combine the diffusion levels into a single tensor.
        diffusion_levels = torch.cat(avgs)
        
        # Reshape the 3d tensor into a 2d tensor and multiply with the wavelet_constructor matrix
        # This simulates the below subtraction:
        # filter0 = avgs[0] - avgs[1]
        # filter1 = avgs[1] - avgs[2] 
        # filter2 = avgs[2] - avgs[4]
        # filter3 = avgs[4] - avgs[8]
        # filter4 = avgs[8] - avgs[16] 
        # filter5 = avgs[16]
        wavelet_coeffs = torch.einsum("ij,jklm->iklm", self.wavelet_constructor, diffusion_levels) # J x num_nodes x num_features x 1
        #subtracted = subtracted.view(6, x.shape[0], x.shape[1]) # reshape into given input shape
        activated = [self.activations[i](wavelet_coeffs) for i in range(len(self.activations))]
        
        s = torch.cat(activated, axis=-1).transpose(1,0)
        
        return s
    
    def out_features(self):
        return 12 * self.in_channels

class BlisNet(LightningModule):
    # this is BlisNet operating at the graph level (i.e. node features get aggregated)
    '''
    Blis net with a fixed layout. The layout is a list of strings, where each string is a layer type.

    Inputs:
    - in_channels: int, the number of input channels
    - hidden_channels: int, the number of hidden channels
    - out_channels: int, the number of output channels
    - edge_in_channels: int, the number of input channels for the edge features
    - trainable_laziness: bool, whether to use trainable laziness parameters
    - graph_agg: nn.Module, the graph aggregation layer
    - layout: list[str], the layout of the network.
    '''
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 edge_in_channels: int,
                 trainable_laziness: bool,
                 num_moments: int,
                 layout: list[str]):
        
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_in_channels = edge_in_channels
        self.trainable_laziness = trainable_laziness
        self.num_moments = num_moments
        self.graph_agg = GraphMomentAggregator(num_moments)

        self.layout = ['id'] + layout
        #self.layers = nn.ModuleList()
        self.layers = [IdentityModule()]
        self.out_dimensions = [in_channels]
        # keep track of which dimensions to include in the output (i.e. we want this to perform like a res-net)
        self.include_in_output_dimensions = [True] # note: at some point this whole list should be user defined

        for layout_ in self.layout[1:]:
            if layout_ == 'blis':
                self.layers.append(Blis(self.out_dimensions[-1], trainable_laziness=trainable_laziness))
                self.out_dimensions.append(self.layers[-1].out_features())
                self.include_in_output_dimensions.append(True)
            elif layout_ == 'gcn':
                self.layers.append(GCNConv(self.out_dimensions[-1], hidden_channels))
                self.out_dimensions.append(hidden_channels)
                self.include_in_output_dimensions.append(False)
            elif layout_ == 'dim_reduction':
                self.layers.append(Linear(self.out_dimensions[-1], hidden_channels))
                self.out_dimensions.append(hidden_channels)
                self.include_in_output_dimensions.append(False)
            elif layout_ == 'blis_w1_exact':
                raise NotImplementedError
            elif layout_ == 'blis_w1_cheby':
                raise NotImplementedError
            elif layout_ == 'blis_w1_lightning_approx':
                raise NotImplementedError

        self.layers = nn.ModuleList(self.layers)

        # output dimension of the module (divided by the number of moments)
        self.module_out_dim = np.array([a* b for a,b in zip(self.out_dimensions, self.include_in_output_dimensions)]).sum()

        in_dim = self.module_out_dim * self.num_moments
        self.batch_norm = BatchNorm(in_dim)
        self.lin1 = Linear(in_dim, in_dim//2 )
        self.mean = global_mean_pool
        self.lin2 = Linear(in_dim//2, in_dim//4)
        self.lin3 = Linear(in_dim//4, out_channels)

        self.act = torch.nn.ReLU()

    def forward(self, data):
        # data.x has shape [num_nodes, num_features] and total number of graphs is data.batch.max().item() + 1
        # get the total output dimension
        num_graphs_batch = data.batch.max().item() + 1

        # initialize the output tensor
        blis_module_output = torch.zeros((num_graphs_batch, self.module_out_dim, self.num_moments))
        output_storage_index = 0
        
        for il, layer in enumerate(self.layers):
            if self.layout[il] == "blis":
                x = layer(data).reshape(data.x.shape[0],-1)
            elif self.layout[il] == "dim_reduction":
                x = layer(data.x)
            elif self.layout[il] == 'id':
                x = data.x
            else:
                x = layer(data.x, data.edge_index)

            # aggregate and store output
            if self.include_in_output_dimensions[il]:
                blis_module_output[:,output_storage_index:output_storage_index + x.shape[1], :] = self.graph_agg(x, data.batch_index)
                output_storage_index += x.shape[1]

            data.x =x

        # reshape to num_graphs, -1
        x = blis_module_output.view(num_graphs_batch, -1)
        x = self.batch_norm(x)
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        x = self.act(x)
        x = self.lin3(x)

        return x
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == '__main__':
    model = BlisNet(1, 1, 1, 1, True, 1, ['blis', 'blis', 'blis'])

    import pdb; pdb.set_trace()

    # Example usage
    num_outputs = 2
    num_nodes = 10

    
