import torch
import numpy as np
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import os
import pytorch_lightning as pl

#from blis import DATA_DIR

# Helper functions to convert adjacency matrix to edge indices and create dataset
def adjacency_to_edge_indices(A):
    edge_indices = torch.nonzero(A, as_tuple=False).t()
    return edge_indices

def create_dataset(X, y, A, transform=None):
    edge_index = adjacency_to_edge_indices(torch.Tensor(A))
    edge_weight = torch.Tensor(A[edge_index[0], edge_index[1]])
    data_list = []

    for i in range(X.shape[0]):
        num_nodes = X[i].shape[0]
        data = Data(
            x=torch.Tensor(X[i]), 
            edge_index=edge_index, 
            y=torch.Tensor([y[i]]).long(), 
            num_nodes=num_nodes, 
            edge_weight=edge_weight
        )

        if transform is not None:
            data = transform(data)
        data_list.append(data)

    return data_list

class TrafficDataModule(pl.LightningDataModule):
    def __init__(self, seed, subdata_type, task_type, batch_size=32, transform=None):
        super().__init__()
        self.seed = seed
        self.subdata_type = subdata_type
        self.task_type = task_type
        self.batch_size = batch_size
        self.transform = transform

        self.label_path = os.path.join(DATA_DIR, "traffic", subdata_type, task_type, "label.npy")
        self.graph_path = os.path.join(DATA_DIR, "traffic", subdata_type, "adjacency_matrix.npy")
        self.signal_path = os.path.join(DATA_DIR, "traffic", subdata_type, "graph_signals.npy")

        self.data = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_data(self):
        # Load data once and prepare it for future splits
        X = np.load(self.signal_path)
        y = np.load(self.label_path)
        A = np.load(self.graph_path)
        self.data = create_dataset(X, y, A, transform=self.transform)

    def setup(self, stage=None):
        # Perform data splits
        train_idx, test_idx = train_test_split(
            np.arange(len(self.data)), test_size=0.3, random_state=self.seed
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.1, random_state=self.seed
        )

        self.train_ds = [self.data[i] for i in train_idx]
        self.val_ds = [self.data[i] for i in val_idx]
        self.test_ds = [self.data[i] for i in test_idx]

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)

    def num_classes(self):
        y = np.load(self.label_path)
        return len(np.unique(y))

class TrafficScatteringDataModule(pl.LightningDataModule):
    def __init__(self, seed, subdata_type, task_type, scattering_dict=None, ignore_graph=False, batch_size=32):
        super().__init__()
        self.seed = seed
        self.subdata_type = subdata_type
        self.task_type = task_type
        self.scattering_dict = scattering_dict
        self.ignore_graph = ignore_graph
        self.batch_size = batch_size

        self.label_path = os.path.join(DATA_DIR, "traffic", subdata_type, task_type, "label.npy")
        self.graph_signal_path = os.path.join(DATA_DIR, "traffic", subdata_type, "graph_signals.npy")
        self.X_train_val = None
        self.y_train_val = None
        self.X_test = None
        self.y_test = None

    def prepare_data(self):
        if self.ignore_graph:
            X = np.load(self.graph_signal_path)
        else:
            layer_paths = [
                os.path.join(DATA_DIR, "traffic", self.subdata_type, "processed",
                             self.scattering_dict["scattering_type"],
                             self.scattering_dict["wavelet_type"],
                             self.scattering_dict["scale_type"],
                             f"layer_{layer}")
                for layer in self.scattering_dict["layers"]
            ]
            moments = []
            for layer_path in layer_paths:
                for moment in self.scattering_dict["moments"]:
                    moments.append(np.load(os.path.join(layer_path, f"moment_{moment}.npy")))
            X = np.concatenate(moments, 1)
        
        y = np.load(self.label_path)

        # Split data into train/validation and test sets
        train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.3, random_state=self.seed)
        val_idx, test_idx = train_test_split(val_idx, test_size=0.5, random_state=self.seed)

        self.X_train_val = np.concatenate((X[train_idx], X[val_idx]), 0)
        self.y_train_val = np.concatenate((y[train_idx], y[val_idx]), 0)
        self.X_test = X[test_idx]
        self.y_test = y[test_idx]

    def setup(self, stage=None):
        pass  # Data is already split in `prepare_data`, nothing else to do here

    def train_dataloader(self):
        train_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(self.X_train_val), torch.Tensor(self.y_train_val).long()
        )
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        test_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(self.X_test), torch.Tensor(self.y_test).long()
        )
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def num_classes(self):
        y = np.load(self.label_path)
        return len(np.unique(y))


# Example usage in PyTorch Lightning Trainer:
if __name__ == "__main__":
    # Set up the DataModule
    traffic_dm = TrafficDataModule(seed=42, subdata_type="PEMS04", task_type="DAY", batch_size=32)
    traffic_scattering_dm = TrafficScatteringDataModule(
        seed=42, 
        subdata_type="PEMS04", 
        task_type="DAY", 
        scattering_dict={
            "scattering_type": "blis", 
            "wavelet_type": "some_wavelet",  # Replace with appropriate wavelet type
            "scale_type": "largest_scale_4", 
            "layers": [1], 
            "moments": [1, 2]
        },
        ignore_graph=False,
        batch_size=32
    )

    # Prepare and setup the data
    traffic_dm.prepare_data()
    traffic_dm.setup()

    traffic_scattering_dm.prepare_data()
    traffic_scattering_dm.setup()

    # Optionally, you can now pass the DataModule into a PyTorch Lightning Trainer
    # trainer = pl.Trainer(max_epochs=10)
    # trainer.fit(model, datamodule=traffic_dm)
