import torch
from torch_geometric.data import Dataset, InMemoryDataset, Data
from tqdm import tqdm
from graphein.ml.conversion import convert_nx_to_pyg_data
import json
import numpy as np
from functools import partial
import multiprocessing
import os

def _mp_graph_constructor(args, features, map_ss):
    result = torch.load(args)
    g = Data(edge_index = result.edge_index, num_nodes = len(result.node_id), node_id = result.node_id,
    edge_attr = torch.cat((torch.FloatTensor(result.distance).reshape(-1,1), torch.tensor(result.ohe_kind)), dim=1), y = result.label, sequence = result.sequence_A,
    name = result.name)
    x = torch.cat((torch.FloatTensor(result.coords[0]), torch.FloatTensor(result.amino_acid_one_hot)), dim=1)
    for feat in features:
        if feat == 'ss':
            feature = np.zeros((x.shape[0],8))
            for i in range(x.shape[0]):
                feature[i][map_ss[result[feat][i]]] = 1      
        else:              
            feature = np.array(result[feat])
            if len(feature.shape)==1:
                feature = feature.reshape(-1,1)
        x = torch.cat((x, torch.FloatTensor(feature)), dim = 1)
    g.x = x
    return g

class ARGDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, features=[], edges=[]):
        with open(os.path.join(root, "list_elements_coala.json"), "r") as fp:
            self.structures = json.load(fp)
        self.edges = edges
        self.features = features
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) :
        """Names of raw files in the dataset."""
        return [f"data_{pdb}t10.pt" for pdb in self.structures]

    @property
    def processed_file_names(self):
        """Names of processed files to look for"""
        feat_name = "_".join(self.features)
        edge_name = "_".join(self.edges)
        return [f'data_{feat_name}_{edge_name}.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass


    def process(self):
        map_ss = {'-':0, 'H':1, 'B':2, 'E':3, 'G':4, 'I':5, 'T':6, 'S':7}
        pool = multiprocessing.Pool(16)
        graphs = []
        y = []
        constructor = partial(
                _mp_graph_constructor, features=self.features, map_ss=map_ss
            )
        for result in tqdm(pool.imap_unordered(constructor, self.raw_paths), total=len(self.raw_paths)):
            graphs.append(result)

        pool.close()
        pool.join()
        print("Converting Networkx graphs to PyG...")

        print("Saving Data...")
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])
        print("Done!")

if __name__=='__main__':
    dataset= ARGDataset('../../PDBFiles', features = ['phi', 'psi','rsa', 'asa', 'b_factor', 'ss','hbond_acceptors', 'hbond_donors', 'expasy'])

