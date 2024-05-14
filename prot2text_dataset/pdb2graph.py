import multiprocessing
import os
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

from torch_geometric.data import Data
import torch

import numpy as np

from .conversion import convert_nx_to_pyg_data
from graphein.protein.config import ProteinGraphConfig, DSSPConfig
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot, meiler_embedding, expasy_protein_scale, hydrogen_bond_acceptor, hydrogen_bond_donor
from graphein.protein.features.nodes.dssp import  phi, psi, asa, rsa, secondary_structure
from graphein.protein.edges.distance import (add_peptide_bonds,
                                             add_hydrogen_bond_interactions,
                                             add_disulfide_interactions,
                                             add_ionic_interactions,
                                             add_delaunay_triangulation,
                                             add_distance_threshold,
                                             add_sequence_distance_edges,
                                             add_k_nn_edges)

from functools import partial
from .graphs import *
from .utils_dataset import *
import os
import sys
import subprocess
import wget

    
class PDB2Graph():
    def __init__(self, root, output_folder, config, n_processors=int(multiprocessing.cpu_count())):
        self.root = root
        self.output_folder = output_folder
        self.map_secondary_structure = {'-':0, 'H':1, 'B':2, 'E':3, 'G':4, 'I':5, 'T':6, 'S':7}
        self.init_ohe_edge_type()
        self.config = config
        self.features = ['phi', 'psi', 'rsa', 'asa', 'ss', 'expasy']
        self.n_processors = n_processors
        self.raw_dir = root
        self.processed_dir = self._processed_dir()
        self.raw_file_names = self._raw_file_names()
        self.processed_file_names = self._processed_file_names()


    def _processed_dir(self):
        #processed_dir = os.path.join(os.path.split(self.root)[0], "processed_new")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        return self.output_folder
        
    def _raw_file_names(self):
        return os.listdir(self.raw_dir)
    
    def _processed_file_names(self):
        return [self.pdb2pathdata(pdb_path.split(".")[0]) for pdb_path in self.raw_file_names]
    
    def create_nx_graph(self, path_to_structure):
        return construct_graph(self.config, pdb_path = path_to_structure)
    
    def create_pyg_graph(self, path_to_structure):
        pyg_graph = convert_nx_to_pyg_data(self.create_nx_graph(path_to_structure))
 
        graph = Data(edge_index = pyg_graph.edge_index, 
                    num_nodes = len(pyg_graph.node_id), 
                    node_id = pyg_graph.node_id, 
                    name = pyg_graph.name[0], 
                    sequence = getattr(pyg_graph, f"sequence_{pyg_graph.chain_id[0]}"),
                    distance_matrix = pyg_graph.dist_mat,
                    distance = pyg_graph.distance,
                    coordinates = torch.FloatTensor(np.array(pyg_graph.coords[0])))
        #create the features
        x = np.array([np.argmax(pyg_graph.amino_acid_one_hot, axis=1)]).reshape(-1,1)
        for feat in self.features:
            if feat == "ss":
                feature = np.array([[self.map_secondary_structure.get(feat_node, 0)] \
                    for feat_node in pyg_graph[feat]])
            else:
                feature = np.array(pyg_graph[feat])
                if len(feature.shape) == 1:
                    feature = feature.reshape(-1,1)
            x = np.concatenate((x, feature), axis = 1)
        graph.edge_type = self.mlb.transform(pyg_graph.kind)
        graph.x = torch.FloatTensor(x)
        # y = self.annotations[graph.name.split("_")[0]]
        # if self.task == 'GeneOntology' :
        #     graph.y_mf = torch.FloatTensor(y["mf"])
        #     graph.y_cc = torch.FloatTensor(y["cc"])
        #     graph.y_bp = torch.FloatTensor(y["bp"])
        # else:
        #     graph.y_ec = torch.FloatTensor(y["ec"])
        return graph
    
    def init_ohe_edge_type(self):
        self.mlb = MultiLabelBinarizer(classes = ['peptide_bond', 'sequence_distance_2', 'sequence_distance_3'
                                             , 'distance_threshold', 'delaunay', 'hbond', 'k_nn'])
        self.mlb.fit([['peptide_bond', 'sequence_distance_2', 'sequence_distance_3'
                                             , 'distance_threshold', 'delaunay', 'hbond', 'k_nn']])
    
    def process(self):
        """Convert the PDB files into torch geometric graphs"""
        # self.pdb2graph = PDB2Graph(self.config)
        to_be_processed = self.get_files_to_process()
        
        # pool = multiprocessing.Pool(self.n_processors)
        # for _ in tqdm(pool.imap_unordered(self.graph_creation, to_be_processed), total=len(to_be_processed)):
        #     continue
        # pool.close()
        # pool.join()
        
        
        
        processes = []
        for prot in tqdm(to_be_processed):
            p = multiprocessing.Process(target=self.graph_creation, args=(prot,))
            processes.append(p)
            p.start()
            
        for process in processes:
            process.join()
      
    
    def graph_creation(self, pdb):
        """Create a graph from the PDB file"""

        # Define the path_to_structure from the pdb name file
        path_to_structure = self.pdb2pathstructure(pdb)

        # Convert the structure into a graph
        g = self.create_pyg_graph(path_to_structure)
        # Save the graph
        torch.save(g, os.path.join(self.output_folder, self.pdb2pathdata(pdb)))

        return None

    def pdb2pathdata(self, pdb):
        return pdb+'.pt'
    
    def pdb2pathstructure(self, pdb):
        return os.path.join(self.raw_dir, pdb+'.pdb')
    
    def get_files_to_process(self):
        RAW_FILES = self.processed_file_names
        PROCESSED_FILES = os.listdir(self.processed_dir)
        to_be_processed = set(RAW_FILES).difference(set(PROCESSED_FILES))
        to_be_processed = [path.split('.')[0] for path in to_be_processed]
        return to_be_processed
    
def download_alphafold_structure(
    uniprot_id: str,
    out_dir: str,
    version: int = 4
    ):
    
    BASE_URL = "https://alphafold.ebi.ac.uk/files/"
    uniprot_id = uniprot_id.upper()

    query_url = f"{BASE_URL}AF-{uniprot_id}-F1-model_v{version}.pdb"
    structure_filename = os.path.join(out_dir, f"AF-{uniprot_id}-F1-model_v{version}.pdb")
    if os.path.exists(structure_filename):
        return structure_filename
    try:
        structure_filename = wget.download(query_url, out=out_dir)
    except:
        print('Error.. could not download: ', f"AF-{uniprot_id}-F1-model_v{version}.pdb")
        return None
    return structure_filename

    