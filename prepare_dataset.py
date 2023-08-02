from prot2text_dataset.pdb2graph import *
from prot2text_dataset.utils_dataset import *
import prot2text_dataset.graphs
import wget
from tqdm import tqdm
import os
import argparse
from functools import partial
from transformers import AutoTokenizer
from prot2text_dataset.torch_geometric_loader import Prot2TextDataset
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

argParser = argparse.ArgumentParser()
argParser.add_argument("--data_save_path", help="folder to save the dataset")
argParser.add_argument("--csv_path", help="csv containing the protein dataset")
argParser.add_argument("--split", help="train, test or eval csv?")
argParser.add_argument("--plm_model", help="protein model to use (from hugging face)")
argParser.add_argument("--decoder_model", help="language model to use (from hugging face)")

# usage:
# python prepare_dataset.py \
#   --data_save_path data/dataset/ \
#   --split test --csv_path data/test.csv \
#   --plm_model facebook/esm2_t12_35M_UR50D \
#   --decoder_model gpt2

args = argParser.parse_args()


# step 1: download the PDB files from AlphaFoldDB
isExist = os.path.exists(os.path.join(args.data_save_path, args.split))
if not isExist:
    os.makedirs(os.path.join(args.data_save_path, args.split, 'pdb'))
    os.makedirs(os.path.join(args.data_save_path, args.split, 'raw'))
    os.makedirs(os.path.join(args.data_save_path, args.split, 'processed'))

print('downloading the data:\n')

df = pd.read_csv(args.csv_path)

pdb_path = os.path.join(args.data_save_path, args.split, 'pdb')
for prot in tqdm(set(df.AlphaFoldDB)):
    if os.path.exists(os.path.join(pdb_path, 'AF-'+str(prot)+'-F1-model_v4.pdb')):
        continue
    download_alphafold_structure(uniprot_id=str(prot), out_dir=pdb_path)

# step 2: construct graphs from the pdb files
print('constructing the graphs:\n')
if len(os.listdir(os.path.join(args.data_save_path, args.split, 'raw'))) == len(os.listdir(os.path.join(args.data_save_path, args.split, 'pdb'))):
    print('graphs already created')
else:
    config = {"node_metadata_functions": [amino_acid_one_hot, 
                                        expasy_protein_scale,
                                        meiler_embedding,
                                        hydrogen_bond_acceptor, 
                                        hydrogen_bond_donor
                                        ],
            "edge_construction_functions": [add_peptide_bonds,
                                            add_hydrogen_bond_interactions,
                                            partial(add_distance_threshold, 
                                                    long_interaction_threshold=3, 
                                                    threshold=10.),],
            "graph_metadata_functions":[asa,phi, psi, secondary_structure, rsa],
            "dssp_config": DSSPConfig(),}
    config = ProteinGraphConfig(**config)
    PDB2Graph(root = pdb_path, 
              output_folder = os.path.join(args.data_save_path, args.split, 'raw'), 
              config=config, n_processors=32).process()

# step 3: process the dataset
esm_tokenizer = AutoTokenizer.from_pretrained(args.plm_model)
tokenizer = AutoTokenizer.from_pretrained(args.decoder_model)
SPECIAL_TOKEN = '<|graph_token|>'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = 50256
tokenizer.add_tokens([SPECIAL_TOKEN])
SPECIAL_TOKEN = '<|stop_token|>'
tokenizer.add_tokens([SPECIAL_TOKEN])
tokenizer.eos_token = '<|stop_token|>'
tokenizer.eos_token_id = 50258
tokenizer.bos_token_id = 50257

dataset = Prot2TextDataset(root=args.data_save_path, 
                          tokenizer=tokenizer, 
                          file_path=args.csv_path, 
                          block_size=256, 
                          split=args.split, 
                          esmtokenizer=esm_tokenizer)
