 
import os.path as osp
import os
import torch
import pandas as pd 
import numpy as np
import torch
import json
import os
import pickle
import random
import time
import warnings
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import scipy.sparse as sp
import numpy as np
import scipy
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging
import pandas as pd
from transformers import DataCollatorForLanguageModeling, GPT2Tokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from torch_geometric.data import Dataset, download_url
import torch
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel
from tqdm import tqdm

class Prot2TextDataset(Dataset):
    def __init__(self,
                 root, 
                 tokenizer: PreTrainedTokenizer,
                 file_path: str,
                 block_size: int,
                 split: str = "train",
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 esmtokenizer: PreTrainedTokenizer = None,):
        
            
        self.split_path = split
        self.files_names_folder = os.path.join(root, split, 'raw')
        self.files_names = os.listdir(self.files_names_folder)
        self.uniprot_csv = pd.read_csv(file_path)   
        
        self.tokenizer = tokenizer
        self.esmtokenizer = esmtokenizer
        self.block_size = block_size
        print('dataset loading:')
            
        self.length = len(self.files_names)
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self):
        return self.files_names
        
    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.length)]
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.split_path, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.split_path, 'processed')

    
    def process(self):
        idx = 0
        print("length:",len(self.files_names))
        for i in tqdm(range(len(self.files_names))):
            try:
                graph = torch.load(os.path.join(self.root, self.split_path, "raw",self.files_names[i]))
                function = '<|graph_token|> '+self.uniprot_csv.loc[self.uniprot_csv['accession'] == self.files_names[i].split("-")[1]]["function"].values[0]+' <|stop_token|> '
                sequence = self.uniprot_csv.loc[self.uniprot_csv['accession'] == self.files_names[i].split("-")[1]]["sequence"].values[0]
                
                text = self.tokenizer([function], add_special_tokens=True, truncation=True, max_length=self.block_size, padding='max_length', return_tensors="pt") #(1, max_length)
                seq = self.esmtokenizer([sequence], add_special_tokens=True, truncation=True, max_length=1021, padding='max_length', return_tensors="pt") #(1, max_length)
                
                graph.encoder_input_ids = seq['input_ids'] 
                graph.attention_mask = seq['attention_mask']
                graph.decoder_input_ids = text['input_ids']
                graph.decoder_attention_mask = text['attention_mask'] 
                labels = text['input_ids'].clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                graph.labels = labels
                graph.edge_type = graph.edge_type.transpose(0,1)

                torch.save(graph, osp.join(self.processed_dir ,f'data_{idx}.pt'))
                idx = idx + 1     
            except:
                print('error loading ', self.files_names[i])
                print("don't forget to delete it from raw files to avoid error")


    def len(self):
        return len(self.processed_file_names)
    
    def __len__(self):
        return len(self.processed_file_names)
        
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    
    def download(self):
        pass
    
    def __cat_dim__(self, key, value, args, *kwargs):
        if 'index' in key or 'face' in key or 'edge_type' in key:
            return 1
        else:
            return 0