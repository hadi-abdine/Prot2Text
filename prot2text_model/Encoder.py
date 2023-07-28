import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GINConv,GATv2Conv,TAGConv,ARMAConv,APPNP,MFConv, GMMConv,HypergraphConv,LEConv,PNAConv, GCNConv,SAGEConv, RGCNConv
from torch_scatter import scatter_add, scatter_mean
from typing import Optional, Tuple, Union
from torch_geometric.nn import global_add_pool,global_mean_pool
from torch.nn import init
import random
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import aggr
from torch_geometric.utils import sort_edge_index
import torch_geometric
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP, GPT2PreTrainedModel, PARALLELIZE_DOCSTRING, DEPARALLELIZE_DOCSTRING
import transformers
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
    
class EncoderRGCN(PreTrainedModel):
    def __init__(self, input_dim, hidden_dim=512, n_layers=6, emb_dim=512, dropout=0.2, num_relation=7):
        super(EncoderRGCN, self).__init__(PretrainedConfig(name='RGCN'))
        self.n_layers = n_layers
        self.output_dim = emb_dim

        self.fc0 = nn.Linear(input_dim, hidden_dim)
        self.batchnorm_final = nn.BatchNorm1d(hidden_dim)
        
        self.batch_norms = nn.ModuleList()
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        lst = list()
        
        lst.append(RGCNConv(hidden_dim, hidden_dim, num_relations=num_relation))
            
        for i in range(n_layers-1):
            lst.append(RGCNConv(hidden_dim,hidden_dim, num_relations=num_relation))

        self.conv = nn.ModuleList(lst)
      
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)
      
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.LeakyReLU()
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.main_input_name = 'nothing'

    def forward(self, x:Optional[torch.FloatTensor] = None, 
                edge_index:Optional[torch.LongTensor] = None,
                edge_type:Optional[torch.LongTensor] = None,
                batch:Optional[torch.LongTensor] = None,
                **kargs):
        #construct pyg edge index shape (2, num_edges) from edge_list
        x = self.relu(self.fc0(x))
        
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index, edge_type)
                
        out = global_mean_pool(x, batch)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        
        return out.unsqueeze(1)