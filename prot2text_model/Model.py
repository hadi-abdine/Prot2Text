from transformers import GPT2Config, AutoConfig, AutoTokenizer, GPT2Config
from transformers import GPT2LMHeadModel, GPT2Model, PretrainedConfig, PreTrainedModel
import transformers
from .Encoder import EncoderRGCN
from typing import Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP, GPT2PreTrainedModel
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from .utils import CABlock, _GPT2LMHeadModel
import os
import sys
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
sys.path.append('../prot2text_dataset')
from prot2text_dataset.pdb2graph import PDB2Graph, download_alphafold_structure
from prot2text_dataset.graphs import *
from prot2text_dataset.utils_dataset import *
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

class Prot2TextModel(PreTrainedModel):
    config_class = PretrainedConfig
    _keys_to_ignore_on_load_missing = [r"transformer"]
    base_model_prefix = "decoder"
    def __init__(self, config):
        super().__init__(config)

        self.gpt_config = GPT2Config.from_dict(config.gpt_config)
        if config.rgcn:
            self.encoder = EncoderRGCN(input_dim=config.rgcn_input_dim, hidden_dim=self.gpt_config.n_embd, n_layers=config.rgcn_n_layers, emb_dim=self.gpt_config.n_embd)

        self.decoder = _GPT2LMHeadModel(self.gpt_config)

        if config.esm:
            self.esm_config = PretrainedConfig.from_dict(config.esm_config)
            self.esm = transformers.EsmModel(self.esm_config)
            self.to_embedding = nn.Linear(self.esm_config.hidden_size, self.gpt_config.n_embd)
            if config.cross_esm_graph and config.rgcn:
                self.h = nn.ModuleList([CABlock(self.gpt_config,  layer_idx=i) for i in range(4)])
                self.ln_f = nn.LayerNorm(self.gpt_config.n_embd, eps=self.gpt_config.layer_norm_epsilon)
            
        self.config = config
        
        
    def get_encoder(self):
        return self.encoder
        
    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        if hasattr(self, "transformer"):
            return self.transformer.wte
        return self.decoder.transformer.wte
    
    def warm_up(self, gpt_model=None, esm_model=None):
        if esm_model is not None:
            self.esm = transformers.EsmModel.from_pretrained(esm_model)
        if gpt_model is not None:    
            self.decoder = _GPT2LMHeadModel.from_pretrained(gpt_model, add_cross_attention=True, use_cache=False)
            self.decoder.resize_token_embeddings(self.gpt_config.vocab_size)
            self.decoder.config = self.gpt_config
                
        
    def forward(self,
                encoder_input_ids: Optional[torch.LongTensor] = None,
                edge_index: Optional[torch.LongTensor] = None,
                batch: Optional[torch.LongTensor] = None,
                x: Optional[torch.FloatTensor] = None,
                edge_type: Optional[torch.LongTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                past_key_values_graph_esm: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                decoder_attention_mask: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                get_graph_emb: Optional[bool] = False,
                **delete_args,
            ):
        use_cache = use_cache if use_cache is not None else self.gpt_config.use_cache
        return_dict = return_dict if return_dict is not None else self.gpt_config.use_return_dict
        
        
        if decoder_input_ids is not None and len(decoder_input_ids.size()) == 3:
            decoder_input_ids = decoder_input_ids.squeeze(0) 

        if x is not None and self.config.rgcn:
            graph_emb = self.encoder(x, edge_index, edge_type, batch)
            
        if self.config.esm:
            esm_emb = self.esm(input_ids=encoder_input_ids, attention_mask=attention_mask, return_dict=return_dict).last_hidden_state
            esm_emb = self.to_embedding(esm_emb)
            if not self.config.cross_esm_graph and self.config.rgcn:
                graph_emb = torch.cat((graph_emb, esm_emb), dim=1) 
            elif self.config.cross_esm_graph and self.config.rgcn:
                if past_key_values_graph_esm is None:
                    past_length = 0
                    past_key_values_graph_esm = tuple([None] * len(self.h))
                else:
                    past_length = past_key_values_graph_esm[0][0].size(-2) 
                output_shape = esm_emb.size()
                
                all_self_attentions = () if output_attentions else None
                all_cross_attentions = () if output_attentions and self.gpt_config.add_cross_attention else None
                all_hidden_states = () if output_hidden_states else None
                for i, (block, layer_past) in enumerate(zip(self.h, past_key_values_graph_esm)):
                    outputs = block(
                        esm_emb,
                        layer_past=layer_past,
                        attention_mask=attention_mask,
                        encoder_hidden_states=graph_emb,
                        encoder_attention_mask=None,
                        use_cache=use_cache,
                        output_attentions=False,
                    )
                    esm_emb = outputs[0]

                esm_emb = self.ln_f(esm_emb)
                esm_emb = esm_emb.view(output_shape)  
                graph_emb = esm_emb
            else:
                graph_emb = esm_emb

        if get_graph_emb:
            return graph_emb
        transformer_outputs = self.decoder(input_ids=decoder_input_ids,
                                            past_key_values=past_key_values,
                                            attention_mask=decoder_attention_mask,
                                            token_type_ids=token_type_ids,
                                            position_ids=position_ids,
                                            head_mask=head_mask,
                                            inputs_embeds=inputs_embeds,
                                            encoder_hidden_states=graph_emb,
                                            encoder_attention_mask=encoder_attention_mask,
                                            labels=labels,
                                            use_cache=use_cache,
                                            output_attentions=output_attentions,
                                            output_hidden_states=output_hidden_states,
                                            return_dict=return_dict,
                                            )
        
        return transformer_outputs
    
    @torch.no_grad()    
    def generate_protein_description(self,
                                    protein_pdbID=None, 
                                    protein_sequence=None,
                                    edge_index: Optional[torch.LongTensor] = None,
                                    x: Optional[torch.FloatTensor] = None,
                                    edge_type: Optional[torch.LongTensor] = None,
                                    tokenizer=None,
                                    device='cpu'
                                     ):
        
        if self.config.esm and not self.config.rgcn and protein_sequence==None:
            raise ValueError(
                "The model you are trying to use is based only on protein sequence, please provide an amino-acid protein_sequence"
            )
        if self.config.rgcn and protein_pdbID==None and (x==None or edge_index==None or edge_type==None):
            raise ValueError(
                "The model you are trying to use is based on protein structure, please provide a AlphaFold ID (you must have to have internet connection using protein_pdbID, or provide the triplet inputs: x (node features), edge_index and edge_type"
            )
        if self.config.esm:
            esmtokenizer = AutoTokenizer.from_pretrained(self.config.esm_model_name)
        
        if protein_pdbID==None and protein_sequence==None:
            raise ValueError(
                "you need to provide either a protein AlphaFold Id or an amino-acid sequence"
            )
            
        if protein_pdbID!=None:
            config = {"node_metadata_functions": [amino_acid_one_hot, 
                                                expasy_protein_scale,
                                                meiler_embedding,
                                                hydrogen_bond_acceptor, hydrogen_bond_donor
                                                ],
                    "edge_construction_functions": [add_peptide_bonds,
                                                    add_hydrogen_bond_interactions,
                                                    partial(add_distance_threshold, long_interaction_threshold=3, threshold=10.),],
                    "graph_metadata_functions":[asa,phi, psi, secondary_structure, rsa],
                    "dssp_config": DSSPConfig(),}
            config = ProteinGraphConfig(**config)

            PATH_TO_DATA = f"./.tmp/pdb/pdb"
            OUTPUT_FOLDER = f"./.tmp/pdb/raw"
            save_dir = f"./.tmp/pdb/"
            isExist = os.path.exists(PATH_TO_DATA)
            if not isExist:
                os.makedirs(PATH_TO_DATA)
            isExist = os.path.exists(OUTPUT_FOLDER)
            if not isExist:
                os.makedirs(OUTPUT_FOLDER)
            isExist = os.path.exists(save_dir+'processed')
            if not isExist:
                os.makedirs(save_dir+'processed')
            
            structure_filename = download_alphafold_structure(uniprot_id=protein_pdbID, out_dir=PATH_TO_DATA)
            if structure_filename is None:
                raise ValueError("Error! the ID does not exist in AlphaFoldDB or you do not have internet connection")
            graph_filename = structure_filename.split('/')
            graph_filename[-2] = 'raw'
            graph_filename[-1] = graph_filename[-1].replace('.pdb', '.pt')
            graph_filename = '/'.join(graph_filename)
            process_filename = structure_filename.split('/')
            process_filename[-2] = 'processed'
            process_filename[-1] = process_filename[-1].replace('.pdb', '.pt')
            process_filename = '/'.join(process_filename)    
            try:            
                gpdb = PDB2Graph(root = PATH_TO_DATA, output_folder = OUTPUT_FOLDER, config=config, n_processors=1).create_pyg_graph(structure_filename)
                seq = esmtokenizer(gpdb.sequence, add_special_tokens=True, truncation=True, max_length=1021, padding='max_length', return_tensors="pt") 
                torch.save(gpdb, graph_filename)
                gpdb.edge_type = [np.array(gpdb.edge_type.transpose(0,1))]
                gpdb.encoder_input_ids = seq['input_ids']
                gpdb.attention_mask = seq['attention_mask']
                torch.save(gpdb, process_filename)
            except:
                os.remove(structure_filename)
                raise ValueError('creating graphs did not work, probably the pdb file of alphaFold is damaged')
            
            self.eval()
            inputs = gpdb
            inputs = inputs.to_dict()
            inputs['edge_type'] =  torch.cat([torch.tensor(inputs['edge_type'][i]) for i in range(len(inputs['edge_type']))], dim=0)
            inputs['edge_type'] = torch.argmax(inputs['edge_type'], dim=1)
            for key in ['num_nodes', 'node_id', 'name', 'sequence', 'distance_matrix', 'distance', 'coordinates']:
                inputs.pop(key)
            inputs['decoder_input_ids'] = inputs['encoder_input_ids'][:,0:1].clone()
            inputs['decoder_input_ids'][:,0] = tokenizer.bos_token_id
            inputs["decoder_attention_mask"] = torch.ones(inputs['decoder_input_ids'].shape[0], 1)
            self.to(device)
            inputs = {k: v.to(device=device, non_blocking=True) if hasattr(v, 'to') else v for k, v in inputs.items()}
            encoder_state = self(**inputs, get_graph_emb=True)
            for key in ['edge_index', 'edge_type', 'x', 'encoder_input_ids']:
                inputs.pop(key)
            tok_ids = self.decoder.generate(input_ids=inputs['decoder_input_ids'], encoder_outputs=encoder_state, use_cache=True)#, just_decoder=True)
            generated = tokenizer.batch_decode(tok_ids, skip_special_tokens=True)
            
            os.remove(structure_filename)
            os.remove(graph_filename)
            os.remove(process_filename)        
                
            return generated[0].replace('<|stop_token|>', '').replace('<|graph_token|>', '')
            
        else:
            seq = esmtokenizer([protein_sequence], add_special_tokens=True, truncation=True, max_length=1021, padding='max_length', return_tensors="pt")
            inputs={}
            inputs['encoder_input_ids'] = seq['input_ids']
            inputs['attention_mask'] = seq['attention_mask']
            inputs['decoder_input_ids'] = inputs['encoder_input_ids'][:,0:1].clone()
            inputs['decoder_input_ids'][:,0] = tokenizer.bos_token_id
            
            self.to(device)
            inputs = {k: v.to(device=device, non_blocking=True) if hasattr(v, 'to') else v for k, v in inputs.items()}
            encoder_state = self(**inputs, get_graph_emb=True)
            generated = tokenizer.batch_decode(self.decoder.generate(input_ids=inputs['decoder_input_ids'], encoder_outputs=encoder_state, use_cache=True), skip_special_tokens=True)
            
            return generated[0].replace('<|stop_token|>', '').replace('<|graph_token|>', '')
    
    @torch.no_grad()
    def generate(self,
                inputs: Optional[torch.Tensor] = None,
                generation_config: Optional[GenerationConfig] = None,
                logits_processor: Optional[LogitsProcessorList] = None,
                stopping_criteria: Optional[StoppingCriteriaList] = None,
                prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
                synced_gpus: Optional[bool] = None,
                assistant_model: Optional["PreTrainedModel"] = None,
                streamer: Optional["BaseStreamer"] = None,
                **kwargs,
            ):

        encoder_state = self(**kwargs, get_graph_emb=True)
        input_ids = kwargs['decoder_input_ids']
        attention_mask = kwargs['decoder_attention_mask']
        for key in ['edge_index', 'edge_type', 'x', 'encoder_input_ids', 'decoder_input_ids', 'decoder_attention_mask', 'batch', 'attention_mask', 'max_length',
                    'num_nodes', 'node_id', 'name', 'sequence', 'distance_matrix', 'distance', 'coordinates', 'ptr']:
            if key in kwargs.keys():
                kwargs.pop(key)
        return self.decoder.generate(input_ids=input_ids,
                                     generation_config=generation_config,
                                     logits_processor=logits_processor,
                                     stopping_criteria=stopping_criteria,
                                     prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                     synced_gpus=synced_gpus,
                                     assistant_model=assistant_model,
                                     streamer=streamer,
                                     encoder_outputs=encoder_state,
                                     **kwargs
                                     )