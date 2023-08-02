from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer
from typing import Optional, Tuple, Union, TYPE_CHECKING, Any, Callable, Dict, List
import torch
from transformers import GPT2Config, AutoConfig, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Model, PretrainedConfig
import transformers
from prot2text_model.Encoder import EncoderRGCN
from prot2text_dataset.torch_geometric_loader import Prot2TextDataset
from prot2text_model.utils import Prot2TextTrainer, CABlock, _GPT2LMHeadModel
from prot2text_model.Model import Prot2TextModel
import torch.nn as nn
from transformers import EvalPrediction, Seq2SeqTrainingArguments
from transformers.trainer_utils import EvaluationStrategy, HubStrategy, IntervalStrategy, SchedulerType, ShardedDDPOption
import evaluate
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Dataset, download_url
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP
import argparse

# argParser = argparse.ArgumentParser()
# argParser.add_argument("--model_path", help="path to the prot2text model")
# argParser.add_argument("--data_path", help="root folder of the data")
# argParser.add_argument("--csv_path", help="csv containing the protein dataset to evaluate")
# argParser.add_argument("--split", help="train, test or eval csv?")
# argParser.add_argument("--batch_per_device", help="batch size for each device")
# argParser.add_argument("--save_results_path", help="path to save the generated description")

# args = argParser.parse_args()


model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
SPECIAL_TOKEN = '<|graph_token|>'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = 50256
tokenizer.add_tokens([SPECIAL_TOKEN])
SPECIAL_TOKEN = '<|stop_token|>'
tokenizer.add_tokens([SPECIAL_TOKEN])
tokenizer.eos_token = '<|stop_token|>'
tokenizer.eos_token_id = 50258
tokenizer.bos_token_id = 50257

esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

config_model = PretrainedConfig(
    _name_or_path='prot2text',
    cross_esm_graph=True,
    esm=True,
    esm_model_name="facebook/esm2_t12_35M_UR50D",
    gpt_model_name=model_name,
    rgcn=True,
    rgcn_input_dim = 67,
    rgcn_n_layers = 6,
    decoder_start_token_id = 50257,
    eos_token_id = 50258,
    max_new_tokens = 256,
    no_repeat_ngram_size = 3,
    early_stopping = True,
    length_penalty = 2.0,
    num_beams = 1,
    pad_token_id = 50256,
    bos_token_id = 50257
    )
esm_config = AutoConfig.from_pretrained(config_model.esm_model_name).to_dict()
config_model.esm_config = esm_config
gpt_config = GPT2Config.from_pretrained(config_model.gpt_model_name,
                                        _name_or_path= config_model.gpt_model_name,
                                        is_encoder_decoder=True,
                                        use_cache=False,
                                        add_cross_attention=True,
                                        bos_token_id=config_model.bos_token_id,
                                        decoder_start_token_id=config_model.decoder_start_token_id,
                                        eos_token_id=config_model.eos_token_id,
                                        max_new_tokens=config_model.max_new_tokens,
                                        pad_token_id = 50256,
                                        vocab_size=50259,
                                        num_beams=1,
                                        max_length=256)
gpt_config.max_new_tokens = 256
config_model.gpt_config = gpt_config.to_dict()

# class Prot2TextModelTrain(Prot2TextModel):
#   def __init__(self, config):
#     super().__init__(config)
    
#     self.gpt_config = GPT2Config.from_dict(config.gpt_config)
#     if config.rgcn:
#         self.encoder = EncoderRGCN(input_dim=config.rgcn_input_dim, hidden_dim=self.gpt_config.n_embd, n_layers=config.rgcn_n_layers, emb_dim=self.gpt_config.n_embd)

#     self.decoder = _GPT2LMHeadModel.from_pretrained(config.gpt_model_name, add_cross_attention=True, use_cache=False)

#     if config.esm:
#         self.esm_config = PretrainedConfig.from_dict(config.esm_config)
#         self.esm = transformers.EsmModel.from_pretrained(config.esm_model_name)
#         self.to_embedding = nn.Linear(self.esm_config.hidden_size, self.gpt_config.n_embd)
#         if config.cross_esm_graph and config.rgcn:
#             self.h = nn.ModuleList([CABlock(self.gpt_config,  layer_idx=i) for i in range(4)])
#             self.ln_f = nn.LayerNorm(self.gpt_config.n_embd, eps=self.gpt_config.layer_norm_epsilon)
        
#     self.config = config
    
#     print('number of parameters: ', self.decoder.num_parameters())
#     a = self.decoder.get_input_embeddings()
#     b = self.decoder.get_output_embeddings()

#     my_new_token_embedding = torch.randn(2, a.embedding_dim)
#     a.weight = torch.nn.Parameter(torch.cat((a.weight, my_new_token_embedding.clone()), dim=0))
#     b.weight = torch.nn.Parameter(torch.cat((b.weight, my_new_token_embedding.clone()), dim=0))
    
#     # Update the vocabulary size
#     a.num_embeddings += 2
#     b.out_features += 2
#     self.decoder.set_input_embeddings(a)
#     self.decoder.set_output_embeddings(b)
#     self.decoder.resize_token_embeddings(len(tokenizer))
#     print('number of parameters: ', self.decoder.num_parameters())
    
#     self.post_init()


model = Prot2TextModel(config=config)
model.warm_up(gpt_model='gpt2', esm_model="facebook/esm2_t12_35M_UR50D")

train_dataset = Prot2TextDataset(root='../data//uniprot_graphs/', tokenizer=tokenizer, file_path="../data/uniprot_sprot_all_functions_train_40split_cleaned.csv", block_size=256, split='train', esmtokenizer=esm_tokenizer)
print('train set loaded')
eval_dataset = Prot2TextDataset(root='../data/uniprot_graphs/', tokenizer=tokenizer, file_path="../data/valid_tmp.csv", block_size=256, split='eval', esmtokenizer=esm_tokenizer)
print('eval set loaded')

num_gpus = 64
train_size =  len(train_dataset)
num_epochs = 25
grad_accumulation = 1
batch_size = 4
warmup = 0.06 * num_epochs * train_size / (num_gpus * batch_size * grad_accumulation)
model_save_name = '../models/model_Prot2Text_Base/'
lr = 2e-4

# bleu = evaluate.load("bleu")
# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     labels_ids[labels_ids == -100] = tokenizer.eos_token_id
#     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
#     data= {'generated': pred_str, 'function':label_str}
#     res = bleu.compute(predictions=pred_str, references=label_str)
#     return {'eval_bleu': res['bleu']}

training_args = Seq2SeqTrainingArguments(
    output_dir=model_save_name,
    overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=grad_accumulation,
    eval_accumulation_steps=None,
    evaluation_strategy=IntervalStrategy.STEPS,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    save_total_limit=15,
    weight_decay=0.1,
    warmup_steps=warmup,
    lr_scheduler_type="cosine",
    learning_rate=lr,
    load_best_model_at_end=True,
    do_train = True,
    do_eval = True,
    prediction_loss_only=True,
    # metric_for_best_model='eval_bleu',
    # greater_is_better=True,
    # predict_with_generate=True,
    # load_best_model_at_end=True,
    # fp16=True, # did not work
    # world_size=1,
    # do_predict = True,
    # no_cuda=True
    )


trainer = Prot2TextTrainer(
    model=model,
    args=training_args,
    data_collator=None,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    #compute_metrics=compute_metrics,
)

trainer.train()

# if torch.distributed.get_rank()==0:
#     model.save_pretrained('')
#     tokenizer.save_pretrained('')