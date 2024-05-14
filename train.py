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
from prot2text_model.tokenization_prot2text import Prot2TextTokenizer
import torch.nn as nn
from transformers import EvalPrediction, Seq2SeqTrainingArguments
from transformers.trainer_utils import EvaluationStrategy, HubStrategy, IntervalStrategy, SchedulerType
import evaluate
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Dataset, download_url
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP
import os
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("--decoder_path", type=str, help="path to the gpt2 model to use (hugging face). options: gpt2, gpt2-medium, gpt2-large..")
argParser.add_argument("--esm_model_path", type=str, help="path to esm model to use. example: facebook/esm2_t12_35M_UR50D")
argParser.add_argument("--use_plm", action='store_true', help="True or False. (use or not protein language model in the encoder)")
argParser.add_argument("--use_rgcn", action='store_true', help="True or False. (use or not RGCN in the encoder)")
argParser.add_argument("--warmup_esm", action='store_true', help="True or False.")
argParser.add_argument("--warmup_gpt", action='store_true', help="True or False.")
argParser.add_argument("--data_path", type=str, default='./data//dataset/', help="root folder of the data")
argParser.add_argument("--train_csv_path", type=str, default='./data/train.csv', help="csv containing the protein dataset for training")
argParser.add_argument("--eval_csv_path", type=str, default='./data/eval.csv', help="csv containing the protein dataset for evaluation")
argParser.add_argument("--batch_per_device", type=int, default=4, help="batch size for each device")
argParser.add_argument("--nb_epochs", type=int, default=1, help="number of epochs")
argParser.add_argument("--nb_gpus", type=int, default=1, help="number of GPUs")
argParser.add_argument("--gradient_accumulation", default=1, help="gradient accumuluation")
argParser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
argParser.add_argument("--save_model_path", type=str, default='./models/model_test/', help="path to save the model and the checkpoints")
argParser.add_argument("--bleu_evaluation", action='store_true', help="True or False")

# usage for single GPU:
# python train.py \
#   --decoder_path gpt2 \
#   --esm_model_path facebook/esm2_t12_35M_UR50D \
#   --use_plm \
#   --use_rgcn \
#   --warmup_esm \
#   --warmup_gpt \    
#   --data_path ./data/dataset/ \
#   --train_csv_path ./data/train.csv \
#   --eval_csv_path ./data/eval.csv \    
#   --batch_per_device 4 \
#   --nb_epochs 25 \
#   --nb_gpus 1 \
#   --gradient_accumulation 64 \ 
#   --lr 2e-4 \ 
#   --save_model_path ./models/prot2text_base/ \
#   --bleu_evaluation \
    

# usage for multiple GPUs:
# python -u -m torch.distributed.run  --nproc_per_node <number of gpus> --nnodes <number of nodes> --node_rank 0 train.py \
#   --decoder_path gpt2 \
#   --esm_model_path facebook/esm2_t12_35M_UR50D \
#   --use_plm \
#   --use_rgcn \
#   --warmup_esm \
#   --warmup_gpt \    
#   --data_path ./data/dataset/ \
#   --train_csv_path ./data/train.csv \
#   --eval_csv_path ./data/eval.csv \    
#   --batch_per_device 4 \
#   --nb_epochs 25 \
#   --nb_gpus <number of gpus> \
#   --gradient_accumulation 1 \ 
#   --lr 2e-4 \ 
#   --save_model_path ./models/prot2text_base/ \
#   --bleu_evaluation \

args = argParser.parse_args()

if args.decoder_path is None:
    raise ValueError(
            "You need to specify a GPT like model path that is compatible with Hugging Face. Please pass the path of a Hugging Face decoder model using --decoder_path."
            )
if args.use_plm and args.esm_model_path is None:
    raise ValueError(
            "You want to use protein language model in the encoder, however you did not specify any PLM path.  Please pass the path of a Hugging Face PLM using --esm_model_path."
            )
if not args.use_plm and not args.use_rgcn:
    raise ValueError(
            "You did not choose which type of encoder to use. Please set --use_plm to train a PLM encoder, --use_rgcn for an RGCN encoder or both for Prot2Text architecture."
            )
if not args.use_plm and args.warmup_esm:
    raise ValueError(
            "You chose to warmup the protein language model however you chose not to use a PLM in the encoder. Please remove --warmup_esm or use --use_plm and specify a PLM using --esm_model_path."
            )    

model_name = args.decoder_path
tokenizer = Prot2TextTokenizer.from_pretrained(model_name)
SPECIAL_TOKEN = '<|graph_token|>'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = "<|endoftext|>"
tokenizer.add_tokens([SPECIAL_TOKEN])
SPECIAL_TOKEN = '<|stop_token|>'
tokenizer.add_tokens([SPECIAL_TOKEN])
tokenizer.eos_token = '<|stop_token|>'
tokenizer.eos_token_id = 50258
tokenizer.bos_token_id = 50257

esm_tokenizer = AutoTokenizer.from_pretrained(args.esm_model_path)

config_model = PretrainedConfig(
    _name_or_path='prot2text',
    prot2text_version="1.1",
    cross_esm_graph=args.use_plm & args.use_rgcn,
    esm=args.use_plm,
    esm_model_name=args.esm_model_path,
    gpt_model_name=model_name,
    rgcn=args.use_rgcn,
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
                                        pad_token_id=50256,
                                        vocab_size=50259,
                                        num_beams=1,
                                        max_length=256,
                                        min_length=1)
gpt_config.max_new_tokens = 256
gpt_config.prot2text_version = config_model.prot2text_version
config_model.gpt_config = gpt_config.to_dict()

model = Prot2TextModel(config=config_model)
if args.warmup_esm and args.warmup_gpt:
    model.warm_up(gpt_model=args.decoder_path, esm_model=args.esm_model_path)
elif args.warmup_esm:
    model.warm_up(esm_model=args.esm_model_path)
elif args.warmup_gpt:
    model.warm_up(gpt_model=args.decoder_path)

train_dataset = Prot2TextDataset(root=args.data_path,
                                 tokenizer=tokenizer,
                                 file_path=args.train_csv_path,
                                 block_size=256,
                                 split='train',
                                 esmtokenizer=esm_tokenizer)
print('train set loaded')
eval_dataset = Prot2TextDataset(root=args.data_path,
                                tokenizer=tokenizer,
                                file_path=args.eval_csv_path,
                                block_size=256,
                                split='eval',
                                esmtokenizer=esm_tokenizer)
print('eval set loaded')

num_gpus = int(args.nb_gpus)
train_size = len(train_dataset)
num_epochs = int(args.nb_epochs)
grad_accumulation = int(args.gradient_accumulation)
batch_size = int(args.batch_per_device)
warmup = 0.06 * num_epochs * train_size / (num_gpus * batch_size * grad_accumulation)
model_save_name = args.save_model_path
lr = args.lr

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_ids[pred_ids == -100] = tokenizer.eos_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.eos_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    try:
        res = bleu.compute(predictions=pred_str, references=label_str)
        return {'eval_bleu': res['bleu']}
    except:
        return {'eval_bleu': 0.0}

if args.bleu_evaluation:
    prediction_loss_only = False
    metric_for_best_model = 'eval_bleu'
    greater_is_better = True
    predict_with_generate = True
    load_best_model_at_end = True
    bleu = evaluate.load("bleu")
    do_predict = True
else:
    compute_metrics = None
    prediction_loss_only = True
    metric_for_best_model = 'loss'
    greater_is_better = False
    predict_with_generate = False
    load_best_model_at_end = False
    do_predict = False

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
    do_train=True,
    do_eval=True,
    do_predict=do_predict,
    prediction_loss_only=prediction_loss_only,
    metric_for_best_model=metric_for_best_model,
    greater_is_better=greater_is_better,
    predict_with_generate=predict_with_generate,
    load_best_model_at_end=load_best_model_at_end,
    )

trainer = Prot2TextTrainer(
    model=model,
    args=training_args,
    data_collator=None,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

if torch.distributed.is_initialized():
    if torch.distributed.get_rank()==0:
        model.save_pretrained(os.path.join(model_save_name,'model/'))
        tokenizer.save_pretrained(os.path.join(model_save_name,'model/'))
    else:
        model.save_pretrained(os.path.join(model_save_name,'model/'))
        tokenizer.save_pretrained(os.path.join(model_save_name,'model/'))