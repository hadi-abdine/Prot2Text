from transformers import GPT2Tokenizer, Seq2SeqTrainingArguments
from prot2text_dataset.torch_geometric_loader import Prot2TextDataset
from prot2text_model.utils import Prot2TextTrainer
from prot2text_model.Model import Prot2TextModel
from prot2text_model.tokenization_prot2text import Prot2TextTokenizer
import evaluate
from torch_geometric.loader import DataLoader
import pandas as pd
from transformers.utils import logging
from tqdm import tqdm
import torch
import os
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("--model_path", help="path to the prot2text model")
argParser.add_argument("--data_path", help="root folder of the data")
argParser.add_argument("--csv_path", help="csv containing the protein dataset to evaluate")
argParser.add_argument("--split", help="train, test or eval csv?")
argParser.add_argument("--batch_per_device", help="batch size for each device")
argParser.add_argument("--save_results_path", help="path to save the generated description")

# usage for single GPU:
# python evaluate_prot2text.py \
#   --model_path ./models/prot2text_base \
#   --data_path ./data/dataset/ \
#   --split test \
#   --csv_path ./data/test.csv \
#   --batch_per_device 4 \
#   --save_results_path ./results/prot2text_base_results.csv

# usage for multiple GPUs:
# python -u -m torch.distributed.run  --nproc_per_node <number of gpus> --nnodes <number of nodes> --node_rank 0 evaluate_prot2text.py \
#   --model_path ./models/prot2text_base \
#   --data_path ./data/dataset/ \
#   --split test \
#   --csv_path ./data/test.csv \
#   --batch_per_device 4 \
#   --save_results_path ./results/prot2text_base_results.csv

args = argParser.parse_args()

tokenizer = Prot2TextTokenizer.from_pretrained(args.model_path)

model = Prot2TextModel.from_pretrained(args.model_path)
eval_dataset = Prot2TextDataset(root=args.data_path, 
                                tokenizer=tokenizer, 
                                file_path=args.csv_path, 
                                block_size=256, 
                                split=args.split)
print('eval set loaded')

batch_size = int(args.batch_per_device)
model.eval()
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bert_score = evaluate.load("bertscore")

args_seq = Seq2SeqTrainingArguments(output_dir='./', per_device_eval_batch_size=batch_size)
trainer = Prot2TextTrainer(model=model, args=args_seq, eval_dataset=eval_dataset)

d = trainer.get_eval_dataloader()

if torch.distributed.is_initialized():
    if torch.distributed.get_rank()==0:
        if os.path.exists(args.save_results_path):
            os.remove(args.save_results_path)
else:
    if os.path.exists(args.save_results_path):
        os.remove(args.save_results_path)
        
names = list()
generated = list()
functions = list()

for inputs in tqdm(d):
    inputs = inputs.to_dict()
    inputs['edge_type'] =  torch.cat([torch.tensor(inputs['edge_type'][i]) for i in range(len(inputs['edge_type']))], dim=0)
    inputs['edge_type'] = torch.argmax(inputs['edge_type'], dim=1)
    names +=  inputs['name']
    functions +=  tokenizer.batch_decode(inputs['decoder_input_ids'], skip_special_tokens=True)
    inputs['decoder_input_ids'] = inputs['decoder_input_ids'][:,0:1]
    inputs["decoder_attention_mask"] = torch.ones(inputs['decoder_input_ids'].shape[0], 1)
    inputs = {k: v.to(device=torch.cuda.current_device(), non_blocking=True) if hasattr(v, 'to') else v for k, v in inputs.items()}
    tok_ids = model.generate(inputs=None, **inputs)
    generated += tokenizer.batch_decode(tok_ids, skip_special_tokens=True)

data= {'name':names, 'generated': generated, 'function':functions}
df = pd.DataFrame(data)
df.to_csv(args.save_results_path, index=False, mode='a')

if torch.distributed.is_initialized():  
    torch.distributed.barrier() 
    if torch.distributed.get_rank() > 0:
        exit(0)   
res = pd.read_csv(args.save_results_path).drop_duplicates()
res = res.drop(res[res['name'] == 'name'].index)   

res_bleu = bleu.compute(predictions=res['generated'].tolist(), references=res['function'].tolist())
res_rouge = rouge.compute(predictions=res['generated'].tolist(), references=res['function'].tolist())
res_bertscore = bert_score.compute(predictions=res['generated'].tolist(), references=res['function'].tolist(),
                                  model_type="dmis-lab/biobert-large-cased-v1.1", num_layers=24)
print(res_bleu)
print(res_rouge)
def Average(lst):
    return sum(lst) / len(lst)
print('Bert Score: ', Average(res_bertscore['f1']))