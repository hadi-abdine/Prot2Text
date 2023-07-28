from transformers import GPT2Tokenizer, Seq2SeqTrainingArguments
from prot2text_dataset.torch_geometric_loader import Prot2TextDataset
from prot2text_model.utils import Prot2TextTrainer
from prot2text_model.Model import Prot2TextModel
import evaluate
from torch_geometric.loader import DataLoader
import pandas as pd
from transformers.utils import logging
from tqdm import tqdm
import torch
import os


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

model = Prot2TextModel.from_pretrained('/datadisk/GPT/Final_Code/tmp/prot2text_base')
eval_dataset = Prot2TextDataset(root='../data/uniprot_graphs/', 
                                tokenizer=tokenizer, 
                                file_path="/datadisk/GPT/data/csvs/uniprot_sprot_all_functions_test_40split_cleaned.csv", 
                                block_size=256, 
                                split='test')
print('eval set loaded')

batch_size = 4
model.eval()
bleu = evaluate.load("bleu")

args = Seq2SeqTrainingArguments(output_dir='./', per_device_eval_batch_size=batch_size)
trainer = Prot2TextTrainer(model=model, args=args, eval_dataset=eval_dataset)

# d = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=None, shuffle = False)
d = trainer.get_eval_dataloader()
names = list()
generated = list()
functions = list()
for inputs in tqdm(d):
    inputs = inputs.to_dict()
    inputs['edge_type'] =  torch.cat([torch.tensor(inputs['edge_type'][i]) for i in range(len(inputs['edge_type']))], dim=0)
    inputs['edge_type'] = torch.argmax(inputs['edge_type'], dim=1)
    names +=  inputs['name']
    functions +=  tokenizer.batch_decode(inputs['decoder_input_ids'], skip_special_tokens=True)
    for key in ['num_nodes', 'node_id', 'name', 'sequence', 'distance_matrix', 'distance', 'coordinates', 'ptr']:
        inputs.pop(key)
    inputs['decoder_input_ids'] = inputs['decoder_input_ids'][:,0:1]
    inputs["decoder_attention_mask"] = torch.ones(inputs['decoder_input_ids'].shape[0], 1)
    inputs = {k: v.to(device=torch.cuda.current_device(), non_blocking=True) if hasattr(v, 'to') else v for k, v in inputs.items()}
    # model.to(torch.cuda.current_device())
    encoder_state = model(**inputs, get_graph_emb=True).detach()
    for key in ['edge_index', 'edge_type', 'x', 'encoder_input_ids']:
                inputs.pop(key)
    tok_ids = model.decoder.generate(input_ids=inputs['decoder_input_ids'],
                                    #  attention_mask=inputs["decoder_attention_mask"],
                                     encoder_outputs=encoder_state,
                                     use_cache=True)
    generated += tokenizer.batch_decode(tok_ids, skip_special_tokens=True)
    # print(generated)

data= {'name':names, 'generated': generated, 'function':functions}
df = pd.DataFrame(data)
filename = 'tmp_prot2text_base_old_config.csv'
filepath = os.path.join(os.getcwd(), filename)
df.to_csv(filepath, index=False, mode='a')

res = bleu.compute(predictions=generated, references=functions)
print(res)
