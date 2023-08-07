from transformers import GPT2Tokenizer, Seq2SeqTrainingArguments
from prot2text_dataset.torch_geometric_loader import Prot2TextDataset
from prot2text_model.utils import Prot2TextTrainer
from prot2text_model.Model import Prot2TextModel
from prot2text_model.tokenization_prot2text import Prot2TextTokenizer
import torch
import os
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("--model_path", help="path to the prot2text model")
argParser.add_argument("--protein_alphafold_id", default=None, help="the AlphaFold ID of the protein")
argParser.add_argument("--protein_sequence", default=None, help="the amino-acid seuqence of the protein")

# usage:
# python generate_description.py \
#   --model_path ./models/prot2text_base \
#   --protein_alphafold_id \
#   --protein_sequence     


args = argParser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = Prot2TextTokenizer.from_pretrained(args.model_path)
model = Prot2TextModel.from_pretrained(args.model_path)

descrpition = model.generate_protein_description(protein_pdbID=args.protein_alphafold_id,
                                                 protein_sequence=args.protein_sequence, 
                                                 tokenizer=tokenizer,
                                                 device=device)
print()
print(descrpition)
    


