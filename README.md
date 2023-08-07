# Prot2Text
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]
[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

This repository contains the code to reporoduce the results of the paper "Prot2Text: Multimodal Protein's Function Generation with GNNs and Transformers" by Hadi Abdine, Michail Chatzianastasis, Costas Bouyioukos and Michalis Vazirgiannis. [https://arxiv.org/abs/2307.14367]

## Setup
#### Environment Setup

Recommended environment is Python >= 3.8 and PyTorch 1.13, although other versions of Python and PyTorch may also work.

To prepare the environment we need to do the following steps:
1- Install pytorch 1.13.*, pytorch-geometric and its optional dependencies according to your cuda version using the following links:
- pytorch: [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)
- pytorch-geometric and its optional dependencies: [https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

2- Install the DSSP library following the insctructions here: [https://ssbio.readthedocs.io/en/latest/instructions/dssp.html](https://ssbio.readthedocs.io/en/latest/instructions/dssp.html)

3- Install the rest of the requirements:
```bash
pip install -r Requirements.txt
```

#### Datasets Download
|  Dataset |  Size | Link  |
|----------|-------|-------|
|  Train | 248 315  |  [Download]() | 
|  Validation |  4 172  | [Download]()  | 
| Test  | 4 203  | [Download]()  | 


## Models

|  Model           |  #params |  BLEU Score |  BERT Score | Link         |
|:----------------:|:--------:|:-----------:|:-----------:|:------------:|
|  Prot2Text_SMALL |          |             |             | [Download]() | 
|  Prot2Text_BASE  |          |             |             | [Download]() | 
|  Prot2Text_LARGE |          |             |             | [Download]() | 
|  Esm2Text_BASE   |          |             |             | [Download]() | 

#### Protein Description Generation
To generate the description of a protein using any Prot2Text model you need to specify the protein AlphaFoldDB ID and have an inernet connection in order to download the structure:
```
python generate_description.py \
  --model_path ./models/prot2text_base \
  --protein_alphafold_id P36108
```

You can also use the Esm2Text model to generate protein description based only on the amino-acid sequence:
```
python generate_description.py \
  --model_path ./models/esm2text_base \
  --protein_sequence AEQAERYEEMVEFMEKL
```

#### Training Prot2Text 
To train Prot2Text model on a single GPU:
```
python train.py \
  --decoder_path gpt2 \
  --esm_model_path facebook/esm2_t12_35M_UR50D \
  --use_plm \
  --use_rgcn \
  --warmup_esm \
  --warmup_gpt \    
  --split test \
  --data_path ./data//dataset/ \
  --train_csv_path ./data/train.csv \
  --eval_csv_path ./data/eval.csv \    
  --batch_per_device 4 \
  --nb_epochs 25 \
  --nb_gpus 1 \
  --gradient_accumulation 64 \ 
  --lr 2e-4 \ 
  --save_model_path ./models/prot2text_base/ \
  --bleu_evaluation \
```

To train Prot2Text model on multiple GPUs:
```
python -u -m torch.distributed.run  --nproc_per_node <number_of_gpus> --nnodes <number_of_nodes> --node_rank 0 train.py \
  --decoder_path gpt2 \
  --esm_model_path facebook/esm2_t12_35M_UR50D \
  --use_plm \
  --use_rgcn \
  --warmup_esm \
  --warmup_gpt \    
  --split test \
  --data_path ./data//dataset/ \
  --train_csv_path ./data/train.csv \
  --eval_csv_path ./data/eval.csv \    
  --batch_per_device 4 \
  --nb_epochs 25 \
  --nb_gpus <number_of_gpus> \
  --gradient_accumulation 1 \ 
  --lr 2e-4 \ 
  --save_model_path ./models/prot2text_base/ \
  --bleu_evaluation \
```
An example script for distributed training using SLURM can be also found in this repository.


#### Evaluation

To evaluate Prot2Text model on a single GPU:
```
python evaluate_prot2text.py \
  --model_path ./models/prot2text_base \
  --data_path ./data/dataset/ \
  --split test \
  --csv_path ./data/test.csv \
  --batch_per_device 4 \
  --save_results_path ./results/prot2text_base_results.csv
```

To evaluate Prot2Text model on multiple GPUs:
```
python -u -m torch.distributed.run  --nproc_per_node <number_of_gpus> --nnodes <number_of_nodes> --node_rank 0 evaluate_prot2text.py \
  --model_path ./models/prot2text_base \
  --data_path ./data/dataset/ \
  --split test \
  --csv_path ./data/test.csv \
  --batch_per_device 4 \
  --save_results_path ./results/prot2text_base_results.csv
```