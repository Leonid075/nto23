# Molecule Generation

This folder contains the molecule generation script. The polymer generation experiment in the paper can be reproduced through the following steps:

## Motif Extraction
Extract substructure vocabulary from a given set of molecules (replace `data/polymers/all.txt` with your own molecules):
```
python3 get_vocab.py --min_frequency 100 --ncpu 8 < ../data/all.txt > vocab.txt
```
The `--min_frequency` means to discard any large motifs with lower than 100 occurances in the dataset. The discarded motifs will be decomposed into simple rings and bonds. Change `--ncpu` to specify the number of jobs for multiprocessing.

## Data Preprocessing
Preprocess the dataset using the vocabulary extracted from the first step: 
```
python3 preprocess.py --train ../data/train.txt --vocab ./vocab.txt --ncpu 8 
mkdir train_processed
mv tensor* train_processed/
```

## Training
Train the generative model with KL regularization weight beta=0.1 and VAE latent dimension 24. You can change it by `--beta` and `--latent_size` argument.
```
mkdir -p ckpt/tmp
python3 vae_train.py --train train_processed/ --vocab ./vocab.txt --save_dir ckpt/tmp
```

## Sample Molecules
```
python sample.py --vocab ./vocab.txt --model ../data/model30.pt > outputs.txt
```
