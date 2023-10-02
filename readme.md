# Task: make weighted word selection that will be annotated by linguists

## Data files:
I encountered a problem because of data size so computations were to long  
After discussion with my mentor we decided to use words that were both in
*pretrained fasttext model vocabulary* and in *https://huggingface.co/datasets/cjvt/sloleks* table

- file: pret_ft_words.txt contain all nouns and verbs from pretrained fasttext model vocabulary
- file: checked_words_3.txt is final data from witch selection will be/was made

## Selection scripts:

- select_words.py: first script I wrote
- select_words_2.py: improved original script that significantly reduces running time
- select_words_3.py: final script that is a little faster than the second script

## Selected words
This file is still in development mode