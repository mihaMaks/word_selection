# Task: make weighted word selection that will be annotated by linguists

## Data files:
I encountered a problem because of data size so computations were to long  
After discussion with my mentor we decided to use words that were both in
*pretrained fasttext model vocabulary* and in *https://huggingface.co/datasets/cjvt/sloleks* table

- pret_ft_words.txt: contains all nouns and verbs from pretrained fasttext model vocabulary
- checked_words_3.txt: final data from witch selection will be/was made (verbs:nouns ratio == 1:3)  


## Selection codes:

- select_words.py: first code I wrote
- select_words_2.py: improved original code that significantly reduces running time
- select_words_3.py: final code that is a little faster than the second code

## Selected words
    out_checked_words.txt: 
        - prefix_size: 3
        - suffix_size: 3
        - n_neighbours: 5
        - prefix_vec weight: 0.3
        - suffix_vec weight: 0.3
        - cosine_vec: 0.4
        - verbs:nouns == 27:73
    
    out_checked_words_2.txt: 
        - prefix_size: 4
        - suffix_size: 3
        - n_neighbours: 5
        - prefix_vec weight: 0.4
        - suffix_vec weight: 0.3
        - cosine_vec: 0.3
        - verbs:nouns == 26:74

    out_checked_words_3.txt: 
        - prefix_size: 4
        - suffix_size: 2
        - n_neighbours: 5
        - prefix_vec weight: 0.4
        - suffix_vec weight: 0.2
        - cosine_vec: 0.4
        - verbs:nouns == 27:73
