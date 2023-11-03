# Task: make weighted word selection that will be annotated by linguists

## Data files:
I encountered a problem because of data size so computations were to long  
After discussion with my mentor we decided to use words that were both in
*pretrained fasttext model vocabulary* and in *https://huggingface.co/datasets/cjvt/sloleks* table


- pret_ft_words.txt: contains all nouns and verbs from pretrained fasttext model vocabulary
- checked_words_3.txt: final data from witch selection will be/was made (verbs:nouns ratio == 1:3)  
- checked_words_4.txt: same as version 3 but removed all the words starting with capital letters because most of those words are noise
  (names, surnames, english words)

## Selection codes:

- select_words.py: first code I wrote
- select_words_2.py: improved original code that significantly reduces running time
- select_words_3.py: final code that is a little faster than the second code
- select_words_4.py: code without cosine similarity vector constructed from pretrained fasttext model 

## Summary
There is a big difference between selection files 1-3 and 4-7.
The distribution of words is more bias in files 4-7. See: *[Picture: destribution](prefix_distribution.png)*.  
Those files were generated using *[select_words_4.py](select_words_4.py)*.
For that reason best selection is one of the first three.
When deciding which is best I  run some metrics that show:
- noise: all wards that start with capital letter(names, surnames, english words)
- root_diff_sum: value calculated for all pairs of words with the same prefix, using Levenshtein distance: [def make_root_vec()](select_words_4.py)

With those two metrics and distribution in mind I suggest that [out_checked_words_2.txt](out_checked_words_2.txt)

## Selected words files
    out_checked_words.txt: 
        - prefix_size: 3
        - suffix_size: 3
        - n_neighbours: 5
        - prefix_vec weight: 0.3
        - suffix_vec weight: 0.3
        - cosine_vec weight: 0.4
        - verbs:nouns == 27:73
        - root_diff_sum: 1227053
        - noise:  252

    out_checked_words_2.txt: 
        - prefix_size: 4
        - suffix_size: 3
        - n_neighbours: 5
        - prefix_vec weight: 0.4
        - suffix_vec weight: 0.3
        - cosine_vec weight: 0.4
        - verbs:nouns == 26:74
        - root_diff_sum: 1295907
        - noise: 236

    out_checked_words_3.txt: 
        - prefix_size: 4
        - suffix_size: 2
        - n_neighbours: 5
        - prefix_vec weight: 0.4
        - suffix_vec weight: 0.2
        - cosine_vec weight: 0.4
        - verbs:nouns == 27:73
        - root_diff_sum: 1233986
        - noise:  242

    out_checked_words_4.txt: 
        - prefix_size: 3
        - suffix_size: 3
        - no nearest neighbour  
        - prefix_vec weight: 0.35
        - suffix_vec weight: 0.35
        - no csine_vec
        - verbs:nouns == 49.5:50.5
        - root_diff_sum: 35814277 
        - noise:  0

    out_checked_words_5.txt: 
        - prefix_size: 3
        - suffix_size: 3
        - root_size: 4
        - no nearest neighbour  
        - prefix_vec weight: 0.4
        - suffix_vec weight: 0.2
        - no cosine_vec
        - verbs:nouns == 40:60
        - root_diff_sum: 33239393
        - noise:  1

    out_checked_words_6.txt: 
        - prefix_size: 3
        - suffix_size: 4
        - root_size: 3
        - no nearest neighbour  
        - prefix_vec weight: 0.3
        - suffix_vec weight: 0.4
        - no cosine_vec
        - verbs:nouns == 38:62
        - root_diff_sum: 21670573
        - noise:  0

    out_checked_words_7.txt: 
        - prefix_size: 2
        - suffix_size: 3
        - root_size: 5
        - no nearest neighbour  
        - prefix_vec weight: 0.3
        - suffix_vec weight: 0.4
        - no cosine_vec
        - verbs:nouns == 50:50
        - root_diff_sum: 35526875
        - noise:  0