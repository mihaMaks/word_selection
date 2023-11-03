import pygtrie as trie
import myWordVector2 as w2v
import torch
import fasttext
import random
import pickle
import numpy as np


# python3 select_words_4.py --data checked_words_3.txt --model model.bin --out out_checked_words_8.txt --select 2000 --prefix 2 --suffix 2 --root 4 --pref_w1 0.2 --suff_w2 0.4

def reverse_slicing(s):
    return s[::-1]


def distance(str1, str2):
    """Simple Levenshtein implementation for evalm."""
    m = np.zeros([len(str2) + 1, len(str1) + 1], dtype=int)
    for x in range(1, len(str2) + 1):
        m[x, 0] = m[x - 1, 0] + 1
    for y in range(1, len(str1) + 1):
        m[0, y] = m[0, y - 1] + 1
    for x in range(1, len(str2) + 1):
        for y in range(1, len(str1) + 1):
            if str1[y-1] == str2[x-1]:
                dg = 0
            else:
                dg = 1
            m[x, y] = min(m[x - 1, y] + 1, m[x, y - 1] + 1, m[x - 1, y - 1] + dg)
    return m[len(str2), len(str1)]


def make_string_vec(i, mwv_word, mwv_vec, trie_data, data_size, first, num_words_cache):
    s = mwv_word[:i]
    if first:
        mwv_vec[i-1] = _get_num_words_starting_with_s(trie_data, s, num_words_cache) / data_size
    else:
        mwv_vec[i-1] = _get_num_words_starting_with_s(trie_data, s, num_words_cache) / _get_num_words_starting_with_s(
            trie_data, s[:-1], num_words_cache)


def make_root_vec(my_vec, word, trie_data, pref_size, root_size, suff_size):
    prefix = word[:pref_size]
    dist_sum = 0
    num = 1 # some words are to short so dist_sum = 0
    for w2 in trie_data.keys(prefix):
        if pref_size + root_size <= len(word) - suff_size:
            dist_sum += distance(w2[pref_size:root_size], word[pref_size:root_size])
            num += 1
    my_vec.root_diff_mean = dist_sum/num

def _get_num_words_starting_with_s(trie_data, s, num_words_cache):
    if s not in num_words_cache:
        num_words_cache[s] = len(trie_data.keys(s))
    return num_words_cache[s]


def main(args):
    # CREATING A TRIE DATA SO WE CAN SAVE CUSTOM OBJECTS LATER
    trie_data = trie.Trie()
    suff_data = trie.Trie()
    my_dict = dict()
    # PARAMETERS
    pref_size = 4
    if args.prefix:
        pref_size = int(args.prefix)
    suff_size = 4
    if args.suffix:
        suff_size = int(args.suffix)
    root_size = 3
    if args.root:
        root_size = int(args.root)
    #model = fasttext.load_model(args.model)
    n_neighbours = 5
    weights = torch.ones(3)
    weights[0] = 0.3
    if args.pref_w1:
        weights[0] = float(args.pref_w1)
    weights[1] = 0.3
    if args.suff_w2:
        weights[1] = float(args.suff_w2)
    weights[2] = 1 - weights[0] - weights[1]
    data_size = 0
    input_data = open(args.data, "r")
    line = input_data.readline()
    line = line.strip("\n")

    while line:
        words = line.split(" ")
    # CREATING OBJECTS myWordVector
        rev_w = reverse_slicing(words[0])
        trie_data[words[0]] = words[0]
        pair = {words[0]: w2v.WVector(words[0], words[1], words[2], rev_w, n_neighbours, pref_size, suff_size,
                                      root_size)}
        my_dict.update(pair)
        suff_data[rev_w] = rev_w

        line = input_data.readline()
        line = line.strip("\n")
        data_size += 1

    afile = open('trie_data.pkl', 'wb')
    bfile = open('my_dict.pkl', 'wb')
    cfile = open('suff_data.pkl', 'wb')
    pickle.dump(trie_data, afile)
    pickle.dump(my_dict, bfile)
    pickle.dump(suff_data, cfile)
    afile.close()
    bfile.close()
    cfile.close()
    """
    file1 = open('trie_data.pkl', 'rb')
    file2 = open('my_dict.pkl', 'rb')
    file3 = open('suff_data.pkl', 'rb')
    tire_data = pickle.load(file1) # prefix_data
    my_dict = pickle.load(file2)
    suff_data = pickle.load(file3)
    file1.close()
    file2.close()
    file3.close()
    """

    # CALCULATING myWordVector PARAMETERS
    sum_of_prob_sc = 0
    num_words_cache = {}
    for key in my_dict.keys():
        v = my_dict[key]
        for i in range(1, min(len(v.word), max(pref_size, suff_size)) + 1):
            make_string_vec(i, v.word, v.prefix_vec, trie_data, data_size, i == 1, num_words_cache)
            make_string_vec(i, v.reverse_word, v.suffix_vec, suff_data, data_size, i == 1, num_words_cache)
        make_root_vec(v, v.word, trie_data, pref_size, root_size, suff_size)
        v.similarity_vec[0] = torch.norm(v.prefix_vec)
        v.similarity_vec[1] = torch.norm(v.suffix_vec)
        v.similarity_vec[2] = v.root_diff_mean
        v.probability_sc = torch.dot(weights, v.similarity_vec)
        sum_of_prob_sc += v.probability_sc
        print(v.word)

    # COMPUTING PROBABILITY WEIGHTS
    sanity_check = 0  # checking if probability weights add up to 1
    for a in my_dict.values():
        a.probability_weight = a.probability_sc / sum_of_prob_sc
        sanity_check += a.probability_weight
    print(sanity_check)

    selected_words = list()
    selection_size = 10
    if args.select:
        selection_size = int(args.select)
    while selection_size > 0:

        rand_num = random.random()
        cumulative_sum = 0
        for vec in my_dict.values():
            cumulative_sum += vec.probability_weight
            if cumulative_sum >= rand_num and vec not in selected_words:
                selected_words.append(vec)
                print(vec.word)
                selection_size -= 1
                break

    selected_words.sort(key=lambda x: x.word)
    out = open(args.out, "w")
    for sw in selected_words:
        out.write(sw.word + " " + sw.lemma + " " + sw.pos + " " + str(sw.probability_weight) +"\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='FILE names')
    parser.add_argument("--data", help="data file name", required=True, type=str)
    parser.add_argument("--model", help="Fasttext model file name", required=True, type=str)
    parser.add_argument("--out", help="chosen words file name", required=True, type=str)
    parser.add_argument("--select", help="number of words to select", required=False, type=str)
    parser.add_argument("--prefix", help="size of prefix", required=False, type=str)
    parser.add_argument("--suffix", help="size of suffix", required=False, type=str)
    parser.add_argument("--root", help="size of root", required=False, type=str)
    parser.add_argument("--pref_w1", help="prefix weight importance", required=False, type=str)
    parser.add_argument("--suff_w2", help="suffix weight importance", required=False, type=str)

    opt = parser.parse_args()
    main(opt)
