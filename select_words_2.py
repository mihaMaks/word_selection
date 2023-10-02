import pygtrie as trie
import myWordVector as w2v
import torch
import fasttext
import random
import time

# python3 select_words_2.py --data checked_words_3.txt --model model.bin --out out_dev.txt --select 20 --prefix 3 --suffix 3

def reverse_slicing(s):
    return s[::-1]


def is_pref(a, b):
    for ch1, ch2 in zip(a, b):
        if ch1 != ch2:
            return False
    return True


def make_string_vec(i, mwv_word, mwv_vec, trie_data, data_size, first):
    c = 0
    for k in trie_data.keys():
        if is_pref(mwv_word[:i], k):
            c += 1

    if first:
        mwv_vec[i-1] = c / data_size
    else:
        d = 0
        for k in trie_data.keys():
            if mwv_word[0] in k:
                d += 1
        mwv_vec[i-1] = c / d


def main(args):
    t = time.time()
    # CREATING A TRIE DATA SO WE CAN CREATE VECTORS LATER
    suff_data = dict()
    my_dict = dict()
    input_data = open(args.data, "r")
    line = input_data.readline()
    line = line.strip("\n")
    # PARAMETERS
    pref_size = 4
    if args.prefix:
        pref_size = int(args.prefix)
    suff_size = 4
    if args.suffix:
        suff_size = int(args.suffix)

    model = fasttext.load_model(args.model)
    n_neighbours = 5
    weights = torch.ones(3)
    weight1 = 0.3
    if args.w1:
        weights[0] = args.w1
    weight2 = 0.3
    if args.w2:
        weights[1] = args.w2
    weights[2] = 1 - weight1 - weight2
    data_size = 0
    while line:
        words = line.split(" ")
        rev_w = reverse_slicing(words[0])

    # CREATING CUSTOM CLASS myWordVector
        pair = {words[0]: w2v.WVector(words[0], words[1], words[2], rev_w, n_neighbours, pref_size, suff_size)}
        my_dict.update(pair)
        pair = {rev_w: rev_w}
        suff_data.update(pair)
        data_size += 1

        line = input_data.readline()
        line = line.strip("\n")


    # CALCULATING myWordVector PARAMETERS
    sum_of_prob_sc = 0
    q = 0
    for key in my_dict.keys():
        v = my_dict[key]
        for i in range(1, min(len(v.word), pref_size) + 1):
            make_string_vec(i, v.word, v.prefix_vec, my_dict, data_size, i == 1)
            make_string_vec(i, v.reverse_word, v.suffix_vec, suff_data, data_size, i == 1)
        v.similarity_vec[0] = torch.norm(v.prefix_vec)
        v.similarity_vec[1] = torch.norm(v.suffix_vec)
        for n, i in zip(model.get_nearest_neighbors(v.word, k=n_neighbours), range(n_neighbours)):
            v.cosine_vec[i] = n[0]
        v.similarity_vec[2] = torch.norm(v.cosine_vec)
        v.probability_sc = torch.dot(weights, v.similarity_vec)
        sum_of_prob_sc += v.probability_sc
        print(v.word)
        q += 1
        if q == 200:
            break

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
                selection_size -= 1
                break

    selected_words.sort(key=lambda x: x.word)
    out = open(args.out, "w")
    for sw in selected_words:
        out.write(sw.word + " " + sw.lemma + " " + sw.pos + "\n")
    print(time.time() - t)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='FILE names')
    parser.add_argument("--data", help="data file name", required=True, type=str)
    parser.add_argument("--model", help="Fasttext model file name", required=True, type=str)
    parser.add_argument("--out", help="chosen words file name", required=True, type=str)
    parser.add_argument("--select", help="number of words to select", required=False, type=str)
    parser.add_argument("--prefix", help="size of prefix", required=False, type=str)
    parser.add_argument("--suffix", help="size of suffix", required=False, type=str)
    parser.add_argument("--w1", help="weight importance", required=False, type=str)
    parser.add_argument("--w2", help="weight importance", required=False, type=str)

    opt = parser.parse_args()
    main(opt)
