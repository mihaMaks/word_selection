import pygtrie as trie
import myWordVector as w2v
import torch
import fasttext
import random


# python3 three_data.py --data new_ft_nouns_min --model model0.bin --out selected_words_dev.txt --select 200 --prefix 3 --suffix 3
def string_sim(wv1, wv2):
    dot_prod = torch.dot(wv1.vector, wv2.vector)
    norm_prod = torch.norm(wv1.vector) * torch.norm(wv2.vector)
    return dot_prod/norm_prod


def main(args):

    # CREATING A TRIE DATA SO WE CAN CREATE VECTORS LATER
    trie_data = trie.Trie()
    input_data = open(args.data, "r")
    line = input_data.readline()
    line = line.strip("\n")
    while line:
        words = line.split(" ")
        line = input_data.readline()
        line = line.strip("\n")

        trie_data[words[0]] = words[0]

    # CREATING CUSTOM VECTORS
    my_vectors = list()
    pref_size = 4
    suf_size = 4
    if args.prefix:
        pref_size = int(args.prefix)
    if args.suffix:
        suf_size = int(args.suffix)

    model = fasttext.load_model(args.model)
    sum_of_prob_sc = 0
    i = 0
    for word in trie_data.values():
        my_vectors.append(w2v.WVector(word, trie_data, pref_size, suf_size, pref_size + suf_size,
                                      model))
        # SUMMING WVectors PROBABILITY SCORE
        sum_of_prob_sc += my_vectors[-1].probability_score
        i += 1
        print(i)

    # COMPUTING PROBABILITY WEIGHTS
    sanity_check = 0 # checking if probability weights add up to 1
    for a in my_vectors:
        a.probability_weight = (a.string_sim_mean + a.cosine_sim_mean)/sum_of_prob_sc
        sanity_check += a.probability_weight
        # print(a.word, a.vector, a.string_sim_mean, a.cosine_sim_mean, a.probability_weight)
    print(sanity_check)

    selected_words = list()
    selection_size = 10
    if args.select:
        selection_size = int(args.select)
    while selection_size > 0:

        rand_num = random.random()
        cumulative_sum = 0
        for vec in my_vectors:
            cumulative_sum += vec.probability_weight
            if cumulative_sum >= rand_num and vec.word not in selected_words:
                selected_words.append(vec.word)
                selection_size -= 1
                break

    print(selected_words)
    out = open(args.out, "w")
    for sw in selected_words:
        out.write(sw + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='FILE names')
    parser.add_argument("--data", help="data file name", required=True, type=str)
    parser.add_argument("--model", help="Fasttext model file name", required=True, type=str)
    parser.add_argument("--out", help="chosen words file name", required=True, type=str)
    parser.add_argument("--select", help="number of words to select", required=False, type=str)
    parser.add_argument("--prefix", help="size of prefix", required=False, type=str)
    parser.add_argument("--suffix", help="size of suffix", required=False, type=str)

    opt = parser.parse_args()
    main(opt)
