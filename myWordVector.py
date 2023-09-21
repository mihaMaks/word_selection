import torch
import numpy as np


def cosine_similarity(word1, word2, model):
    return np.dot(model[word1], model[word2]) / (np.linalg.norm(model[word1]) * np.linalg.norm(model[word2]))


class WVector:
    def __init__(self, word, trie_data, prefix_size, suffix_size, dim, ft_model):
        self.word = word
        pref = list()
        for i in range(1, min(prefix_size, len(word)) + 1):
            prefix = word[:i]
            pref.append(len(trie_data.keys(prefix))/len(trie_data.keys()))

        suf = list()
        ix = len(word) - min(len(word), suffix_size)
        for i in range(ix, len(word)):
            suffix = word[i:]
            sum_ = 0
            all_ = 0
            for value in trie_data.values():
                if suffix[-1] == value[-1]:
                    all_ += 1
                if value[len(value)-len(word)+i:] == suffix:
                    sum_ += 1
            suf.append(sum_/len(trie_data.keys()))

        # make all vectors same length
        for i in range(dim - min(prefix_size, len(word)) + ix - len(word)):
            pref.append(0)
        for s in suf:
            pref.append(s)
        self.vector = torch.FloatTensor(pref)
        self.dim = dim
        self.string_sim_mean = torch.sum(self.vector)/self.dim
        cosine_list = list()
        for w in trie_data.values():
            cosine_list.append(cosine_similarity(self.word, w, ft_model))
        self.cosine_sim_mean = sum(cosine_list)/len(cosine_list)
        self.probability_score = self.cosine_sim_mean + self.string_sim_mean
        self.probability_weight = 0
