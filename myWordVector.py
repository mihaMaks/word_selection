import numpy as np
import torch


class WVector:
    def __init__(self, word, lemma, pos, reverse_word, n_neighbours, prefix_size, suffix_size):
        self.word = word
        self.lemma = lemma
        self.pos = pos
        self.reverse_word = reverse_word
        self.embeding = 0
        self.prefix_vec = torch.zeros(prefix_size)
        self.suffix_vec = torch.zeros(suffix_size)
        self.cosine_vec = torch.zeros(n_neighbours)
        self.prefix_sim_mean = 0
        self.suffix_sim_mean = 0
        self.cosine_sim_mean = 0
        self.similarity_vec = torch.zeros(3)
        self.probability_sc = 0
        self.probability_weight = 0

    def __eq__(self, other):
        return self.word == other.word
