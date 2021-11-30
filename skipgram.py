""" Implementation of the Skip-gram model with neg. sampling and undersampling
of frequent words. """


import torch
from torch import nn
import torch.nn.functional as F


class Skipgram(nn.Module):
  def __init__(self, n_words, n_dims):
    """ n_dims: no. of dimensions of each word.
    context_size: no. of words to predict in the output in each direction.
    frequencies: list of frequencies of each word in the vocabulary.
    neg_samples: no. of neg. samples to be drawn when computing the score. """
    super(Skipgram, self).__init__()
    self.n_words = n_words
    self.n_dims = n_dims
    self.input_vectors = nn.Embedding(n_words, n_dims)
    self.output_vectors = nn.Embedding(n_words, n_dims)

  def forward(self, input_idx, output_idx, neg_idx):
    in_vecs = self.input_vectors(input_idx).unsqueeze(1)
    out_vecs = self.output_vectors(output_idx).unsqueeze(2)
    neg_vecs = -self.output_vectors(neg_idx).transpose(1, 2)
    prob = torch.bmm(in_vecs, out_vecs).squeeze()
    noise = torch.bmm(in_vecs, neg_vecs).squeeze()
    return F.logsigmoid(prob) + F.logsigmoid(noise).sum(dim=1)
