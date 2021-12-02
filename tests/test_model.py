""" Test that the model is correct. """


import torch
from torch import nn
import torch.nn.functional as F

from model import Skipgram


def test_forward():
  """ Test the forward pass with a simple example. """
  model = Skipgram(4, 3)
  model.input_vectors = nn.Embedding.from_pretrained(
    torch.FloatTensor([[0,1,0], [1,0,0], [0,0,1], [1,1,1]])
  )
  model.output_vectors = nn.Embedding.from_pretrained(
    torch.FloatTensor([[0,0,1], [1,1,0], [0,1,1], [1,0,1]])
  )
  out = model.forward(
    torch.LongTensor([0,1]),
    torch.LongTensor([2,3]),
    torch.LongTensor([[2,3], [0,1]])
  )
  prob = F.logsigmoid(torch.FloatTensor([1,1]))
  noise = F.logsigmoid(-torch.eye(2, dtype=torch.float)).sum(dim=1)
  assert torch.all(out == prob + noise)


def test_more_noise():
  """ Test the forward pass with three noise samples instead of two. """
  model = Skipgram(4, 3)
  model.input_vectors = nn.Embedding.from_pretrained(
    torch.FloatTensor([[1,2,0], [3,1,2], [0,0,1], [1,1,1]])
  )
  model.output_vectors = nn.Embedding.from_pretrained(
    torch.FloatTensor([[1,1,1], [2,0,3], [2,1,2], [0,0,4]])
  )
  out = model.forward(
    torch.LongTensor([0,1]),
    torch.LongTensor([0,1]),
    torch.LongTensor([[2,0,3], [0,3,1]])
  )
  prob = F.logsigmoid(torch.FloatTensor([3,12]))
  noise = F.logsigmoid(-torch.FloatTensor([[4,3,0], [6,8,12]])).sum(dim=1)
  assert torch.all(out == prob + noise)
