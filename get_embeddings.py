""" Store the embeddings as a dictionary, with the word as the key and its
corresponding vector representation as the value. """


import json
import torch

from model import Skipgram


def get_embeddings(timestamp, epoch, n_dims, dump_file):
  """ The embeddings are retrieved from the state file stored in the
  'timestamp' folder, which describes the moment at which the training
  started, and the epoch. The resulting dict is stored in the dump_file.
  n_dims (int): no. of dimensions of the embeddings we want to retrieve.
  timestamp (int): folder where the embeddings are stored.
  epoch (int): epoch after which we want to retrieve the embeddings.
  dump_file (str): desired name for the file in which the results are stored.
  """
  entries = json.load(open(f'embeddings/{timestamp}/entries.json'))
  model = Skipgram(len(entries), n_dims)
  model.load_state_dict(torch.load(f'embeddings/{timestamp}/epoch_{epoch}.pt'))
  vecs = {entries[i]: model.input_vectors(torch.LongTensor([i])).tolist()[0] 
      for i in range(len(entries))}
  json.dump(vecs, open(dump_file, 'w'))


if __name__ == '__main__':
  timestamp = 1638959563
  epoch = 500
  n_dims = 3
  dump_file = 'full_data_vecs.json'
  get_embeddings(timestamp, epoch, n_dims, dump_file)