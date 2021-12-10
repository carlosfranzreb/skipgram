""" Example script of how to train the model. """


from time import time
import logging

from skipgram.train import init_training
from skipgram.embeddings import get_embeddings


run_id = int(time())  # ID of this training run
logging.basicConfig(
  filename=f'logs/training_{run_id}.log',
  level=logging.INFO
)
vocab_file = 'tests/data/full/vocab.json'
data_file = 'tests/data/full/data.txt'
neg_samples = 2
window = 1
batch_size = 4
n_dims = 3
init_training(run_id, vocab_file, data_file, neg_samples, window,
  n_dims, batch_size)  # train the model
get_embeddings(run_id, epoch=5, n_dims=n_dims, dump_file='embeddings.json')
