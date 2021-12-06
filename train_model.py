""" Train the skip-gram model. """


import logging
from time import time
import os
import json

import torch
from torch.utils.data import DataLoader
from torch.optim import SparseAdam
from torch.nn.utils import clip_grad_norm_

from load_data import Dataset
from model import Skipgram


class ModelTrainer:
  def __init__(self, run_id, model, dataset):
    """ Initialize the trainer. 
    run_id (int): ID of this training run; used to save embeddings.
    model (torch.nn): model to be trained.
    dataset (torch's Dataset): dataset to be used. """
    self.run_id = run_id
    self.model = model
    self.dataset = dataset

  def train(self, batch_size=32, n_epochs=5, lr=.002):
    optimizer = SparseAdam(self.model.parameters(), lr=lr)
    for epoch in range(n_epochs+1):
      loader = DataLoader(self.dataset, batch_size=batch_size)
      self.cnt, self.current_loss = 0, 0  # for last 100 batches
      self.epoch_cnt, self.epoch_loss = 0, 0  # for epoch
      logging.info(f'Starting epoch {epoch+1}')
      for batch in loader:
        optimizer.zero_grad()
        loss = self.model(*batch)
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=5)
        optimizer.step()
        self.cnt += 1
        self.current_loss += loss
        if self.cnt % 100 == 0:
          self.log_loss()
      avg_loss = self.epoch_loss / self.epoch_cnt
      logging.info(f'Avg. loss of epoch {n_epochs+1}: {avg_loss}')
      self.save_embeddings()
  
  def log_loss(self):
    """ Log avg. loss of the last 100 batches. Before resetting the cnt and
    current_loss, add them to the totals for the epoch. """
    avg_loss = self.current_loss / self.cnt
    self.epoch_loss += self.current_loss
    self.epoch_cnt += self.cnt
    self.cnt = 0
    self.current_loss = 0
    logging.info(f'Avg. loss in the last 100 batches: {avg_loss}')
  
  def save_embeddings(self, epoch):
    """ Save the embeddings of the model as a dict with the words as keys and
    the embeddings as values. The file should be named
    'epoch_{epoch}', in the 'run_id' folder. """
    folder = f'embeddings/{self.run_id}'
    if not os.path.exists(folder):
      os.makedir(folder)
    if not os.path.exists(f'{folder}/entries.json'):
      json.dump(self.dataset.vocab.entries, open(f'{folder}/entries.json', 'w'))
    torch.save(self.model.state_dict(), f'{folder}/epoch_{epoch}.pt')
      

def init_training(run_id, vocab_file, data_file, neg_samples, window,
    batch_size, n_dims):
  """ Configure logging, log the parameters of this training procedure and
  initialize training. """
  logging.basicConfig(
    filename=f'logs/training_{run_id}.log',
    level=logging.INFO
  )
  logging.info('Training embeddings with the following parameters:')
  logging.info(f'Vocab file: {vocab_file}')
  logging.info(f'Data file: {data_file}')
  logging.info(f'No. of negative samples: {neg_samples}')
  logging.info(f'No. of context words at each side: {window}')
  logging.info(f'Batch size: {batch_size}')
  logging.info(f'No. of dimensions of the embeddings: {n_dims}')
  dataset = Dataset( vocab_file, data_file, k=neg_samples, w=window)
  logging.info(f'Dataset has {dataset.vocab.n_words} words\n\n')
  loader = DataLoader(dataset, batch_size=batch_size)
  model = Skipgram(dataset.vocab.n_words, n_dims)
  trainer = ModelTrainer(run_id, model, loader)
  trainer.train()


if __name__ == '__main__':
  run_id = int(time())  # ID of this training run
  vocab_file = 'tests/data/full/vocab.json'
  data_file = 'tests/data/full/data.txt'
  neg_samples = 2
  window = 1
  batch_size = 4
  n_dims = 3
  init_training(vocab_file, data_file, neg_samples, window, batch_size, n_dims)
