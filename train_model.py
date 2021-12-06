""" Train the skip-gram model. """


from torch.utils.data import DataLoader

from load_data import Dataset
from model import Skipgram


def train(model, dataset, batch_size=32):
  loader = DataLoader(dataset, batch_size=batch_size)
  for sample in loader:
    out = model.forward(*sample)
    print(out.size())


if __name__ == '__main__':
  dataset = Dataset(
    'tests/data/full/vocab.json',
    'tests/data/full/data.txt',
    k=2,
    w=1
  )
  n_words = dataset.vocab.n_words
  model = Skipgram(n_words, 3)
  train(model, dataset, batch_size=4)