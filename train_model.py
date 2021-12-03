""" Train the skip-gram model. """


from torch.utils.data import DataLoader

from load_data import Dataset
from model import Skipgram


def train(model, loader):
dataset = Dataset(
  'tests/data/full/vocab.json',
  'tests/data/full/data.txt'
)
loader = DataLoader(dataset, batch_size=4)

for sample in loader:
  print(sample)
