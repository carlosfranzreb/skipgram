""" Test load_data.py """


import json

from load_data import Dataset


def test_full():
  """ Given a small dataset and its corresponding vocabulary, run the whole
  procedure. The data as well as the expected result is in the data folder. """
  dataset = Dataset('tests/data/full/vocab.json', 'tests/data/full/data.txt')
  results = json.load(open('tests/data/full/results.json'))
  for data in dataset:
    assert str(data) == results.pop(0)


def test_long():
  """ Test just one sentence, but with a longer window. """
  dataset = Dataset(
    'tests/data/long/vocab.json', 
    'tests/data/long/data.txt',
    window=3
  )
  results = json.load(open('tests/data/long/results.json'))
  for data in dataset:
    assert str(data) == results.pop(0)
