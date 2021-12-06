""" Test load_data.py """


import json

from load_data import Dataset


def test_full():
  """ Given a small dataset and its corresponding vocabulary, run the whole
  procedure. The data as well as the expected result is in the data folder. """
  dataset = Dataset(
    'tests/data/full/vocab.json',
    'tests/data/full/data.txt',
    k=0
  )
  results = json.load(open('tests/data/full/results.json'))
  for x, y, _ in dataset:
    assert str((x, y)) == results.pop(0)


def test_long():
  """ Test just one sentence, but with a longer window. """
  dataset = Dataset(
    'tests/data/long/vocab.json', 
    'tests/data/long/data.txt',
    w=3,
    k=0
  )
  results = json.load(open('tests/data/long/results.json'))
  for x, y, _ in dataset:
    assert str((x, y)) == results.pop(0)


def test_negsamples():
  """ Ensure that the no. of neg. samples corresponds to the parameter k. """
  dataset = Dataset(
    'tests/data/full/vocab.json', 'tests/data/full/data.txt',
    k=1, w=1
  )
  for x, y, neg in dataset:
    assert neg.size(0)== 1

  dataset = Dataset(
    'tests/data/full/vocab.json', 'tests/data/full/data.txt',
    k=2, w=1
  )
  for x, y, neg in dataset:
    assert neg.size(0)== 2
