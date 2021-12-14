""" Test load_data.py """


import json

from skipgram.load_data import Dataset


def test_full():
  """ Given a small dataset and its corresponding vocabulary, run the whole
  procedure. The data as well as the expected result is in the data folder. """
  dataset = Dataset(
    'tests/data/full/vocab.json',
    'tests/data/full/data.txt',
    k=0
  )
  results = json.load(open('tests/data/full/results.json', encoding='utf-8'))
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
  results = json.load(open('tests/data/long/results.json', encoding='utf-8'))
  for x, y, _ in dataset:
    assert str((x, y)) == results.pop(0)


def test_negsamples():
  """ Ensure that the no. of neg. samples corresponds to the parameter k. """
  for k in range(3):
    dataset = Dataset(
      'tests/data/full/vocab.json', 'tests/data/full/data.txt',
      k=k, w=1
    )
    for _, _, neg in dataset:
      assert neg.size(0)== k
