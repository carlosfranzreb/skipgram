""" Given a dictionary with words as keys and counts as values, compute their
frequencies raised to a given power. """


import json


def compute_freqs(count_file, dump_file, power=.75):
  """Compute the frequencies of the counts raised to the given power. 
  count_file (str): file with the dictionary with words and their counts.
  dump_file (str): file where the result should be dumped. 
  power (float): value to which the counts should be raised before computing
  the frequencies. The default value is the recommendation of the authors. """
  vocab = json.load(open(count_file))
  vocab = {word: cnt**power for word, cnt in vocab.items()}
  total = sum(vocab.values())
  vocab = {word: cnt/total for word, cnt in vocab.items()}
  json.dump(vocab, open(dump_file, 'w'))


if __name__ == '__main__':
  count_file = 'tests/data/long/cnt_vocab.json'
  dump_file = 'tests/data/long/vocab.json'
  compute_freqs(count_file, dump_file)