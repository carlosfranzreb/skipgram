""" Lazily iterate over the data. """


from torch.utils.data import IterableDataset

from vocab import Vocab
from word import Word


class Dataset(IterableDataset):
  def __init__(self, vocab_file, data_file, window=2):
    """ Initializes the dataset object, which is fed to PyTorch's DataLoader.
    vocab_file (str): refers to a JSON file with the vocab. Used to initialize
      the Vocab class.
    data_file (str): refers to a TXT file with one sentence per line.
    context (int): no. of words (forwards and backwards) that form the context
    of a center word. They must belong to the same sentence as the center word.
    """
    self.vocab = Vocab(vocab_file)
    self.data = open(data_file, encoding='utf-8')
    self.window = window
    self.get_sentence()

  
  def get_sentence(self):
    """ Read a line from the file and store its words as a list. Set the index
    of the current center word to zero. """
    line = self.data.readline()
    if len(line) == 0:
      self.sentence = None
    else:
      words = line.replace('\n', '').split(' ')
      self.sentence = [self.get_word(words, i) for i in range(len(words))]
      self.center = self.sentence[0]  # center starts at the first word
      until = self.window+1 if len(self.sentence) > self.window else len(self.sentence)
      self.context = [self.sentence[i] for i in range(1, until)]
  
  def get_word(self, sentence, idx):
    """ Return a Word object for the idx-th word of the sentence. """
    vocab_idx = self.vocab.get_idx(sentence[idx])
    return Word(sentence[idx], idx, vocab_idx)
  
  def __iter__(self):
    while self.sentence is not None:
      yield [(self.center.vocab_idx, c.vocab_idx) for c in self.context]
      self.step()
  
  def step(self):
    """ Move the center word to the left if possible and update the context.
    If the end of the sentence has been reached, call get_sentence(). """
    center_idx = self.center.sentence_idx + 1  # new center index
    if center_idx == len(self.sentence):
      self.get_sentence()
    else:
      if center_idx > self.window:  # left context is full
        self.context.pop(0)  # remove left-most word
        self.context[self.window-1] = self.center  # replace new center with old
      else:  # left context is not full; removal not necessary
        self.context[center_idx-1] = self.center  # insert old center
      if center_idx < len(self.sentence) - self.window:  # update right context      
        self.context.append(self.sentence[center_idx + self.window])
      self.center = self.sentence[center_idx]  # update center
