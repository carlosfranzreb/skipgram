# Skip-gram model

Implementation of the Skip-gram model, as detailed in the paper _Distributed Representations of Words and Phrases and their Compositionality_, by Tomas Mikolov. This is the extended version, which includes negative sampling and subsampling of frequent words.

You can learn about the expected behavior of the scripts by looking at the tests, located in the `tests` folder. They include simple examples that illustrate how the classes work.

## The model

When initialized, it receives as inputs the size of the vocabulary and the number of dimensions each vector representation should have. These are used to initialize the [embeddings](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html).

The forward method of the model receives three tensors with indices:

1. Input indices: indices of the words that are fed into the model.
2. Output indices: indices of the context words we want to compute a score for.
3. Noise indices: indices of the words used for negative sampling (explained above).

Give a batch size _N_, the input and output indices should be 1D tensors of size _N_, while the noise indices are given as a 2D tensor of size _(N x k)_, where k is the number of negative samples used for each input-output pair.

The authors recommend _k_ to be between as low as 2 for large datasets, and as large as 20 for smaller ones.

The forward method returns the score of the input-output pair, considering the negative samples. The higher the score is, the more likely it is that the output word can be found near the input word in the data. The formula for the score function can be found at the bottom of p. 3 of the paper.

## Negative Sampling

The Dataset (see `load_data.py`) samples _k_ negative samples for each pair of center and context words. The number is set when initializing the Dataset. If it is not explicitly set, the default value is 15. In the paper they recommend values between 5 and 20 for small datasets. For larger ones, 2 neg. samples may suffice.

Words are sampled using the unigram distribution, as this is the best performing distribution according to the authors. This is basically a weighted uniform distribution, where the frequencies of the words are the weights.

The authors state that the unigram distribution raised to three quarters perform best. This means that the counts of the words are raised to 0.75 when computing the frequencies. This increases the frequencies of words that appear less in the corpus. Given a dictionary with the words as keys and their counts as values, you can compute frequencies raised to any power with the script `compute_frequencies.py`.

## The data

Two data files are required. One containing the terms of the vocabulary, with a term per line, and another containing the text data, with a sentence per line. Only words present in the vocabulary can appear in the text data.

Usually, you will want the data to be already tokenized and lemmatized. This can be achieved by running the script `process_data.py` (TODO).

## The results

After running the `train_model.py` script, the state of the model will be stored in the `embeddings` folder, under the timestamp when the training started. The avg. score of each epoch, as well as from every 100 batches is logged in the file named with the same timestamp used to name the folder where the model is stored.

You can retrieve the embeddings with the script `get_embeddings.py`. The embeddings will be stored as a dictionary, with the words as keys and their vector representations as values.


## Installing the package

1. Install the EGG file by running the command `python setup.py install`.
2. Move to the `dist`folder and run the command `pip install skipgram-0.0.1-py3.7.egg` to install the module.
    * You may need to change the Python version in the name of the EGG file.
3. You can then access the module from other projects by importing it (`import skipgram`).
