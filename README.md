# Skip-gram model

Implementation of the Skip-gram model, as detailed in the paper _Distributed Representations of Words and Phrases and their Compositionality_, by Tomas Mikolov. This is the extended version, which includes negative sampling and subsampling of frequent words.

## The model

When initialized, it receives as inputs the size of the vocabulary and the number of dimensions each vector representation should have. These are used to initialize the [embeddings](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html).

The forward method of the model receives three tensors with indices:

1. Input indices: indices of the words that are fed into the model.
2. Output indices: indices of the context words we want to compute a score for.
3. Noise indices: indices of the words used for negative sampling (explained above).

Give a batch size _N_, the input and output indices should be 1D tensors of size _N_, while the noise indices are given as a 2D tensor of size _(N x k)_, where k is the number of negative samples used for each input-output pair.

The authors recommend _k_ to  be between as low as 2 for large datasets, and as large as 20 for smaller ones.

The forward method returns the score of the input-output pair, considering the negative samples. The higher the score is, the more likely it is that the output word can be found near the input word in the data.

The model only considers one context word for each center word. Usually you will want to consider multiple context words for each center words. To do so, just feed the model the input word paired with different words.