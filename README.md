# Text Classification

This repository is a simple demonstration of text classification, where four different types of Neural Network Architectures 
were experimented with, using the TensorFlow 2.1 framework.

The task at hand is to assign labels/categories to only the important words/phrases/sub-sentences in a document, in a multi-label classification setting.

### Approach
1. Breaking down the whole document into sentence/sub-sentence blocks
2. Using the learnt text classification model for every text block to predict categories, if any.
   - Being a multi-label classification task, the final layer is a fully connected layer with sigmoid activation.
   - And Binary Crossentropy used as the loss function.
   
### Model architectures
Model architectures experimented with: 
* LSTM 
   - Input -> Word Embedding Layer -> LSTM -> Spatial Dropout -> Fully Connected -> RELU -> Fully Connected -> Sigmoid
* CNN [1]
   - Input -> Word Embedding Layer -> Convolution Layer -> RELU -> Global Max Pool -> Fully Connected -> Sigmoid
* CNN with GLove Embedding
   - Input -> GLoVe Word Embedding -> Convolution Layer -> RELU -> Global Max Pool -> Fully Connected -> Sigmoid
* MLP
   - Input -> Word Embedding Layer -> 4x(Fully Connected -> RELU) -> Fully Connected -> Sigmoid 
* Bi-LSTM with Attention [2]
   - Input -> Word Embedding Layer -> Bi-LSTM -> Attention layer -> Fully Connected -> Sigmoid
   



- review_classifier.ipynb is the dashboard which trains, tests and fine tune all the 5 mentioned models. This notebook takes you through a step-by-step process of data pre-processing to Model selection.

- text_models.py has the implementation of text classifiers. LSTM, CNN, CNN with GLoVe and MLP are the four
implemented models.

- lstm_attention_model.py Contains the implementation of the model in paper[2]

- helper.py Contains some custom utility functions for visualization and data pre-processing.

## References

Pre-trained GLoVe embedding file at https://nlp.stanford.edu/projects/glove/

<a id="1">[1]</a> 
Yoon Kim (2014). 
Convolutional Neural Networks for Sentence Classification. [arxiv link](http://arxiv.org/abs/1408.5882)

<a id="1">[2]</a> 
P Zhou (2016). 
Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification [paper](https://www.aclweb.org/anthology/P16-2034.pdf)

## Requirement
- Python3
- TensorFlow >= 2.1

