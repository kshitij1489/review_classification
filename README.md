# Text Classification

This repository is a simple demonstration of text classification, where four different types of Neural Network Architectures 
were experimented with, using the TensorFlow 2.1 framework.

The task at hand is to assign labels/categories to only the important words/phrases/sub-sentences in a document.

Here I had used a simple approach of breaking down the whole document into sentences/sub-sentences first.
Then running the text classification algorithm for every sentence in the document.

- text_models.py has the implementation of text classifiers. LSTM, CNN, CNN with GLoVe and MLP are the four
implemented models.

- review_classifier.ipynb is the dashboard which contains the comparison between the different models and hyperparameters.

## References

Pre-trained GLoVe embedding file at https://nlp.stanford.edu/projects/glove/

<a id="1">[1]</a> 
Yoon Kim (2014). 
Convolutional Neural Networks for Sentence Classification. [arxiv link](http://arxiv.org/abs/1408.5882)
