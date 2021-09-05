# HMM-Tagger

The first project of Udacity's [Natural Language Processing Nanodegree Program](https://www.udacity.com/course/natural-language-processing-nanodegree--nd892) has as its aim to build a part-of-speech tagger using Hidden Markov models. In this repository I will cover all the steps that are required to complete this project.

My model achieved an accuracy of 97.49 % on the training dataset and 95.87 % on the testing dataset.

### Libraries used:
- The [counter](https://docs.python.org/3/library/collections.html#collections.Counter) and [defaultdict](https://docs.python.org/3/library/collections.html#collections.defaultdict) container datatypes of the module [collections](https://docs.python.org/3/library/collections.html) were used to implement the counting functions.

- The [Pomegranate](https://github.com/jmschrei/pomegranate) library was used to build the Hidden Markov model. This library allows us to specify the two distributions that will serve as parameters to the HMM network, the emission probability distribution and the transition probability distribution, which corresponds to the conditional probability of changing states throughout a sequence. Moreover, we can specify the starting probability distribution, which describes the probability of a sequence starting at a given state.

### Resources:

For details on the dataset, refer to [Chapter 5](http://www.nltk.org/book/ch05.html) of the [NLTK](https://www.nltk.org/book/) book.

For more information about Hidden Markov models you can consult [Chapter A](https://web.stanford.edu/~jurafsky/slp3/A.pdf) of the [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) book.

