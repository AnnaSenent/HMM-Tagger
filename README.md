# HMM-Tagger

The first project of [Udacity's](https://www.udacity.com/) [Natural Language Processing Nanodegree Program](https://www.udacity.com/course/natural-language-processing-nanodegree--nd892) consists of building a part-of-speech tagger using Hidden Markov models. In this repository I will cover all the steps that are required to complete this project.

My model achieved an accuracy of 97.49 % on the trianing dataset and 95.87 % on the testing dataset.

### Libraries used:
- The [counter](https://docs.python.org/3/library/collections.html#collections.Counter) and [defaultdict](https://docs.python.org/3/library/collections.html#collections.defaultdict) container datatypes of the module [collections](https://docs.python.org/3/library/collections.html) were used to implement the counting functions.

- The [Pomegranate](https://github.com/jmschrei/pomegranate) library was used to build the Hidden Markov model. This library allows us to specify the two distributions that will serve as parameters to the HMM network, the emission probability distribution and the transition probability distribution, which corresponds to the conditional probability of changing states in a given sequence. Moreover, we can specify the starting probability distribution, which describes the probability of a sequence starting at that state.

For details on the dataset, refer to Chapter 5 of the NLTK book.

