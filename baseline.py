from utils import *

emission_counts = pair_counts(data.training_set.Y, data.training_set.X)
tag_unigrams = unigram_counts(data.training_set.Y)
tag_bigrams = bigram_counts(data.training_set.Y)
tag_starts = starting_counts(data.training_set.Y)
tag_ends = ending_counts(data.training_set.Y)

basic_model = HiddenMarkovModel(name="base-hmm-tagger")

# TODO: create states with emission probability distributions P(word | tag) and add to the model
# (Hint: you may need to loop & create/add new states)
# emission_unigrams = DiscreteDistribution({k:v/sum(tag_unigrams.values()) for k, v in tag_unigrams.items()})

emission_states = {}
for tag in data.training_set.tagset:
    # total = sum([sum(emission_counts[tag].values()) for tag, words in emission_counts.items()])
    p = {word:count/tag_unigrams[tag] for word, count in emission_counts[tag].items()}

    emission_state = State(DiscreteDistribution(p), name=tag)
    emission_states[tag] = emission_state
    basic_model.add_states(emission_state)

# # TODO: add edges between states for the observed transition frequencies P(tag_i | tag_i-1)
# # (Hint: you may need to loop & add transitions

for tag, count in tag_starts.items():
    basic_model.add_transition(basic_model.start, emission_states[tag], tag_starts[tag] * 1.0 /tag_unigrams[tag])

for tag_pair, count in tag_bigrams.items():
    basic_model.add_transition(emission_states[tag_pair[0]], emission_states[tag_pair[1]], count * 1.0 /tag_unigrams[tag_pair[0]])

for tag, count in tag_ends.items():
    basic_model.add_transition(emission_states[tag], basic_model.end, tag_ends[tag] * 1.0 /tag_unigrams[tag])



# NOTE: YOU SHOULD NOT NEED TO MODIFY ANYTHING BELOW THIS LINE
# finalize the model
basic_model.bake()

assert all(tag in set(s.name for s in basic_model.states) for tag in data.training_set.tagset), \
    "Every state in your network should use the name of the associated tag, which must be one of the training set tags."
assert basic_model.edge_count() == 168, \
    ("Your network should have an edge from the start node to each state, one edge between every " +
     "pair of tags (states), and an edge from each state to the end node.")
HTML('<div class="alert alert-block alert-success">Your HMM network topology looks good!</div>')

hmm_training_acc = accuracy(data.training_set.X, data.training_set.Y, basic_model)
print("training accuracy basic hmm model: {:.2f}%".format(100 * hmm_training_acc))

hmm_testing_acc = accuracy(data.testing_set.X, data.testing_set.Y, basic_model)
print("testing accuracy basic hmm model: {:.2f}%".format(100 * hmm_testing_acc))

assert hmm_training_acc > 0.97, "Uh oh. Your HMM accuracy on the training set doesn't look right."
assert hmm_testing_acc > 0.955, "Uh oh. Your HMM accuracy on the testing set doesn't look right."
HTML('<div class="alert alert-block alert-success">Your HMM tagger accuracy looks correct! Congratulations, you\'ve finished the project.</div>')