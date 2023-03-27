import numpy as np
from histogram_filter import HistogramFilter

'''
Test for Bayes filter
'''

# load the data
data = np.load(open('starter.npz', 'rb'))
cmap = data['arr_0']
actions = data['arr_1']
observations = data['arr_2']
belief_states = data['arr_3']

# test your code here
bayes_filter = HistogramFilter()

n, m = cmap.shape
belief = np.full((n, m), 1/(n*m))
posterior = np.zeros((len(actions), 2), dtype=int)

for i in range(len(actions)):
    belief = bayes_filter.histogram_filter(cmap, belief, actions[i], observations[i])
    belief = np.flip(belief, 0)
    argmax = np.argmax(belief)
    posterior[i, 0] = argmax % m
    posterior[i, 1] = argmax // m
    belief = np.flip(belief, 0)

print("bayes filter correct implementation: %s" % np.array_equal(belief_states[-1], posterior[-1]), "\n")
