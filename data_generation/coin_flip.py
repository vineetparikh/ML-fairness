import numpy as np
class CoinFlipGenerator():
    def __init__(self, flip_probabilities = [0.5]):
        """
        Takes an input a d-dimensional array of flip_probabilities so that 
        generated data has entry a 1 in position i with probability d[i - 1]
        """
        self.flip_probabilities = flip_probabilities

    def get_features(self, n=1000):
        return np.random.binomial(1, self.flip_probabilities, size=(n,len(self.flip_probabilities)))

    def get_labels(self, features):
        """
        gets the labels for the features, where the label is 1 with probability equal
        to the fraction of the features that are 1, and 0 otherwise
        """
        # vector where entry i is fraction of heads in row i
        num_heads = features.sum(axis=1)
        fraction_heads = num_heads/float(features.shape[1])
        labels = np.random.binomial(1, fraction_heads, size=(1,features.shape[0]))
        return labels
# example code for generate 1000 by 5 matrix of coin flips
five_flip_probs = [0.5,0.5,0.5,0.5,0.0]
cfg = CoinFlipGenerator(five_flip_probs)
features = cfg.get_features()
labels = cfg.get_labels(features)



