import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def compute_range(array_fascia, array_mean, array_std, q):
    Z_Scores = []
    greater_than_median = []
    lower_than_median = []
    for ix, value in array_fascia.iteritems():  # normalizzo il test
        z = (value - array_mean) / array_std
        Z_Scores.append(z)
    Z_Scores = np.array(Z_Scores)
    Z_Scores_mean = Z_Scores.mean()

    for i in range(len(Z_Scores)):
        if Z_Scores[i] > Z_Scores_mean:
            greater_than_median.append(Z_Scores[i])
        else:
            lower_than_median.append(Z_Scores[i])
    q1 = np.array(lower_than_median).mean()
    q3 = np.array(greater_than_median).mean()
    iqr = q3 - q1
    lower_bound = q1 - q * iqr
    upper_bound = q3 + q * iqr
    return lower_bound, upper_bound


class BasicAnomalyDetector:
    def __init__(self, feature, q=1.5):

        self.feature = feature
        self.q = q

        self.mean = 0.0
        self.std = 0.0

        self.lower_bound = 0.0
        self.upper_bound = 0.0



    def fit(self, dataset):

        """ Vado a calcolarmi i range """

        array = dataset[self.feature]


        """ Fascia minore 0"""

        self.mean = array.mean()
        self.std = array.std()
        self.lower_bound, self.upper_bound = compute_range(array, self.mean, self.std, q=self.q)

    def predict_anomaly(self, x):
        z = (x - self.mean) / self.std
        if z < self.lower_bound or z > self.upper_bound:
            return True










