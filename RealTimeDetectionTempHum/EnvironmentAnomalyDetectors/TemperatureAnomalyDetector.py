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


class TemperatureAnomalyDetector:
    def __init__(self, pollutant, q=1.5):

        
        self.pollutant = pollutant
        self.q = q

        self.meno_10_0_mean = 0.0
        self.meno_10_0_std = 0.0

        self._0_10_mean = 0.0
        self._0_10_std = 0.0

        self._10_20_mean = 0.0
        self._10_20_std = 0.0

        self._20_30_mean = 0.0
        self._20_30_std = 0.0

        self._30_40_mean = 0.0
        self._30_40_std = 0.0

        self._40_50_mean = 0.0
        self._40_50_std = 0.0

        self.meno_10_0_lower_bound = 0.0
        self.meno_10_0_upper_bound = 0.0

        self._0_10_lower_bound = 0.0
        self._0_10_upper_bound = 0.0

        self._10_20_lower_bound = 0.0
        self._10_20_upper_bound = 0.0

        self._20_30_lower_bound = 0.0
        self._20_30_upper_bound = 0.0

        self._30_40_lower_bound = 0.0
        self._30_40_upper_bound = 0.0

        self._40_50_lower_bound = 0.0
        self._40_50_upper_bound = 0.0

    def fit(self, dataset):

        """ Vado a calcolarmi i range """

        df_meno_10_0 = dataset[dataset['temperature'] < 0]
        df_0_10 = dataset[(dataset['temperature'] >= 0) & (dataset['temperature'] < 10)]
        df_10_20 = dataset[(dataset['temperature'] >= 10) & (dataset['temperature'] < 20)]
        df_20_30 = dataset[(dataset['temperature'] >= 20) & (dataset['temperature'] < 30)]
        df_30_40 = dataset[(dataset['temperature'] >= 30) & (dataset['temperature'] < 40)]
        df_40_50 = dataset[dataset['temperature'] >= 40]

        meno_10_0_array = df_meno_10_0[self.pollutant]
        _0_10_array = df_0_10[self.pollutant]
        _10_20_array = df_10_20[self.pollutant]
        _20_30_array = df_20_30[self.pollutant]
        _30_40_array = df_30_40[self.pollutant]
        _40_50_array = df_40_50[self.pollutant]

        """ Fascia minore 0"""

        self.meno_10_0_mean = meno_10_0_array.mean()
        self.meno_10_0_std = meno_10_0_array.std()
        self.meno_10_0_lower_bound, self.meno_10_0_upper_bound = compute_range(meno_10_0_array,
                                                                               self.meno_10_0_mean,
                                                                               self.meno_10_0_std,
                                                                               q=self.q)
        """ Da 0 a 10 gradi """

        self._0_10_mean = _0_10_array.mean()
        self._0_10_std = _0_10_array.std()
        self._0_10_lower_bound, self._0_10_upper_bound = compute_range(_0_10_array,
                                                                       self._0_10_mean,
                                                                       self._0_10_std,
                                                                       q=self.q)
        """ Da 10 a 20 gradi """

        self._10_20_mean = _10_20_array.mean()
        self._10_20_std = _10_20_array.std()
        self._10_20_lower_bound, self._10_20_upper_bound = compute_range(_10_20_array,
                                                                         self._10_20_mean,
                                                                         self._10_20_std,
                                                                         q=self.q)

        """ Da 20 a 30 gradi """

        self._20_30_mean = _20_30_array.mean()
        self._20_30_std = _20_30_array.std()
        self._20_30_lower_bound, self._20_30_upper_bound = compute_range(_20_30_array,
                                                                         self._20_30_mean,
                                                                         self._20_30_std,
                                                                         q=self.q)

        """ Da 30 a 40 gradi """

        self._30_40_mean = _30_40_array.mean()
        self._30_40_std = _30_40_array.std()
        self._30_40_lower_bound, self._30_40_upper_bound = compute_range(_30_40_array,
                                                                         self._30_40_mean,
                                                                         self._30_40_std,
                                                                         q=self.q)
        """ Da 40 a 50 gradi """

        self._40_50_mean = _40_50_array.mean()
        self._40_50_std = _40_50_array.std()
        self._40_50_lower_bound, self._40_50_upper_bound = compute_range(_40_50_array,
                                                                         self._40_50_mean,
                                                                         self._40_50_std,
                                                                         q=self.q)

    def predict_anomaly(self, x, temperature):
        if temperature < 0:
            z = (x - self.meno_10_0_mean) / self.meno_10_0_std
            if z < self.meno_10_0_lower_bound or z > self.meno_10_0_upper_bound:

                return True
        if temperature >= 0 and temperature < 10 :
            z = (x - self._0_10_mean) / self._0_10_std
            if z < self._0_10_lower_bound or z > self._0_10_upper_bound:
                return True
        if temperature >= 10 and temperature < 20 :
            z = (x - self._10_20_mean) / self._10_20_std
            if z < self._10_20_lower_bound or z > self._10_20_upper_bound:
                return True
        if temperature >= 20 and temperature < 30 :
            z = (x - self._20_30_mean) / self._20_30_std
            if z < self._20_30_lower_bound or z > self._20_30_upper_bound:
                return True
        if temperature >= 30 and temperature < 40 :
            z = (x - self._30_40_mean) / self._30_40_std
            if z < self._30_40_lower_bound or z > self._30_40_upper_bound:
                return True
        if temperature >= 40 :
            z = (x - self._40_50_mean) / self._40_50_std
            if z < self._40_50_lower_bound or z > self._40_50_upper_bound:
                return True
















