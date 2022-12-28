import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def compute_range(array_fascia, array_mean, array_std, q):
  Z_Scores = []
  greater_than_median = []
  lower_than_median = []
  for ix, value in array_fascia.iteritems():
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


class HumidityAnomalyDetector:
    def __init__(self, pollutant, q=1.5):

        self.pollutant = pollutant
        self.q = q

        self._0_20_mean = 0.0
        self._0_20_std = 0.0

        self._20_40_mean = 0.0
        self._20_40_std = 0.0

        self._40_60_mean = 0.0
        self._40_60_std = 0.0

        self._60_80_mean = 0.0
        self._60_80_std = 0.0

        self._80_100_mean = 0.0
        self._80_100_std = 0.0

        self._0_20_lower_bound = 0.0
        self._0_20_upper_bound = 0.0

        self._20_40_lower_bound = 0.0
        self._20_40_upper_bound = 0.0

        self._40_60_lower_bound = 0.0
        self._40_60_upper_bound = 0.0

        self._60_80_lower_bound = 0.0
        self._60_80_upper_bound = 0.0

        self._80_100_lower_bound = 0.0
        self._80_100_upper_bound = 0.0

    def fit(self, dataset):

        """ Vado a calcolarmi i range """

        df_0_20 = dataset[dataset['humidity'] < 20]
        df_20_40 = dataset[(dataset['humidity'] >= 20) & (dataset['humidity'] < 40)]
        df_40_60 = dataset[(dataset['humidity'] >= 40) & (dataset['humidity'] < 60)]
        df_60_80 = dataset[(dataset['humidity'] >= 60) & (dataset['humidity'] < 80)]
        df_80_100 = dataset[dataset['humidity'] >= 80]

        _0_20_array = df_0_20[self.pollutant]
        _20_40_array = df_20_40[self.pollutant]
        _40_60_array = df_40_60[self.pollutant]
        _60_80_array = df_60_80[self.pollutant]
        _80_100_array = df_80_100[self.pollutant]

        """ Fascia minore 20 """

        self._0_20_mean = _0_20_array.mean()
        self._0_20_std = _0_20_array.std()
        self._0_20_lower_bound, self._0_20_upper_bound = compute_range(_0_20_array,
                                                                       self._0_20_mean,
                                                                       self._0_20_std,
                                                                       q=self.q)
        """ Da 20 a 40 """

        self._20_40_mean = _20_40_array.mean()
        self._20_40_std = _20_40_array.std()
        self._20_40_lower_bound, self._20_40_upper_bound = compute_range(_20_40_array,
                                                                         self._20_40_mean,
                                                                         self._20_40_std,
                                                                         q=self.q)
        """ Da 40 a 60 """

        self._40_60_mean = _40_60_array.mean()
        self._40_60_std = _40_60_array.std()
        self._40_60_lower_bound, self._40_60_upper_bound = compute_range(_40_60_array,
                                                                         self._40_60_mean,
                                                                         self._40_60_std,
                                                                         q=self.q)

        """ Da 20 a 30 gradi """

        self._60_80_mean = _60_80_array.mean()
        self._60_80_std = _60_80_array.std()
        self._60_80_lower_bound, self._60_80_upper_bound = compute_range(_60_80_array,
                                                                         self._60_80_mean,
                                                                         self._60_80_std,
                                                                         q=self.q)

        """ Da 30 a 40 gradi """

        self._80_100_mean = _80_100_array.mean()
        self._80_100_std = _80_100_array.std()
        self._80_100_lower_bound, self._80_100_upper_bound = compute_range(_80_100_array,
                                                                           self._80_100_mean,
                                                                           self._80_100_std,
                                                                           q=self.q)

    def predict_anomaly(self, x, humidity):
        if humidity < 20:
            z = (x - self._0_20_mean) / self._0_20_std
            if z < self._0_20_lower_bound or z > self._0_20_upper_bound:
                return True
        if humidity >= 20 and humidity < 40 :
            z = (x - self._20_40_mean) / self._20_40_std
            if z < self._20_40_lower_bound or z > self._20_40_upper_bound:
                return True
        if humidity >= 40 and humidity < 60 :
            z = (x - self._40_60_mean) / self._40_60_std
            if z < self._40_60_lower_bound or z > self._40_60_upper_bound:
                return True
        if humidity >= 60 and humidity < 80 :
            z = (x - self._60_80_mean) / self._60_80_std
            if z < self._60_80_lower_bound or z > self._60_80_upper_bound:
                return True
        if humidity >= 80 :
            z = (x - self._80_100_mean) / self._80_100_std
            if z < self._80_100_lower_bound or z > self._80_100_upper_bound:
                return True















