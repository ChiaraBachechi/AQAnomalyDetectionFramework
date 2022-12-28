import pandas as pd
from SlidingWindowAnomalyDetection.ZScoreCombinedFilter import CombinedFilter
from datetime import datetime
import numpy as np
import dill
import json
import seaborn as sns #da installare sul server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from AnomalyDetectionSystem_interfaces import *


class TrainCalibratedSlidingWindowAlgorithm(TrainAnomalyDetection_interface):

    def __init__(self, info_dictionary, sensor):
        self.sensor = sensor
        with open(info_dictionary, 'r') as prop_file:
            detector_properties = json.load(prop_file)
            self.window = detector_properties['window']
            self.time_column = detector_properties['time_column']
            self.q = detector_properties['q']
            self.differences_upper_bound = detector_properties['differences_upper_bound']
        print("Trainer initialized correctly")

    def fit(self, dataset):
        sw_no_filter = CombinedFilter(dataset=dataset,
                                         time_col=self.time_column,
                                         column_to_analyze="no",
                                         set_bound=True,
                                         differences_upper_bound=self.differences_upper_bound,
                                         q=self.q,
                                         window=self.window)

        sw_no2_filter = CombinedFilter(dataset=dataset,
                                          time_col=self.time_column,
                                          column_to_analyze="no2",
                                          set_bound=True,
                                          differences_upper_bound=self.differences_upper_bound,
                                          q=self.q,
                                          window=self.window)

        sw_co_filter = CombinedFilter(dataset=dataset,
                                          time_col=self.time_column,
                                          column_to_analyze="co",
                                          set_bound=True,
                                          differences_upper_bound=self.differences_upper_bound,
                                          q=self.q,
                                          window=self.window)

        sw_o3_filter = CombinedFilter(dataset=dataset,
                                           time_col=self.time_column,
                                           column_to_analyze="o3",
                                           set_bound=True,
                                           differences_upper_bound=self.differences_upper_bound,
                                           q=self.q,
                                           window=self.window)



        dill.dump(sw_no_filter, open('sensors/' + self.sensor + '/models/sw_no_filter.dill', 'wb'))
        dill.dump(sw_no2_filter, open('sensors/' + self.sensor + '/models/sw_no2_filter.dill', 'wb'))
        dill.dump(sw_co_filter, open('sensors/' + self.sensor + '/models/sw_co_filter.dill', 'wb'))
        dill.dump(sw_o3_filter, open('sensors/' + self.sensor + '/models/sw_o3_filter.dill', 'wb'))



class CalibratedSlidingWindowAlgorithm(AnomalyDetectionSystem_interface):

    def __init__(self, sensor):
        self.sensor = sensor
        self.sw_no_filter = dill.load(open('sensors/' + self.sensor + '/models/sw_no_filter.dill', 'rb'))
        self.sw_no2_filter = dill.load(open('sensors/' + self.sensor + '/models/sw_no2_filter.dill', 'rb'))
        self.sw_co_filter = dill.load(open('sensors/' + self.sensor + '/models/sw_co_filter.dill', 'rb'))
        self.sw_o3_filter = dill.load(open('sensors/' + self.sensor + '/models/sw_o3_filter.dill', 'rb'))


    def apply_dataframe(self, dataset):
        anomalies_NO = []
        anomalies_NO2 = []
        anomalies_CO = []
        anomalies_O3 = []

        tempi_per_istanza = []
        start_assouluto = datetime.now()
        for ix, value in dataset.iterrows():
            start_time = datetime.now()
            value.phenomenon_time = self.sw_no_filter.fix_time(value.phenomenon_time)

            if self.sw_no_filter.check_anomaly_rt(value.no, value.phenomenon_time):
                anomalies_NO.append(ix)

            if self.sw_no2_filter.check_anomaly_rt(value.no2, value.phenomenon_time):
                anomalies_NO2.append(ix)

            if self.sw_co_filter.check_anomaly_rt(value.co, value.phenomenon_time):
                anomalies_CO.append(ix)

            if self.sw_o3_filter.check_anomaly_rt(value.o3, value.phenomenon_time):
                anomalies_O3.append(ix)
            tempi_per_istanza.append(datetime.now() - start_time)

        print("Tempo medio per istanza: ", np.array(tempi_per_istanza).mean())
        print("Tempo totale: ", datetime.now() - start_assouluto)
        self.sw_no_filter.time_series_plot(self.sensor)
        self.sw_no2_filter.time_series_plot(self.sensor)
        self.sw_co_filter.time_series_plot(self.sensor)
        self.sw_o3_filter.time_series_plot(self.sensor)
        return anomalies_NO, anomalies_NO2, anomalies_CO, anomalies_O3

    def predict_anomaly_no(self,  value, phen_time):
        time = self.sw_no_filter(value, phen_time)
        if self.sw_no_filter.check_anomaly_rt(value, time):
            return True

    def predict_anomaly_no2(self, value, phen_time):
        time = self.sw_no2_filter(value, phen_time)
        if self.sw_no2_filter.check_anomaly_rt(value, time):
            return True

    def predict_anomaly_co(self, value, phen_time):
        time = self.sw_co_filter(value, phen_time)
        if self.sw_co_filter.check_anomaly_rt(value, time):
            return True

    def predict_anomaly_o3(self, value, phen_time):
        time = self.sw_po3_filter(value, phen_time)
        if self.sw_o3_filter.check_anomaly_rt(value, time):
            return True

    def apply_single_row(self, json_values):
        dictionary = {'no': False, 'no2': False, 'co': False, 'o3': False}
        data = json.loads(json_values)
        print("Data:")
        print(data)
        if self.predict_anomaly_no(value=data['no'], phen_time=data['phenomenon_time']):
            dictionary['no'] = True
            print("NO anomalous")
        else:
            print("NO not anomalous")
        if self.predict_anomaly_no2(value=data['no2'],phen_time=data['phenomenon_time']):
            dictionary['no2'] = True
            print("NO2 anomalous")
        else:
            print("NO2 not anomalous")
        if self.predict_anomaly_co(value=data['co'], phen_time=data['phenomenon_time']):
            dictionary['co'] = True
            print("CO anomalous")
        else:
            print("CO not anomalous")
        if self.predict_anomaly_ox(value=data['o3'], phen_time=data['phenomenon_time']):
            dictionary['o3'] = True
            print("O3 anomalous")
        else:
            print("O3 not anomalous")
        print(dictionary)
        return dictionary


