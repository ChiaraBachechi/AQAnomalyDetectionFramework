import pandas as pd
from SlidingWindowAnomalyDetection.ZScoreCombinedFilter import CombinedFilter,
from datetime import datetime
import numpy as np
import dill
import json
import seaborn as sns #da installare sul server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from AnomalyDetectionSystem_interfaces import *


class TrainSlidingWindowAlgorithm(TrainAnomalyDetection_interface):

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
        sw_no_we_filter = CombinedFilter(dataset=dataset,
                                         time_col=self.time_column,
                                         column_to_analyze="no_we",
                                         set_bound=True,
#                                         differences_upper_bound=self.differences_upper_bound,
                                         q=self.q,
                                         window=self.window)
        sw_no_we_filter.compute_range()
        sw_no_aux_filter = CombinedFilter(dataset=dataset,
                                          time_col=self.time_column,
                                          column_to_analyze="no_aux",
                                          set_bound=True,
#                                          differences_upper_bound=self.differences_upper_bound,
                                          q=self.q,
                                          window=self.window)
        sw_no_aux_filter.compute_range()
        sw_no2_we_filter = CombinedFilter(dataset=dataset,
                                          time_col=self.time_column,
                                          column_to_analyze="no2_we",
                                          set_bound=True,
#                                          differences_upper_bound=self.differences_upper_bound,
                                          q=self.q,
                                          window=self.window)
        sw_no2_we_filter.compute_range()
        sw_no2_aux_filter = CombinedFilter(dataset=dataset,
                                           time_col=self.time_column,
                                           column_to_analyze="no2_aux",
                                           set_bound=True,
#                                           differences_upper_bound=self.differences_upper_bound,
                                           q=self.q,
                                           window=self.window)
        sw_no2_aux_filter.compute_range()
        sw_co_we_filter = CombinedFilter(dataset=dataset,
                                         time_col=self.time_column,
                                         column_to_analyze="co_we",
                                         set_bound=True,
#                                         differences_upper_bound=self.differences_upper_bound,
                                         q=self.q,
                                         window=self.window)
        sw_co_we_filter.compute_range()
        sw_co_aux_filter = CombinedFilter(dataset=dataset,
                                          time_col=self.time_column,
                                          column_to_analyze="co_aux",
                                          set_bound=True,
#                                          differences_upper_bound=self.differences_upper_bound,
                                          q=self.q,
                                          window=self.window)
        sw_co_aux_filter.compute_range()
        sw_ox_we_filter = CombinedFilter(dataset=dataset,
                                         time_col=self.time_column,
                                         column_to_analyze="ox_we",
                                         set_bound=True,
#                                         differences_upper_bound=self.differences_upper_bound,
                                         q=self.q,
                                         window=self.window)
        sw_ox_we_filter.compute_range()
        sw_ox_aux_filter = CombinedFilter(dataset=dataset,
                                          time_col=self.time_column,
                                          column_to_analyze="ox_aux",
                                          set_bound=True,
#                                          differences_upper_bound=self.differences_upper_bound,
                                          q=self.q,
                                          window=self.window)
        sw_ox_aux_filter.compute_range()
        sw_temperature_filter = CombinedFilter(dataset=dataset,
                                               time_col=self.time_column,
                                               column_to_analyze="temperature",
                                               set_bound=True,
#                                               differences_upper_bound=self.differences_upper_bound,
                                               q=self.q,
                                               window=self.window)
        sw_temperature_filter.compute_range()
        sw_humidity_filter = CombinedFilter(dataset=dataset,
                                            time_col=self.time_column,
                                            column_to_analyze="humidity",
                                            set_bound=True,
#                                            differences_upper_bound=self.differences_upper_bound,
                                            q=self.q,
                                            window=self.window)
        sw_humidity_filter.compute_range()
		print(sw_humidity_filter.differences)
        dill.dump(sw_no_we_filter, open('sensors/' + self.sensor + '/models/sw_no_we_filter.dill', 'wb'))
        dill.dump(sw_no_aux_filter, open('sensors/' + self.sensor + '/models/sw_no_aux_filter.dill', 'wb'))
        dill.dump(sw_no2_we_filter, open('sensors/' + self.sensor + '/models/sw_no2_we_filter.dill', 'wb'))
        dill.dump(sw_no2_aux_filter, open('sensors/' + self.sensor + '/models/sw_no2_aux_filter.dill', 'wb'))
        dill.dump(sw_co_we_filter, open('sensors/' + self.sensor + '/models/sw_co_we_filter.dill', 'wb'))
        dill.dump(sw_co_aux_filter, open('sensors/' + self.sensor + '/models/sw_co_aux_filter.dill', 'wb'))
        dill.dump(sw_ox_we_filter, open('sensors/' + self.sensor + '/models/sw_ox_we_filter.dill', 'wb'))
        dill.dump(sw_ox_aux_filter, open('sensors/' + self.sensor + '/models/sw_ox_aux_filter.dill', 'wb'))
        dill.dump(sw_temperature_filter, open('sensors/' + self.sensor + '/models/sw_temperature_filter.dill', 'wb'))
        dill.dump(sw_humidity_filter, open('sensors/' + self.sensor + '/models/sw_humidity_filter.dill', 'wb'))


class SlidingWindowAlgorithm(AnomalyDetectionSystem_interface):

    def __init__(self, sensor):
        self.sensor = sensor
        self.sw_no_we_filter = dill.load(open('sensors/' + self.sensor + '/models/sw_no_we_filter.dill', 'rb'))
        self.sw_no_aux_filter = dill.load(open('sensors/' + self.sensor + '/models/sw_no_aux_filter.dill', 'rb'))
        self.sw_no2_we_filter = dill.load(open('sensors/' + self.sensor + '/models/sw_no2_we_filter.dill', 'rb'))
        self.sw_no2_aux_filter = dill.load(open('sensors/' + self.sensor + '/models/sw_no2_aux_filter.dill', 'rb'))
        self.sw_co_we_filter = dill.load(open('sensors/' + self.sensor + '/models/sw_co_we_filter.dill', 'rb'))
        self.sw_co_aux_filter = dill.load(open('sensors/' + self.sensor + '/models/sw_co_aux_filter.dill', 'rb'))
        self.sw_ox_we_filter = dill.load(open('sensors/' + self.sensor + '/models/sw_ox_we_filter.dill', 'rb'))
        self.sw_ox_aux_filter = dill.load(open('sensors/' + self.sensor + '/models/sw_ox_aux_filter.dill', 'rb'))
        self.sw_temperature_filter = dill.load(open('sensors/' + self.sensor + '/models/sw_temperature_filter.dill', 'rb'))
        self.sw_humidity_filter = dill.load(open('sensors/' + self.sensor + '/models/sw_humidity_filter.dill', 'rb'))

    def apply_dataframe(self, dataset):
        anomalies_NO = []
        anomalies_NO2 = []
        anomalies_CO = []
        anomalies_OX = []
        anomalies_TEMP_HUM = []
        tempi_per_istanza = []
        start_assouluto = datetime.now()
        for ix, value in dataset.iterrows():
            start_time = datetime.now()
            value.phenomenon_time = self.sw_ox_we_filter.fix_time(value.phenomenon_time)

            if self.sw_no_we_filter.check_anomaly_rt(value.no_we, value.phenomenon_time) or \
                    self.sw_no_aux_filter.check_anomaly_rt(value.no_aux, value.phenomenon_time):
                anomalies_NO.append(ix)

            if self.sw_no2_we_filter.check_anomaly_rt(value.no2_we, value.phenomenon_time) or \
                    self.sw_no2_aux_filter.check_anomaly_rt(value.no2_aux, value.phenomenon_time):
                anomalies_NO2.append(ix)

            if self.sw_co_we_filter.check_anomaly_rt(value.co_we, value.phenomenon_time) or \
                    self.sw_co_aux_filter.check_anomaly_rt(value.co_aux, value.phenomenon_time):
                anomalies_CO.append(ix)

            if self.sw_ox_we_filter.check_anomaly_rt(value.ox_we, value.phenomenon_time) or \
                    self.sw_ox_aux_filter.check_anomaly_rt(value.ox_aux, value.phenomenon_time):
                anomalies_OX.append(ix)

            if self.sw_temperature_filter.check_anomaly_rt(value.temperature, value.phenomenon_time) or \
                    self.sw_humidity_filter.check_anomaly_rt(value.humidity, value.phenomenon_time):
                anomalies_TEMP_HUM.append(ix)
            tempi_per_istanza.append(datetime.now() - start_time)

        print("Tempo medio per istanza: ", np.array(tempi_per_istanza).mean())
        print("Tempo totale: ", datetime.now() - start_assouluto)
        return anomalies_NO, anomalies_NO2, anomalies_CO, anomalies_OX, anomalies_TEMP_HUM

    def predict_anomaly_no(self,  value_we, value_aux, phen_time):
        time = self.sw_no_we_filter.fix_time(phen_time)
        if self.sw_no_we_filter.check_anomaly_rt(value_we, time) or \
                self.sw_no_aux_filter.check_anomaly_rt(value_aux, time):
            return True

    def predict_anomaly_no2(self, value_we, value_aux, phen_time):
        time = self.sw_no2_we_filter.fix_time(phen_time)
        if self.sw_no2_we_filter.check_anomaly_rt(value_we, time) or \
                self.sw_no2_aux_filter.check_anomaly_rt(value_aux, time):
            return True

    def predict_anomaly_co(self, value_we, value_aux, phen_time):
        time = self.sw_co_we_filter.fix_time(phen_time)
        if self.sw_co_we_filter.check_anomaly_rt(value_we, time) or \
                self.sw_co_aux_filter.check_anomaly_rt(value_aux, time):
            return True

    def predict_anomaly_ox(self, value_we, value_aux, phen_time):
        time = self.sw_ox_we_filter.fix_time(phen_time)
        if self.sw_ox_we_filter.check_anomaly_rt(value_we, time) or \
                self.sw_ox_aux_filter.check_anomaly_rt(value_aux, time):
            return True

    def predict_anomaly_temp_hum(self, temp, hum, phen_time):
        time = self.sw_temperature_filter.fix_time(phen_time)
        if self.sw_temperature_filter.check_anomaly_rt(temp, time) or \
                self.sw_humidity_filter.check_anomaly_rt(hum, time):
            return True


    def apply_single_row(self, json_values):
        dictionary = {'phenomenon_time_sensor_raw_observation': None,
                      'id_sensor_low_cost_status': None,
                      'id_anomaly_detection_algorithm': None,
                      'no': False, 'no2': False, 'co': False, 'ox': False, 'temperature': False, 'humidity': False}
        data = json.loads(json_values)
        print("Data:")
        print(data)
        if self.predict_anomaly_no(value_we=data['no_we'], value_aux=data['no_aux'], phen_time=data['phenomenon_time']):
            dictionary['NO'] = True
            print("NO anomalous")
        else:
            print("NO not anomalous")
        if self.predict_anomaly_no2(value_we=data['no2_we'], value_aux=data['no2_aux'], phen_time=data['phenomenon_time']):
            dictionary['NO2'] = True
            print("NO2 anomalous")
        else:
            print("NO2 not anomalous")
        if self.predict_anomaly_co(value_we=data['co_we'], value_aux=data['co_aux'], phen_time=data['phenomenon_time']):
            dictionary['CO'] = True
            print("CO anomalous")
        else:
            print("CO not anomalous")
        if self.predict_anomaly_ox(value_we=data['ox_we'], value_aux=data['ox_aux'], phen_time=data['phenomenon_time']):
            dictionary['OX'] = True
            print("CO anomalous")
        else:
            print("CO not anomalous")
        if self.predict_anomaly_temp_hum(temp=data['temperature'], hum=data['humidity'], phen_time=data['phenomenon_time']):
            print("Temperature and Humidity are anomalous")
            dictionary['Temperature'] = True
            dictionary['Humidity'] = True
        dictionary['id_anomaly_detection_algorithm'] = data['algorithm']
        dictionary['phenomenon_time_sensor_raw_observation'] = data['phenomenon_time']
        dictionary['id_sensor_low_cost_status'] = data['id_sensor_low_cost_status']

        print(dictionary)
        return dictionary


