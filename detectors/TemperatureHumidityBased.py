import pandas as pd
from FFIDCAD.FFIDCAD import FFIDCAD
from SlidingWindowAnomalyDetection.ZScoreCombinedFilter import CombinedFilter
from RealTimeDetectionTempHum.EnvironmentAnomalyDetectors.BasicAnomalyDetector import BasicAnomalyDetector
from RealTimeDetectionTempHum.EnvironmentAnomalyDetectors.TemperatureAnomalyDetector import TemperatureAnomalyDetector
from RealTimeDetectionTempHum.EnvironmentAnomalyDetectors.HumidityAnomalyDetector import HumidityAnomalyDetector
from datetime import datetime
import numpy as np
import dill
import json
import seaborn as sns #da installare sul server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from AnomalyDetectionSystem_interfaces import *


class TrainTemperatureHumidityBased(TrainAnomalyDetection_interface):

    def __init__(self, info_dictionary, sensor):
        self.sensor = sensor
        with open(info_dictionary, 'r') as prop_file:
            detector_properties = json.load(prop_file)

            self.q = detector_properties['q']

        print("Trainer initialized correctly")

    def fit(self, dataset):

        """ Fit RealTimeDetectionTempHum"""

        temperature_filter = BasicAnomalyDetector("temperature", self.q)
        temperature_filter.fit(dataset)

        humidity_filter = BasicAnomalyDetector("humidity", self.q)
        humidity_filter.fit(dataset)

        no_we_filter = TemperatureAnomalyDetector("no_we", self.q)
        no_we_filter.fit(dataset)

        no_aux_filter = TemperatureAnomalyDetector("no_aux", self.q)
        no_aux_filter.fit(dataset)

        no2_we_filter = TemperatureAnomalyDetector("no2_we", self.q)
        no2_we_filter.fit(dataset)

        no2_aux_filter = TemperatureAnomalyDetector("no2_aux", self.q)
        no2_aux_filter.fit(dataset)

        co_we_filter = HumidityAnomalyDetector("co_we", self.q)
        co_we_filter.fit(dataset)

        co_aux_filter = HumidityAnomalyDetector("co_aux", self.q)
        co_aux_filter.fit(dataset)

        ox_we_filter = HumidityAnomalyDetector("ox_we", self.q)
        ox_we_filter.fit(dataset)

        ox_aux_filter = HumidityAnomalyDetector("ox_aux", self.q)
        ox_aux_filter.fit(dataset)

        """ Saving Temp Hum based algorithm models """

        dill.dump(no_we_filter, open('sensors/'+self.sensor+'/models/no_we_filter.dill', 'wb'))
        dill.dump(no_aux_filter, open('sensors/' + self.sensor + '/models/no_aux_filter.dill', 'wb'))
        dill.dump(no2_we_filter, open('sensors/' + self.sensor + '/models/no2_we_filter.dill', 'wb'))
        dill.dump(no2_we_filter, open('sensors/' + self.sensor + '/models/no2_aux_filter.dill', 'wb'))
        dill.dump(co_we_filter, open('sensors/' + self.sensor + '/models/co_we_filter.dill', 'wb'))
        dill.dump(co_aux_filter, open('sensors/' + self.sensor + '/models/co_aux_filter.dill', 'wb'))
        dill.dump(ox_we_filter, open('sensors/' + self.sensor + '/models/ox_we_filter.dill', 'wb'))
        dill.dump(ox_aux_filter, open('sensors/' + self.sensor + '/models/ox_aux_filter.dill', 'wb'))
        dill.dump(temperature_filter, open('sensors/' + self.sensor + '/models/temperature_filter.dill', 'wb'))
        dill.dump(humidity_filter, open('sensors/' + self.sensor + '/models/humidity_filter.dill', 'wb'))




class TemperatureHumidityBased(AnomalyDetectionSystem_interface):

    def __init__(self, sensor):
        self.sensor = sensor

        self.no_we_filter = dill.load(open('sensors/' + self.sensor + '/models/no_we_filter.dill', 'rb'))
        self.no_aux_filter = dill.load(open('sensors/' + self.sensor + '/models/no_aux_filter.dill', 'rb'))
        self.no2_we_filter = dill.load(open('sensors/' + self.sensor + '/models/no2_we_filter.dill', 'rb'))
        self.no2_aux_filter = dill.load(open('sensors/' + self.sensor + '/models/no2_aux_filter.dill', 'rb'))
        self.co_we_filter = dill.load(open('sensors/' + self.sensor + '/models/co_we_filter.dill', 'rb'))
        self.co_aux_filter = dill.load(open('sensors/' + self.sensor + '/models/co_aux_filter.dill', 'rb'))
        self.ox_we_filter = dill.load(open('sensors/' + self.sensor + '/models/ox_we_filter.dill', 'rb'))
        self.ox_aux_filter = dill.load(open('sensors/' + self.sensor + '/models/ox_aux_filter.dill', 'rb'))
        self.temperature_filter = dill.load(open('sensors/' + self.sensor + '/models/temperature_filter.dill', 'rb'))
        self.humidity_filter = dill.load(open('sensors/' + self.sensor + '/models/humidity_filter.dill', 'rb'))
        print("Loaded correctly")

    def apply_dataframe(self, df):
        anomalies_NO = []
        anomalies_NO2 = []
        anomalies_CO = []
        anomalies_OX = []
        anomalies_TEMP_HUM = []
        for ix, value in df.iterrows():
            if not self.temperature_filter.predict_anomaly(value.temperature):
                if self.no_we_filter.predict_anomaly(value.no_we, value.temperature) or \
                        self.no_aux_filter.predict_anomaly(value.no_aux, value.temperature):
                    anomalies_NO.append(ix)
            if not self.temperature_filter.predict_anomaly(value.temperature):
                if self.no2_we_filter.predict_anomaly(value.no2_we, value.temperature) and \
                         self.no2_aux_filter.predict_anomaly(value.no2_aux, value.temperature):
                    anomalies_NO2.append(ix)
            else:
                print("Temperature is anomalous")
                anomalies_TEMP_HUM.append(ix)
                anomalies_NO.append(ix)
                anomalies_NO2.append(ix)
            if not self.humidity_filter.predict_anomaly(value.humidity):
                if self.co_we_filter.predict_anomaly(value.co_we, value.humidity) or \
                         self.co_aux_filter.predict_anomaly(value.co_aux, value.humidity):
                    anomalies_CO.append(ix)
                if self.ox_we_filter.predict_anomaly(value.ox_we, value.humidity) or\
                        self.ox_aux_filter.predict_anomaly(value.ox_aux, value.humidity):
                    anomalies_OX.append(ix)
            else:
                print("Humidity is anomalous")
                anomalies_TEMP_HUM.append(ix)
                anomalies_CO.append(ix)
                anomalies_OX.append(ix)
        return anomalies_NO, anomalies_NO2, anomalies_CO, anomalies_OX, anomalies_TEMP_HUM

    def predict_anomaly_no(self,  value_we, value_aux, temp):
        if not self.temperature_filter.predict_anomaly(temp):
            if self.no_we_filter.predict_anomaly(value_we, temp) or\
                    self.no_aux_filter.predict_anomaly(value_aux, temp):
                return True
        else:
            return True

    def predict_anomaly_no2(self, value_we, value_aux, temp):
        if not self.temperature_filter.predict_anomaly(temp):
            if self.no2_we_filter.predict_anomaly(value_we, temp) or\
                    self.no2_aux_filter.predict_anomaly(value_aux, temp):
                return True
        else:
            return True

    def predict_anomaly_co(self, value_we, value_aux, hum):
        if not self.humidity_filter.predict_anomaly(hum):
            if self.co_we_filter.predict_anomaly(value_we, hum) or\
                    self.co_aux_filter.predict_anomaly(value_aux, hum):
                return True
        else:
            return True

    def predict_anomaly_ox(self, value_we, value_aux, hum):
        if not self.humidity_filter.predict_anomaly(hum):
            if self.ox_we_filter.predict_anomaly(value_we, hum) or\
                    self.ox_aux_filter.predict_anomaly(value_aux, hum):
                return True
        else:
            return True

    def predict_anomaly_temp_hum(self, temp, hum):
        if self.humidity_filter.predict_anomaly(hum) or self.temperature_filter.predict_anomaly(temp):
            return True

    def apply_single_row(self, json_values):
        dictionary = {'phenomenon_time_sensor_raw_observation':None,
                      'id_sensor_low_cost_status': None,
                      'id_anomaly_detection_algorithm':None,
                      'no': False, 'no2': False, 'co': False, 'ox': False, 'temperature': False, 'humidity': False}
        data = json.loads(json_values)
        print("Data:")
        print(data)
        if self.predict_anomaly_no(value_we=data['no_we'], value_aux=data['no_aux'], temp=data['temperature']):
            dictionary['no'] = True
            print("NO anomalous")
        else:
            print("NO not anomalous")
        if self.predict_anomaly_no2(value_we=data['no2_we'], value_aux=data['no2_aux'], temp=data['temperature']):
            dictionary['no2'] = True
            print("NO2 anomalous")
        else:
            print("NO2 not anomalous")
        if self.predict_anomaly_co(value_we=data['co_we'], value_aux=data['co_aux'], hum=data['humidity']):
            dictionary['co'] = True
            print("CO anomalous")
        else:
            print("CO not anomalous")
        if self.predict_anomaly_ox(value_we=data['ox_we'], value_aux=data['ox_aux'], hum=data['humidity']):
            dictionary['ox'] = True
            print("CO anomalous")
        else:
            print("CO not anomalous")
        if self.predict_anomaly_temp_hum(temp=data['temperature'], hum=data['humidity']):
            print("Temperature and Humidity are anomalous")
            dictionary['temperature'] = True
            dictionary['humidity'] = True
        dictionary['id_anomaly_detection_algorithm'] = data['algorithm']
        dictionary['phenomenon_time_sensor_raw_observation'] = data['phenomenon_time']
        dictionary['id_sensor_low_cost_status'] = data['id_sensor_low_cost_status']

        return dictionary

