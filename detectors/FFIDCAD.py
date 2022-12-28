import pandas as pd
from FFIDCAD.FFIDCAD import FFIDCAD
from datetime import datetime
import numpy as np
import dill
import json
import seaborn as sns #da installare sul server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from AnomalyDetectionSystem_interfaces import *


class TrainFFIDCAD(TrainAnomalyDetection_interface):

    def __init__(self, info_dictionary, sensor):
        self.sensor = sensor
        with open(info_dictionary, 'r') as prop_file:
            detector_properties = json.load(prop_file)
            self.esponente = detector_properties['esponente']
        print("Trainer initialized correctly")

    def fit(self, dataset):

        filter_NO = FFIDCAD(feature_1="no_we", feature_2="no_aux", esponente=self.esponente)
        filter_NO.fit(dataset=dataset)

        filter_NO2 = FFIDCAD(feature_1="no2_we", feature_2="no2_aux", esponente=self.esponente)
        filter_NO2.fit(dataset=dataset)

        filter_CO = FFIDCAD(feature_1="co_we", feature_2="co_aux", esponente=self.esponente)
        filter_CO.fit(dataset=dataset)

        filter_OX = FFIDCAD(feature_1="ox_we", feature_2="ox_aux", esponente=self.esponente)
        filter_OX.fit(dataset=dataset)

        filter_TEMP_HUM = FFIDCAD(feature_1="temperature", feature_2="humidity", esponente=self.esponente)
        filter_TEMP_HUM.fit(dataset=dataset)

        dill.dump(filter_NO, open('sensors/' + self.sensor + '/models/filter_NO.dill', 'wb'))
        dill.dump(filter_NO2, open('sensors/' + self.sensor + '/models/filter_NO2.dill', 'wb'))
        dill.dump(filter_CO, open('sensors/' + self.sensor + '/models/filter_CO.dill', 'wb'))
        dill.dump(filter_OX, open('sensors/' + self.sensor + '/models/filter_OX.dill', 'wb'))
        dill.dump(filter_TEMP_HUM, open('sensors/' + self.sensor + '/models/filter_TH.dill', 'wb'))


class FFDICAD(AnomalyDetectionSystem_interface):

        def __init__(self, sensor):
            self.sensor = sensor
            self.filter_NO = dill.load(open('sensors/' + self.sensor + '/models/filter_NO.dill', 'rb'))
            self.filter_NO2 = dill.load(open('sensors/' + self.sensor + '/models/filter_NO2.dill', 'rb'))
            self.filter_CO = dill.load(open('sensors/' + self.sensor + '/models/filter_CO.dill', 'rb'))
            self.filter_OX = dill.load(open('sensors/' + self.sensor + '/models/filter_OX.dill', 'rb'))
            self.filter_TH = dill.load(open('sensors/' + self.sensor + '/models/filter_TH.dill', 'rb'))
            print("Caricato correttamente")

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

                if self.filter_NO.predict_anomaly(value.no_we, value.no_aux):
                    anomalies_NO.append(ix)

                if self.filter_NO2.predict_anomaly(value.no2_we, value.no2_aux):
                    anomalies_NO2.append(ix)

                if self.filter_CO.predict_anomaly(value.co_we, value.co_aux):
                    anomalies_CO.append(ix)

                if self.filter_OX.predict_anomaly(value.ox_we, value.ox_aux):
                    anomalies_OX.append(ix)

                if self.filter_TH.predict_anomaly(value.temperature, value.humidity):
                    anomalies_TEMP_HUM.append(ix)

                tempi_per_istanza.append(datetime.now() - start_time)

            print("Tempo medio per istanza: ", np.array(tempi_per_istanza).mean())
            print("Tempo totale: ", datetime.now() - start_assouluto)
            return anomalies_NO, anomalies_NO2, anomalies_CO, anomalies_OX, anomalies_TEMP_HUM

        def predict_anomaly_no(self, value_we, value_aux):
            if self.filter_NO.predict_anomaly(value_we,value_aux):
                return True

        def predict_anomaly_no2(self, value_we, value_aux):
            if self.filter_NO2.predict_anomaly(value_we,value_aux):
                return True

        def predict_anomaly_co(self, value_we, value_aux):
            if self.filter_CO.predict_anomaly(value_we,value_aux):
                return True

        def predict_anomaly_ox(self, value_we, value_aux):
            if self.filter_OX.predict_anomaly(value_we,value_aux):
                return True

        def predict_anomaly_temp_hum(self, temp, hum):
            if self.filter_TH.predict_anomaly(temp, hum):
                return True

        def apply_single_row(self, json_values):
            dictionary = {'phenomenon_time_sensor_raw_observation': None,
                          'id_sensor_low_cost_status': None,
                          'id_anomaly_detection_algorithm': None,
                          'no': False, 'no2': False, 'co': False, 'ox': False, 'temperature': False, 'humidity': False}
            data = json.loads(json_values)
            print("Data:")
            print(data)
            if self.predict_anomaly_no(value_we=data['no_we'], value_aux=data['no_aux']):
                dictionary['NO'] = True
                print("NO anomalous")
            else:
                print("NO not anomalous")
            if self.predict_anomaly_no2(value_we=data['no2_we'], value_aux=data['no2_aux']):
                dictionary['NO2'] = True
                print("NO2 anomalous")
            else:
                print("NO2 not anomalous")
            if self.predict_anomaly_co(value_we=data['co_we'], value_aux=data['co_aux']):
                dictionary['CO'] = True
                print("CO anomalous")
            else:
                print("CO not anomalous")
            if self.predict_anomaly_ox(value_we=data['ox_we'], value_aux=data['ox_aux']):
                dictionary['OX'] = True
                print("CO anomalous")
            else:
                print("CO not anomalous")
            if self.predict_anomaly_temp_hum(temp=data['temperature'], hum=data['humidity']):
                print("Temperature and Humidity are anomalous")
                dictionary['Temperature'] = True
                dictionary['Humidity'] = True
            print(dictionary)
            dictionary['id_anomaly_detection_algorithm'] = data['algorithm']
            dictionary['phenomenon_time_sensor_raw_observation'] = data['phenomenon_time']
            dictionary['id_sensor_low_cost_status'] = data['id_sensor_low_cost_status']
            return dictionary




