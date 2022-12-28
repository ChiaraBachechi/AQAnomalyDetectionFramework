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



def vote(FFIDCAD_obj, sw_obj1, sw_obj2, batch_env, batch_obj1, batch_obj2, we, aux, enviroment, time):
    counter = 0
    if sw_obj1.check_anomaly_rt(we, time) or sw_obj2.check_anomaly_rt(aux, time):
       counter = counter + 1
    if FFIDCAD_obj.predict_anomaly(we, aux):
       counter = counter + 1
    if not batch_env.predict_anomaly(enviroment):
        if batch_obj1.predict_anomaly(we, enviroment) or batch_obj2.predict_anomaly(aux, enviroment):
            counter = counter +1
    elif batch_env.predict_anomaly(enviroment):
        counter = counter + 1
    print("Counter: ", counter)
    if counter >= 2:
        return True

def vote_temp_hum(FFIDCAD_obj, sw_obj1, sw_obj2, batch_obj1, batch_obj2, feature1, feature2, time):
    counter = 0
    if sw_obj1.check_anomaly_rt(feature1, time) or sw_obj2.check_anomaly_rt(feature2, time):
        counter = counter + 1
    if FFIDCAD_obj.predict_anomaly(feature1, feature2):
        counter = counter + 1
    if batch_obj1.predict_anomaly(feature1) or batch_obj2.predict_anomaly(feature2):
        counter = counter + 1
    #print("Counter: ", counter)
    if counter >= 2:
        return True


class TrainMajorityVotingSystem(TrainAnomalyDetection_interface):

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

        """ Fit FFIDCAD"""

        filter_NO = FFIDCAD( feature_1="no_we", feature_2="no_aux")
        filter_NO.fit(dataset=dataset)

        filter_NO2 = FFIDCAD(feature_1="no2_we", feature_2="no2_aux")
        filter_NO2.fit(dataset=dataset)

        filter_CO = FFIDCAD(feature_1="co_we", feature_2="co_aux")
        filter_CO.fit(dataset=dataset)

        filter_OX = FFIDCAD(feature_1="ox_we", feature_2="ox_aux")
        filter_OX.fit(dataset=dataset)

        filter_TEMP_HUM = FFIDCAD(feature_1="temperature", feature_2="humidity")
        filter_TEMP_HUM.fit(dataset=dataset)

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

        """ Sliding Window fit """

        sw_no_we_filter = CombinedFilter(
                                         time_col=self.time_column,
                                         column_to_analyze="no_we",
                                         set_bound=True,
                                         differences_upper_bound=4000,
                                         q=self.q,
                                         window=self.window)

        sw_no_aux_filter = CombinedFilter(
                                          time_col=self.time_column,
                                          column_to_analyze="no_aux",
                                          set_bound=True,
                                          differences_upper_bound=4000,
                                          q=self.q,
                                          window=self.window)

        sw_no2_we_filter = CombinedFilter(
                                          time_col=self.time_column,
                                          column_to_analyze="no2_we",
                                          set_bound=True,
                                          differences_upper_bound=4000,
                                          q=self.q,
                                          window=self.window)

        sw_no2_aux_filter = CombinedFilter(
                                           time_col=self.time_column,
                                           column_to_analyze="no2_aux",
                                           set_bound=True,
                                           differences_upper_bound=4000,
                                           q=self.q,
                                           window=self.window)

        sw_co_we_filter = CombinedFilter(
                                         time_col=self.time_column,
                                         column_to_analyze="co_we",
                                         set_bound=True,
                                         differences_upper_bound=4000,
                                         q=self.q,
                                         window=self.window)

        sw_co_aux_filter = CombinedFilter(
                                          time_col=self.time_column,
                                          column_to_analyze="co_aux",
                                          set_bound=True,
                                          differences_upper_bound=4000,
                                          q=self.q,
                                          window=self.window)

        sw_ox_we_filter = CombinedFilter(
                                         time_col=self.time_column,
                                         column_to_analyze="ox_we",
                                         set_bound=True,
                                         differences_upper_bound=4000,
                                         q=self.q,
                                         window=self.window)

        sw_ox_aux_filter = CombinedFilter(
                                          time_col=self.time_column,
                                          column_to_analyze="ox_aux",
                                          set_bound=True,
                                          differences_upper_bound=4000,
                                          q=self.q,
                                          window=self.window)

        sw_temperature_filter = CombinedFilter(
                                               time_col=self.time_column,
                                               column_to_analyze="temperature",
                                               set_bound=True,
                                               differences_upper_bound=4000,
                                               q=self.q,
                                               window=self.window)

        sw_humidity_filter = CombinedFilter(
                                            time_col=self.time_column,
                                            column_to_analyze="humidity",
                                            set_bound=True,
                                            differences_upper_bound=4000,
                                            q=self.q,
                                            window=self.window)

        """ Saving FFIDCAD models"""

        dill.dump(filter_NO, open('sensors/'+self.sensor+'/models/voting_filter_NO.dill', 'wb'))
        dill.dump(filter_NO2, open('sensors/' + self.sensor + '/models/voting_filter_NO2.dill', 'wb'))
        dill.dump(filter_CO, open('sensors/' + self.sensor + '/models/voting_filter_CO.dill', 'wb'))
        dill.dump(filter_OX, open('sensors/' + self.sensor + '/models/voting_filter_OX.dill', 'wb'))
        dill.dump(filter_TEMP_HUM, open('sensors/' + self.sensor + '/models/voting_filter_TH.dill', 'wb'))

        """ Saving Temp Hum based algorithm models """

        dill.dump(no_we_filter, open('sensors/'+self.sensor+'/models/voting_no_we_filter.dill', 'wb'))
        dill.dump(no_aux_filter, open('sensors/' + self.sensor + '/models/voting_no_aux_filter.dill', 'wb'))
        dill.dump(no2_we_filter, open('sensors/' + self.sensor + '/models/voting_no2_we_filter.dill', 'wb'))
        dill.dump(no2_we_filter, open('sensors/' + self.sensor + '/models/voting_no2_aux_filter.dill', 'wb'))
        dill.dump(co_we_filter, open('sensors/' + self.sensor + '/models/voting_co_we_filter.dill', 'wb'))
        dill.dump(co_aux_filter, open('sensors/' + self.sensor + '/models/voting_co_aux_filter.dill', 'wb'))
        dill.dump(ox_we_filter, open('sensors/' + self.sensor + '/models/voting_ox_we_filter.dill', 'wb'))
        dill.dump(ox_aux_filter, open('sensors/' + self.sensor + '/models/voting_ox_aux_filter.dill', 'wb'))
        dill.dump(temperature_filter, open('sensors/' + self.sensor + '/models/voting_temperature_filter.dill', 'wb'))
        dill.dump(humidity_filter, open('sensors/' + self.sensor + '/models/voting_humidity_filter.dill', 'wb'))

        """ Saving Sliding Window models"""

        dill.dump(sw_no_we_filter, open('sensors/' + self.sensor + '/models/voting_sw_no_we_filter.dill', 'wb'))
        dill.dump(sw_no_aux_filter, open('sensors/' + self.sensor + '/models/voting_sw_no_aux_filter.dill', 'wb'))
        dill.dump(sw_no2_we_filter, open('sensors/' + self.sensor + '/models/voting_sw_no2_we_filter.dill', 'wb'))
        dill.dump(sw_no2_aux_filter, open('sensors/' + self.sensor + '/models/voting_sw_no2_aux_filter.dill', 'wb'))
        dill.dump(sw_co_we_filter, open('sensors/' + self.sensor + '/models/voting_sw_co_we_filter.dill', 'wb'))
        dill.dump(sw_co_aux_filter, open('sensors/' + self.sensor + '/models/voting_sw_co_aux_filter.dill', 'wb'))
        dill.dump(sw_ox_we_filter, open('sensors/' + self.sensor + '/models/voting_sw_ox_we_filter.dill', 'wb'))
        dill.dump(sw_ox_aux_filter, open('sensors/' + self.sensor + '/models/voting_sw_ox_aux_filter.dill', 'wb'))
        dill.dump(sw_temperature_filter, open('sensors/' + self.sensor + '/models/voting_sw_temperature_filter.dill', 'wb'))
        dill.dump(sw_humidity_filter, open('sensors/' + self.sensor + '/models/voting_sw_humidity_filter.dill', 'wb'))

        """ Genero l'utlimo dump"""
        #detector = MajorityVotingSystem(sensor=self.sensor)
        #dill.dump(detector, open('dills/detector_' + self.sensor + '.dill', 'wb'))


class MajorityVotingSystem(AnomalyDetectionSystem_interface):

    def __init__(self, sensor):
        self.sensor = sensor
        self.filter_NO = dill.load(open('sensors/'+self.sensor+'/models/voting_filter_NO.dill', 'rb'))
        self.filter_NO2 = dill.load(open('sensors/' + self.sensor + '/models/voting_filter_NO2.dill', 'rb'))
        self.filter_CO = dill.load(open('sensors/' + self.sensor + '/models/voting_filter_CO.dill', 'rb'))
        self.filter_OX = dill.load(open('sensors/' + self.sensor + '/models/voting_filter_OX.dill', 'rb'))
        self.filter_TH = dill.load(open('sensors/' + self.sensor + '/models/voting_filter_TH.dill', 'rb'))
        self.no_we_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_no_we_filter.dill', 'rb'))
        self.no_aux_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_no_aux_filter.dill', 'rb'))
        self.no2_we_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_no2_we_filter.dill', 'rb'))
        self.no2_aux_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_no2_aux_filter.dill', 'rb'))
        self.co_we_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_co_we_filter.dill', 'rb'))
        self.co_aux_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_co_aux_filter.dill', 'rb'))
        self.ox_we_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_ox_we_filter.dill', 'rb'))
        self.ox_aux_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_ox_aux_filter.dill', 'rb'))
        self.temperature_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_temperature_filter.dill', 'rb'))
        self.humidity_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_humidity_filter.dill', 'rb'))
        self.sw_no_we_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_sw_no_we_filter.dill', 'rb'))
        self.sw_no_aux_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_sw_no_aux_filter.dill', 'rb'))
        self.sw_no2_we_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_sw_no2_we_filter.dill', 'rb'))
        self.sw_no2_aux_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_sw_no2_aux_filter.dill', 'rb'))
        self.sw_co_we_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_sw_co_we_filter.dill', 'rb'))
        self.sw_co_aux_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_sw_co_aux_filter.dill', 'rb'))
        self.sw_ox_we_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_sw_ox_we_filter.dill', 'rb'))
        self.sw_ox_aux_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_sw_ox_aux_filter.dill', 'rb'))
        self.sw_temperature_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_sw_temperature_filter.dill', 'rb'))
        self.sw_humidity_filter = dill.load(open('sensors/' + self.sensor + '/models/voting_sw_humidity_filter.dill', 'rb'))
        self.df_anomalies_no = None
        self.df_anomalies_no2 = None
        self.df_anomalies_co = None
        self.df_anomalies_ox = None
        self.df_anomalies_th = None
        print("Caricato correttamente")




    def apply_dataframe(self, df):
        anomalies_NO = []
        anomalies_NO2 = []
        anomalies_CO = []
        anomalies_OX = []
        anomalies_TEMP_HUM = []
        tempi_per_istanza = []
        start_assouluto = datetime.now()
        for ix, value in df.iterrows():
            start_time = datetime.now()
            value.phenomenon_time = self.sw_ox_we_filter.fix_time(value.phenomenon_time)

            """ NO """
            if vote(FFIDCAD_obj=self.filter_NO,
                    sw_obj1=self.sw_no_we_filter,
                    sw_obj2=self.sw_no_aux_filter,
                    batch_env=self.temperature_filter,
                    batch_obj1=self.no_we_filter,
                    batch_obj2=self.no_aux_filter,
                    we=value.no_we,
                    aux=value.no_aux,
                    enviroment=value.temperature,
                    time=value.phenomenon_time):
                anomalies_NO.append(ix)

            """ NO2 """

            if vote(FFIDCAD_obj=self.filter_NO2,
                    sw_obj1=self.sw_no2_we_filter,
                    sw_obj2=self.sw_no2_aux_filter,
                    batch_env=self.temperature_filter,
                    batch_obj1=self.no2_we_filter,
                    batch_obj2=self.no2_aux_filter,
                    we=value.no2_we,
                    aux=value.no2_aux,
                    enviroment=value.temperature,
                    time=value.phenomenon_time):
                anomalies_NO2.append(ix)

            """ CO """

            if vote(FFIDCAD_obj=self.filter_CO,
                    sw_obj1=self.sw_co_we_filter,
                    sw_obj2=self.sw_co_aux_filter,
                    batch_env=self.humidity_filter,
                    batch_obj1=self.co_we_filter,
                    batch_obj2=self.co_aux_filter,
                    we=value.co_we,
                    aux=value.co_aux,
                    enviroment=value.humidity,
                    time=value.phenomenon_time):
                anomalies_CO.append(ix)

            """ OX """

            if vote(FFIDCAD_obj=self.filter_OX,
                    sw_obj1=self.sw_ox_we_filter,
                    sw_obj2=self.sw_ox_aux_filter,
                    batch_env=self.humidity_filter,
                    batch_obj1=self.ox_we_filter,
                    batch_obj2=self.ox_aux_filter,
                    we=value.ox_we,
                    aux=value.ox_aux,
                    enviroment=value.humidity,
                    time=value.phenomenon_time):
                anomalies_OX.append(ix)

            """ TEMP HUM"""

            if vote_temp_hum(FFIDCAD_obj=self.filter_TH,
                             sw_obj1=self.sw_temperature_filter,
                             sw_obj2=self.sw_humidity_filter,
                             batch_obj1=self.temperature_filter,
                             batch_obj2=self.humidity_filter,
                             feature1=value.temperature,
                             feature2=value.humidity,
                             time=value.phenomenon_time):
                anomalies_TEMP_HUM.append(ix)

            tempi_per_istanza.append(datetime.now() - start_time)

        print("Tempo medio per istanza: ", np.array(tempi_per_istanza).mean())
        print("Tempo totale: ", datetime.now() - start_assouluto)
        self.df_anomalies_no = df[df.index.isin(anomalies_NO)]
        self.df_anomalies_no2 = df[df.index.isin(anomalies_NO2)]
        self.df_anomalies_co = df[df.index.isin(anomalies_CO)]
        self.df_anomalies_ox = df[df.index.isin(anomalies_OX)]
        self.df_anomalies_th = df[df.index.isin(anomalies_TEMP_HUM)]
        self.df_anomalies_no.to_csv('sensors/' + self.sensor + '/anomalies_NO')
        self.df_anomalies_no2.to_csv('sensors/' + self.sensor + '/anomalies_NO2')
        self.df_anomalies_co.to_csv('sensors/' + self.sensor + '/anomalies_CO')
        self.df_anomalies_ox.to_csv('sensors/' + self.sensor + '/anomalies_OX')
        self.df_anomalies_th.to_csv('sensors/' + self.sensor + '/anomalies_TH')
        return anomalies_NO, anomalies_NO2, anomalies_CO, anomalies_OX, anomalies_TEMP_HUM

    def predict_anomaly_no(self, phen_time, value_we, value_aux, temp):
        phenomenon_time = self.sw_no_we_filter.fix_time(phen_time)
        if vote(FFIDCAD_obj=self.filter_NO,
                sw_obj1=self.sw_no_we_filter,
                sw_obj2=self.sw_no_aux_filter,
                batch_env=self.temperature_filter,
                batch_obj1=self.no_we_filter,
                batch_obj2=self.no_aux_filter,
                we=value_we,
                aux=value_aux,
                enviroment=temp,
                time=phenomenon_time):
            return True

    def predict_anomaly_no2(self, phen_time, value_we, value_aux, temp):
        phenomenon_time = self.sw_no2_we_filter.fix_time(phen_time)
        if vote(FFIDCAD_obj=self.filter_NO2,
                sw_obj1=self.sw_no2_we_filter,
                sw_obj2=self.sw_no2_aux_filter,
                batch_env=self.temperature_filter,
                batch_obj1=self.no2_we_filter,
                batch_obj2=self.no2_aux_filter,
                we=value_we,
                aux=value_aux,
                enviroment=temp,
                time=phenomenon_time):
            return True

    def predict_anomaly_co(self, phen_time, value_we, value_aux, hum):
        phenomenon_time = self.sw_co_we_filter.fix_time(phen_time)
        if vote(FFIDCAD_obj=self.filter_CO,
                sw_obj1=self.sw_co_we_filter,
                sw_obj2=self.sw_co_aux_filter,
                batch_env=self.humidity_filter,
                batch_obj1=self.co_we_filter,
                batch_obj2=self.co_aux_filter,
                we=value_we,
                aux=value_aux,
                enviroment=hum,
                time=phenomenon_time):
            return True

    def predict_anomaly_ox(self, phen_time, value_we, value_aux, hum):
        phenomenon_time = self.sw_ox_we_filter.fix_time(phen_time)
        if vote(FFIDCAD_obj=self.filter_OX,
                sw_obj1=self.sw_ox_we_filter,
                sw_obj2=self.sw_ox_aux_filter,
                batch_env=self.humidity_filter,
                batch_obj1=self.ox_we_filter,
                batch_obj2=self.ox_aux_filter,
                we=value_we,
                aux=value_aux,
                enviroment=hum,
                time=phenomenon_time):
            return True

    def predict_anomaly_temp_hum(self, temp, hum, phen_time):
        phenomenon_time = self.sw_ox_we_filter.fix_time(phen_time)
        if vote_temp_hum(FFIDCAD_obj=self.filter_TH,
                         sw_obj1=self.sw_temperature_filter,
                         sw_obj2=self.sw_humidity_filter,
                         batch_obj1=self.temperature_filter,
                         batch_obj2=self.humidity_filter,
                         feature1=temp,
                         feature2=hum,
                         time=phenomenon_time):
            return True

    def apply_single_row(self, json_values):
        dictionary = {'phenomenon_time_sensor_raw_observation': None,
                      'id_sensor_low_cost_status': None,
                      'id_anomaly_detection_algorithm': None,
                      'no': False, 'no2': False, 'co': False, 'ox': False, 'temperature': False, 'humidity': False}
        data=json.loads(json_values)
        print("Data:")
        print(data)
        phenomenon_time = self.sw_no_we_filter.fix_time(data['phenomenon_time'])
        if vote(FFIDCAD_obj=self.filter_NO,
                sw_obj1=self.sw_no_we_filter,
                sw_obj2=self.sw_no_aux_filter,
                batch_env=self.temperature_filter,
                batch_obj1=self.no_we_filter,
                batch_obj2=self.no_aux_filter,
                we=data['no_we'],
                aux=data['no_aux'],
                enviroment=data['temperature'],
                time=phenomenon_time):
            dictionary['NO'] = True
            print("NO anomalous")
        else:
            print("NO not anomalous")

        if vote(FFIDCAD_obj=self.filter_NO2,
                sw_obj1=self.sw_no2_we_filter,
                sw_obj2=self.sw_no2_aux_filter,
                batch_env=self.temperature_filter,
                batch_obj1=self.no2_we_filter,
                batch_obj2=self.no2_aux_filter,
                we=data['no2_we'],
                aux=data['no2_aux'],
                enviroment=data['temperature'],
                time=phenomenon_time):
            dictionary['NO2'] = True
            print("NO2 anomalous")
        else:
            print("NO2 not anomalous")

        if vote(FFIDCAD_obj=self.filter_CO,
                sw_obj1=self.sw_co_we_filter,
                sw_obj2=self.sw_co_aux_filter,
                batch_env=self.humidity_filter,
                batch_obj1=self.co_we_filter,
                batch_obj2=self.co_aux_filter,
                we=data['co_we'],
                aux=data['co_aux'],
                enviroment=data['humidity'],
                time=phenomenon_time):
            dictionary['CO'] = True
            print("CO anomalous")
        else:
            print("CO not anomalous")

        if vote(FFIDCAD_obj=self.filter_OX,
                sw_obj1=self.sw_ox_we_filter,
                sw_obj2=self.sw_ox_aux_filter,
                batch_env=self.humidity_filter,
                batch_obj1=self.ox_we_filter,
                batch_obj2=self.ox_aux_filter,
                we=data['ox_we'],
                aux=data['ox_aux'],
                enviroment=data['humidity'],
                time=phenomenon_time):
            dictionary['OX'] = True
            print("OX anomalous")
        else:
            print("OX not anomalous")

        if vote_temp_hum(FFIDCAD_obj=self.filter_TH,
                         sw_obj1=self.sw_temperature_filter,
                         sw_obj2=self.sw_humidity_filter,
                         batch_obj1=self.temperature_filter,
                         batch_obj2=self.humidity_filter,
                         feature1=data['temperature'],
                         feature2=data['humidity'],
                         time=phenomenon_time):
            dictionary['Temperature'] = True
            dictionary['Humidity'] = True
            print("TH anomalous")
        else:
            print("TH not anomalous")
        print(dictionary)
        dictionary['id_anomaly_detection_algorithm'] = data['algorithm']
        dictionary['phenomenon_time_sensor_raw_observation'] = data['phenomenon_time']
        dictionary['id_sensor_low_cost_status'] = data['id_sensor_low_cost_status']
        return dictionary