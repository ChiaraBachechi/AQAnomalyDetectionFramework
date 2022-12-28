import psycopg2
import datetime
import json
import pandas.io.sql as sqlio
from detectors.MajorityVotingSystem import *
from AnomalyDetectionSystem_interfaces import *
from detectors.TemperatureHumidityBased import *
from detectors.FFIDCAD import *
from detectors.SlidingWindowAnomalyDetection import *
from detectors.CalibratedSlidingWindowAnomalyDetection import *


class AnomalyDetectionSystem_Trainer:
    def __init__(self, id_sensor, id_algorithm, json_parameters):
        self.id_algorithm = id_algorithm
        self.id_sensor = id_sensor
        self.json_parameters = json_parameters
        print("trainer initialized")
        print(self.json_parameters)

    def train(self, dataset):
        if self.id_algorithm == '6':
            print("Majority Voting System is training...")
            #voting_system = TrainMajorityVotingSystem(info_dictionary=options.jsonParameters,
             #                                         sensor=options.id_sensor)
            """ Versione windows"""
            voting_system = TrainMajorityVotingSystem(info_dictionary=self.json_parameters,
                                                      sensor=self.id_sensor)
            voting_system.fit(dataset=dataset)
            detector = MajorityVotingSystem(sensor=self.id_sensor)
            return detector

        if self.id_algorithm == '2':
            print("Temperature and humidity based is training...")
            # voting_system = TrainMajorityVotingSystem(info_dictionary=options.jsonParameters,
            #                                         sensor=options.id_sensor)
            """ Versione windows"""
            anomaly_detector = TrainTemperatureHumidityBased(info_dictionary=self.json_parameters,
                                                             sensor=self.id_sensor)
            anomaly_detector.fit(dataset=dataset)
            detector = TemperatureHumidityBased(sensor=self.id_sensor)
            return detector

        if self.id_algorithm == '1':
            print("FFIDCAD is training...")
            anomaly_detector = TrainFFIDCAD(info_dictionary=self.json_parameters,
                                                             sensor=self.id_sensor)
            anomaly_detector.fit(dataset=dataset)
            detector = FFDICAD(sensor=self.id_sensor)
            return detector

        if self.id_algorithm == '3':
            print("Sliding Window Algorithm is training...")
            anomaly_detector = TrainSlidingWindowAlgorithm(info_dictionary=self.json_parameters, sensor=self.id_sensor)
            print("sono qui")
            anomaly_detector.fit(dataset=dataset)
            detector = SlidingWindowAlgorithm(sensor=self.id_sensor)
            return detector

        if self.id_algorithm == '7':
            print("Calibrated Sliding Window Algorithm is training...")
            anomaly_detector = TrainCalibratedSlidingWindowAlgorithm(info_dictionary=self.json_parameters, sensor=self.id_sensor)
            anomaly_detector.fit(dataset=dataset)
            detector = CalibratedSlidingWindowAlgorithm(sensor=self.id_sensor)
            return detector



