import argparse
import ast
import numpy as np
import pandas as pd
import pandas.io.sql as sqlio
import psycopg2
import sys
import importlib
import dill
import json
import time
import os
# ---

#from AnomalyDetectionSystem import TrainAnomalyDetectionSystem
#from AnomalyDetectionSystem import AnomalyDetectionSystem
#import AnomalyDetectionSystem_interfaces
from AnomalyDetectionFramework import AnomalyDetectionFramework

sys.path.insert(0, '../..')


def usage():
    usageString = """
 usage
   python3 main.py [action] [options]

   available actions:

   InsertAlgorithmFromCSV: copy id and configuration_json from a csv file in table anomaly_detection algorithm
   TrainAnomalyDetectionSystem: initializes the parameters of the of an algorithm and generates the dills. The dill
                                has the name "detector_<id_sensor>_<id_algorithm>.dill"
   AnomalyDetectionHistoricalWriteCSV: extracts raw data from the database and analyzes them. The output is a CSV file
                                       containing the anomalies.
   AnomalyDetectionRealTImeTest: Analyzes a json string containing data of the pollutants. Returns a dictionary 
                                 containing a boolean value for each pollutant, which is True in case of anomaly.
   AnomalyDetectionHistoricalWriteDB: Analyzes a json string containing data of the pollutants. Returns a dictionary 
                                 containing a boolean value for each pollutant, which is True in case of anomaly. If
                                 at least one pollutant is True, then insert the dictionary in table 
                                 sensor_raw_observation_anomaly
   AnomalyDetectionRealTImeWriteDB: Analyzes a json string containing data of the pollutants. Returns a dictionary 
                                 containing a boolean value for each pollutant, which is True in case of anomaly. If
                                 the dictionary contains at least one True value, then the row gets inserted in table
                                 sensor_raw_observation_anomaly
   InsertCSVtoDB: copies data from a csv containing the anomalies in table sensor_raw_observation_anomaly
   AnomalyDetectionCalibratedDataWriteCSV: extracts calibrated data and analyzes them. The output is a csv file.
   AnomalyDetectionCalibratedDataWriteDB: extracts calibrated data and analyzes them. Anomalies contained in csv file
                                          are inserted into the database.
   InsertCalibratedCSVtoDB: copies data from a csv containing the anomalies of calibrated data into the database
   CompareRawCalibratedAnomalies: performs an analysis of anomaly propagation: check if there is a daily corrispondence
                                  between raw anomalies and calibrated anomalies.


   available options: 

   action: the action to perform. 
   id_sensor: the id of the sensor to be considered
   id_algorithm: the id of the algorithm to use, it can be chosen in table anomaly_detection_algorithm
   begin_time: start of the period to consider in historical analysis
   end_time: end of the period to consider in historical analysis
   json_values: string json containing values to analyze in real time
   plot: a boolean value. If True, graphical representation of anomalies will be generated at the end of a historical
         analysis.
   json_parameters: a file containing the parameters to use for the initialization of an algorithm.
   algorithm_file_name: name of the csv file that contains id and configuration json of an algorithm.

 usage: 

   python3 main.py --action InsertAlgorithmFromCSV [algorithm_file_name]
   python3 main.py --action TrainAnomalyDetectionSystem [id_algorithm][id_sensor][json_parameters][begin_time][end_time]
   python3 main.py --action AnomalyDetectionHistoricalWriteCSV [id_algorithm][id_sensor][begin_time][end_time][plot]
   python3 main.py --action AnomalyDetectionHistoricalWriteDB [id_algorithm][id_sensor][begin_time][end_time][plot]
   python3 main.py --action AnomalyDetectionRealTImeTest [id_algorithm][id_sensor][json_values]
   python3 main.py --action AnomalyDetectionRealTImeWriteDB [id_algorithm][id_sensor][json_values]
   python3 main.py --action InsertCSVtoDB [id_sensor]
   python3 main.py --action AnomalyDetectionCalibratedDataWriteCSV [id_algorithm][id_sensor][begin_time][end_time]
   python3 main.py --action AnomalyDetectionCalibratedDataWriteDB [id_algorithm][id_sensor][begin_time][end_time]
   python3 main.py --action InsertCalibratedCSVtoDB [id_sensor]
   python3 main.py --action CompareRawCalibratedAnomalies [id_sensor]

 examples:
   in the code below it is possible to find some example for almost all the actions

"""
    print(usageString)

def addOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', dest='action', type=str, help='The framework action to perform.', default="")
    parser.add_argument('--id_sensor', '-s', dest='id_sensor', type=str,
                        help='Id of sensor which has to be considered', default="all")
    parser.add_argument('--id_algorithm', dest='id_algorithm', type=str,
                        help='The id of the algorithm in the database')
    parser.add_argument('--begin_time', '-b', dest='begin_time', type=str,
                        help='Insert the date and time to start the calibration from. Formatted as YYYY-MM-DD HH:MM:SS')
    parser.add_argument('--end_time', '-e', dest='end_time', type=str,
                        help='Insert the date and time to end the calibration. Formatted as YYYY-MM-DD HH:MM:SS')
    parser.add_argument('--json_values',  dest='json_values', type=str,
                        help='Couples key values for the detection in real time')
    parser.add_argument('--plot', dest='plot', type=bool, default=False,
                        help='set True to plot data after historical analysis')
    parser.add_argument('--json_parameters', dest='jsonParameters', type=str,
                        help='File that contains a set of parameters for detector')
    parser.add_argument('--algorithm_file_name', dest='algorithmFileName', type=str,
                        help='contains id and description of the algorithm to insert in the db')
    return parser


def optionsToInfo(options):
    status = {}
    status['id_sensor']=options.id_sensor
    status['id_algorithm'] = op.id_algorithm
    if options.algorithm_parameters == "":
        status['json_parameters'] = {}
    else:
        status['json_parameters'] = json.load(options.json_parameters)
    print(" --- optionsToInfo:\n", json.dumps(status, sort_keys=True, indent=2))
    return status


def main(args=None):
    argParser = addOptions()
    options = argParser.parse_args(args=args)
    iterative_algorithms = ['1','3','6', '7'] # Id degli algoiritmi iterativi
    algorithms_for_calibrated_data = ['7'] # Id degli algoritmi che lavorano con dati calibrati
    #all_sensors_available = ['4003', '4004','4005','4006','4007','4008','4009','4010','4011','4012','4013','4014']
    #if options.id_sensor not in all_sensors_available:
     # raise ValueError(options.id_sensor + ' is not a valid sensor. Choose between 4003 and 4014')
    action = options.action

    if action == 'InsertAlgorithmFromCSV':
        if options.algorithmFileName == None:
            raise ValueError("Insert the name of the file containing algorithm to insert into"
                             "'anomaly_detection_algorithm table'")
        framework = AnomalyDetectionFramework()
        framework.insertAnomalyDetectionAlgorithmFromCSV(options.algorithmFileName)

    if action == 'TrainAnomalyDetectionSystem':
        """ example execution server: 
                    python3 main.py --id_sensor 4007 
                    --id_algorithm 6 
                    --json_parameters parametersVoting001.JSON 
                    --action TrainAnomalyDetectionSystem
                    --begin_time "2019-08-01 00:00:00" 
                    --end_time "2020-08-01 00:00:00"
        """
        if options.id_sensor == None:
            raise ValueError("Insert the id of the considered sensor")
        if options.id_algorithm == None:
            raise ValueError("Insert an algorithm to use. Available algorithms can be chosen from"
                             "the table 'anomaly_detection_algorithm'")
        if options.begin_time == None:
            raise ValueError("Insert the begin time of the analysis")
        if options.end_time == None:
            raise ValueError("Insert the end time of the analysis")
        if options.jsonParameters == None:
            raise ValueError("Insert the name of the file in directory json_parameters containing the parameters"
                             "to initialize the algorithm")
        framework = AnomalyDetectionFramework()
        dataset = framework.getDataFromDB(options.id_sensor, options.begin_time, options.end_time)
        if options.id_algorithm in algorithms_for_calibrated_data:
            dataset = framework.getCalibratedData(options.id_sensor, options.begin_time, options.end_time)
        else:
            #dataset = pd.read_csv('training_datasets/'+options.id_sensor+'_training')
            dataset.sort_values('phenomenon_time', inplace=True)
        dataset.to_csv('tmp')
        dataset_train = pd.read_csv('tmp')
        print(len(dataset))
        print(dataset.head())
        print(dataset.tail())
        detector = framework.getDetector(options.id_algorithm, options.id_sensor, dataset_train,
                                         'json_parameters/'+options.jsonParameters)
        dill.dump(detector, open('dills/detector_' + options.id_sensor + '_' + options.id_algorithm + '.dill', 'wb'))
        if options.id_algorithm in iterative_algorithms:
            dill.dump(detector, open('dills/rt_detector_' + options.id_sensor + '_' + options.id_algorithm + '.dill', 'wb'))

    if action == 'AnomalyDetectionHistoricalWriteCSV':
        """ example execution server: 
            python3 main.py --id_sensor 4007
            --id_algorithm 6
            --action AnomalyDetectionHistoricalWriteCSV
            --begin_time "2019-08-01 00:00:00" 
            --end_time "2020-08-01 00:00:00"
            --plot True
        """
        if options.id_sensor == None:
            raise ValueError("Insert the id of the considered sensor")
        if options.id_algorithm == None:
            raise ValueError("Insert an algorithm to use. Available algorithms can be chosen from"
                             "the table 'anomaly_detection_algorithm'")
        if options.begin_time == None:
            raise ValueError("Insert the begin time of the analysis")
        if options.end_time == None:
            raise ValueError("Insert the end time of the analysis")
        framework = AnomalyDetectionFramework()
        dataset = framework.getDataFromDB(options.id_sensor, options.begin_time, options.end_time)
        #dataset=dataset[['phenomenon_time','no_we','no_aux','no2_we','no2_aux','co_we','co_aux', 'ox_we','ox_aux']]
        dataset.sort_values('phenomenon_time', inplace=True)
        dataset.reset_index(inplace=True)
        dataset.to_csv('tmp')
        ad = dill.load(open('dills/detector_' + options.id_sensor + '_' + options.id_algorithm + '.dill', 'rb'))
        no, no2, co, ox, th = ad.apply_dataframe(dataset)
        tmp=pd.read_csv('tmp')
        framework.generate_csv_sensor_raw_anomalies(tmp, options.id_sensor, options.id_algorithm, no, no2, co, ox, th)
        if options.plot:
            framework.plot(df=tmp, sensor=options.id_sensor, anomalies_array=no, feature_1='no_we', feature_2='no_aux', feature_name='NO')
            framework.plot(df=tmp, sensor=options.id_sensor, anomalies_array=no2, feature_1='no2_we', feature_2='no2_aux', feature_name='NO2')
            framework.plot(df=tmp, sensor=options.id_sensor, anomalies_array=co, feature_1='co_we', feature_2='co_aux', feature_name='CO')
            framework.plot(df=tmp, sensor=options.id_sensor, anomalies_array=ox, feature_1='ox_we', feature_2='ox_aux', feature_name='OX')
            framework.plot(df=tmp, sensor=options.id_sensor, anomalies_array=th, feature_1='temperature', feature_2='humidity', feature_name='TEMPERATURE-HUMIDITY')

    if action == 'AnomalyDetectionRealTImeTest':
        """ Example in 'esempio stringa real time"""

        if options.id_algorithm in iterative_algorithms:
            ad = dill.load(open('dills/rt_detector_' + options.id_sensor + '_' + options.id_algorithm + '.dill', 'rb'))
        else:
            ad = dill.load(open('dills/detector_' + options.id_sensor + '_' + options.id_algorithm + '.dill', 'rb'))
        print("dill caricato")
        dictionary = ad.apply_single_row(options.json_values)
        print(dictionary)
        if options.id_algorithm in iterative_algorithms:
            dill.dump(ad, open('dills/rt_detector_' + options.id_sensor + '_' + options.id_algorithm + '.dill', 'wb'))
            print("Updating dill...")

    if action == 'AnomalyDetectionHistoricalWriteDB':
        """ example execution server: 
            python3 main.py --id_sensor 4007
            --id_algorithm 6
            --action AnomalyDetectionRealTimeWriteDB
            --begin_time "2019-08-01 00:00:00" 
            --end_time "2020-08-01 00:00:00"
            
        """
        if options.id_sensor == None:
            raise ValueError("Insert the id of the considered sensor")
        if options.id_algorithm == None:
            raise ValueError("Insert an algorithm to use. Available algorithms can be chosen from"
                             "the table 'anomaly_detection_algorithm'")
        if options.begin_time == None:
            raise ValueError("Insert the begin time of the analysis")
        if options.end_time == None:
            raise ValueError("Insert the end time of the analysis")
        framework = AnomalyDetectionFramework()
        dataset = framework.getDataFromDB(options.id_sensor, options.begin_time, options.end_time)
        dataset.sort_values('phenomenon_time', inplace=True)
        dataset.reset_index(inplace=True)
        dataset.to_csv('tmp')
        ad = dill.load(open('dills/detector_' + options.id_sensor + '_' + options.id_algorithm + '.dill', 'rb'))
        no, no2, co, ox, th = ad.apply_dataframe(dataset)
        tmp = pd.read_csv('tmp')
        framework.generate_csv_sensor_raw_anomalies(tmp, options.id_sensor, options.id_algorithm, no, no2, co, ox, th)
        """ Inserire query per caricare il dataset """
        framework.insertAnomalyIntoDBfromCSV(options.id_sensor)
        print("Data inserted")
        if options.plot:
            framework.plot(df=tmp, sensor=options.id_sensor, anomalies_array=no, feature_1='no_we', feature_2='no_aux', feature_name='NO')
            framework.plot(df=tmp, sensor=options.id_sensor, anomalies_array=no2, feature_1='no2_we', feature_2='no2_aux', feature_name='NO2')
            framework.plot(df=tmp, sensor=options.id_sensor, anomalies_array=co, feature_1='co_we', feature_2='co_aux', feature_name='CO')
            framework.plot(df=tmp, sensor=options.id_sensor, anomalies_array=ox, feature_1='ox_we', feature_2='ox_aux', feature_name='OX')
            framework.plot(df=tmp, sensor=options.id_sensor, anomalies_array=th, feature_1='temperature', feature_2='humidity', feature_name='TEMPERATURE-HUMIDITY')

    if action == 'AnomalyDetectionRealTimeWriteDB':
        """ Example in 'esempio stringa real time' """
        if options.id_algorithm in iterative_algorithms:
            ad = dill.load(open('dills/rt_detector_' + options.id_sensor + '_' + options.id_algorithm + '.dill', 'rb'))
        else:
            ad = dill.load(open('dills/detector_' + options.id_sensor + '_' + options.id_algorithm + '.dill', 'rb'))
        dictionary = ad.apply_single_row(options.json_values)
        print(dictionary)
        framework = AnomalyDetectionFramework()
        framework.insertAnomalySingleRow(dictionary)
        print("Data inserted")
        if options.id_algorithm in iterative_algorithms:
            dill.dump(ad, open('dills/rt_detector_' + options.id_sensor + '_' + options.id_algorithm + '.dill', 'wb'))
            print("Updating dill...")

    if action == 'InsertCSVtoDB':
        if options.id_sensor == None:
            raise ValueError("Insert the id of the considered sensor")
        framework = AnomalyDetectionFramework()
        framework.insertAnomalyIntoDBfromCSV(options.id_sensor)
        print("Data inserted")

    if action=='AnomalyDetectionCalibratedDataWriteCSV':
        """ example
        python3 main.py --id_sensor 4010 --action AnomalyDetectionCalibratedDataWriteCSV
         --begin_time "2019-08-01 00:00:00" --end_time "2020-09-15 00:00:00" --id_algorithm 7  """
        if options.id_sensor == None:
            raise ValueError("Insert the id of the considered sensor")
        if options.id_algorithm == None:
            raise ValueError("Insert an algorithm to use. Available algorithms can be chosen from"
                             "the table 'anomaly_detection_algorithm'")
        if options.begin_time == None:
            raise ValueError("Insert the begin time of the analysis")
        if options.end_time == None:
            raise ValueError("Insert the end time of the analysis")
        ad = dill.load(open('dills/detector_' + options.id_sensor + '_' + options.id_algorithm + '.dill', 'rb'))
        framework = AnomalyDetectionFramework()
        dataset = framework.getCalibratedData(options.id_sensor, options.begin_time, options.end_time)
        dataset.reset_index(inplace=True)
        print(dataset)
        dataset.to_csv('tmp')

        no, no2, co, o3 = ad.apply_dataframe(dataset)
        tmp = pd.read_csv('tmp')
        framework.generate_csv_sensor_observation_anomalies(tmp, options.id_sensor,
                                                            options.id_algorithm, no, no2, co, o3)

    if action == 'AnomalyDetectionCalibratedDataWriteDB':
        """ example
        python3 main.py --id_sensor 4010 --action AnomalyDetectionCalibratedDataWriteDB
         --begin_time "2019-08-01 00:00:00" --end_time "2020-09-15 00:00:00" --id_algorithm 7  """
        if options.id_sensor == None:
            raise ValueError("Insert the id of the considered sensor")
        if options.id_algorithm == None:
            raise ValueError("Insert an algorithm to use. Available algorithms can be chosen from"
                             "the table 'anomaly_detection_algorithm'")
        if options.begin_time == None:
            raise ValueError("Insert the begin time of the analysis")
        if options.end_time == None:
            raise ValueError("Insert the end time of the analysis")
        ad = dill.load(open('dills/detector_' + options.id_sensor + '_' + options.id_algorithm + '.dill', 'rb'))
        framework = AnomalyDetectionFramework()
        dataset = framework.getCalibratedData(options.id_sensor, options.begin_time, options.end_time)
        dataset.reset_index(inplace=True)
        print(dataset)
        dataset.to_csv('tmp')

        no, no2, co, o3 = ad.apply_dataframe(dataset)
        tmp = pd.read_csv('tmp')
        framework.generate_csv_sensor_observation_anomalies(tmp, options.id_sensor,
                                                            options.id_algorithm, no, no2, co, o3)
        framework.insertCalibratedAnomalyFromCSV(options.id_sensor)
        print("Calibrated Data inserted")

    if action == 'InsertCalibratedCSVtoDB':
        if options.id_sensor == None:
            raise ValueError("Insert the id of the considered sensor")
        framework = AnomalyDetectionFramework()
        framework.insertCalibratedAnomalyFromCSV(options.id_sensor)
        print("Data inserted")

    if action == 'CompareRawCalibratedAnomalies':
        if options.id_sensor == None:
            raise ValueError("Insert the id of the considered sensor")
        framework = AnomalyDetectionFramework()
        framework.compare_raw_calibrated_anomalies(options.id_sensor)

main()
