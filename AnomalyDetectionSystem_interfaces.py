from interface import *

""" generate db -> all'esterno"""
""" togliere i singoli predict """


class TrainAnomalyDetection_interface(metaclass=Interface):

    @abstractfunc
    def __init__(self, sensor):
        """
        info_dictionary is a "json" dictionary
        """

    @abstractfunc
    def fit(self, dataset):
        """ Method to train the detectors and saving dill files"""


class AnomalyDetectionSystem_interface(metaclass=Interface):

    @abstractfunc
    def __init__(self,sensor):
        """ Method that load dill files """
        self.sensor=sensor
        
    @abstractfunc
    def apply_dataframe(self, df):
        """
        df is the dataframe in which anomalies are wanted to be found
        """

    def apply_single_row(self, row):
        """ Call the methods to predict anomaly for the single polluttant"""