import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd

def get_z_score(array):
    z_array = []
    mean = np.array(array).mean()
    std = np.array(array).std()
    for i in range(len(array)):
        z_array.append((array[i]-mean) / std)
    return z_array


class CombinedFilter:

    def __init__(self, dataset=None, time_col=None, column_to_analyze=None, set_bound=False, differences_upper_bound=0,
                 k=3, q=2.5, window=1000):

        #self.dataset = dataset
        self.time_col = time_col
        self.column_to_analyze = column_to_analyze
        self.counter = 0
        self.anomalies = []
        self.previous_data = 0
        self.previous_timestamp = 0

        """ Attributi per il filtro sulle differenze """

        self.mean = 0
        self.std = 0
        self.differences_upper_bound = 0
        self.k = k   #1.5 * 7 / prob_a_priori
        self.max_time_difference = datetime.timedelta(minutes=5)
        self.differences = []
        self.set_bound = set_bound
        self.z_differences = []
        self.container_timestamps = []
        self.max_time_to_init = datetime.timedelta(hours=24)
        """ Attributi per la parte iqr """

        self.iqr_lower_bound = 0
        self.iqr_upper_bound = 0
        self.median = 0
        self.iqr = 0
        self.q1 = 0
        self.q3 = 0
        self.lower_than_median = []
        self.greater_than_median = []
        self.container = []
        self.all_data = []
        self.q = q # il k dell'iqr based alghoritm
        self.iqr_lb_to_plot = []
        self.iqr_ub_to_plot = []
        self.z_scores = []
        self.iqr_mean = 0
        self.iqr_std = 0
        self.window = window
        self.previous_mean = 0 # usato per correggere il bug della media = 0 negli aux
        """ Metodi iqr senza check anomaly e plot """

    def update_counter(self):
        self.counter = self.counter + 1

    def update_range(self):
        self.iqr_lower_bound = self.q1 - self.q * self.iqr
        self.iqr_upper_bound = self.q3 + self.q * self.iqr

    def init_iqr_range(self):
        self.median = np.array(self.z_scores).mean()
        for value in range(len(self.z_scores)):
            if self.z_scores[value] <= self.median:
                self.lower_than_median.append(self.z_scores[value])
            else:
                self.greater_than_median.append(self.z_scores[value])
        self.update_q1_q3()
        self.iqr = self.q3 - self.q1
        self.iqr_lower_bound = self.q1 - self.q * self.iqr
        self.iqr_upper_bound = self.q3 + self.q * self.iqr
        #print(self.iqr_lower_bound)
        #print(self.iqr_upper_bound)

    def update_q1_q3(self):
        self.q1 = np.array(self.lower_than_median).mean()
        self.q3 = np.array(self.greater_than_median).mean()

    def update_plot_data(self):
        self.iqr_lb_to_plot.append(self.iqr_lower_bound)
        self.iqr_ub_to_plot.append(self.iqr_upper_bound)

    def split(self, value):
        if value <= self.median:
            self.lower_than_median.append(value)
        else:
            self.greater_than_median.append(value)


    def fix_time(self, timestamp):
        timestamp = pd.to_datetime(timestamp, format='%Y-%m-%d %H:%M:%S')
        return timestamp

    def compute_range(self):
        self.mean = np.array(self.differences).mean()
        self.std = np.array(self.differences).std()
        self.differences_upper_bound = self.mean + self.k * self.std
        #print("Differences Mean : ", self.mean)
        #print("Differences Std : ", self.std)
        print("Differences upper bound : ", self.differences_upper_bound)

    def check_anomaly_rt(self, x, x_timestamp):
        self.lower_than_median = []
        self.greater_than_median = []
        self.update_counter()
        #print("container: ",self.container)
        #print("counter: ",self.counter)
        """ Fase di pre-training """

        if self.counter < self.window:
            if self.counter == 1:
                diff = 0
            else:
                diff = abs(x - self.previous_data)
            self.differences.append(diff)
            self.previous_data = x
            self.previous_timestamp = x_timestamp
            self.container.append(x)
            self.container_timestamps.append(x_timestamp)
            if self.counter < 3:
                print(self.container)


        elif self.counter == self.window:
            self.previous_data = x
            self.previous_timestamp = x_timestamp
            self.z_scores = get_z_score(self.container)  # normalizzo gli elementi raccolti fin'ora
            self.init_iqr_range()
            self.update_counter()
            self.compute_range()
            #print(self.iqr_lower_bound)
            #print(self.iqr_upper_bound)

        elif self.counter > self.window:  # Procedo un elemento alla volta
            time_diff = x_timestamp - self.previous_timestamp

            #print(self.counter)
            self.container = self.container[-self.window-1:]
            self.container_timestamps = self.container_timestamps[-self.window-1:]

            self.z_scores = get_z_score(self.container)
            self.median = np.array(self.z_scores).mean()
            #print(self.median)
            if self.median == 0:
                self.median = self.previous_mean
            self.previous_mean = self.median

            diff = abs(x - self.previous_data)
            self.differences.append(diff)
            for j in range(len(self.z_scores)):
                self.split(self.z_scores[j])
            self.iqr_mean = np.array(self.container).mean()
            self.iqr_std = np.array(self.container).std()
            z = (x - self.iqr_mean) / self.iqr_std
            self.all_data.append(z)
            if diff > self.differences_upper_bound:
                if time_diff < self.max_time_difference:
                    self.anomalies.append(x)
                    return True
            else:
                self.compute_range()
            if z < self.iqr_lower_bound or z > self.iqr_upper_bound:

                self.update_plot_data()
                return True
            else:
                self.container.append(x)
                self.previous_data = x
                self.previous_timestamp = x_timestamp
                self.split(x)
                self.update_q1_q3()
                self.iqr = self.q3 - self.q1
                self.update_range()
                self.update_plot_data()
                return False


    def time_series_plot(self, sensor):

        fig, ax1 = plt.subplots(figsize=(16, 10))
        ax1.grid()
        # ax1.set_ylim([10 * self.iqr_lower_bound, 10 * self.iqr_upper_bound])
        ax1.set_ylabel(str(self.column_to_analyze), fontsize=18)
        ax1.set_xlabel(str(self.time_col), fontsize=18)
        ax1.set_title(str(self.column_to_analyze) + " IQR algorithm", fontsize=22)
        #ax1.set_xticks(ticks)
        #ax1.set_xticklabels(self.dataset[self.time_col][ticks])
        ax1.plot(self.all_data)
        ax1.plot(self.iqr_lb_to_plot, color='r')
        ax1.plot(self.iqr_ub_to_plot, color='r', label='upper bound')

        fig.autofmt_xdate()
        fig.savefig('sensors/'+sensor+'/calibrated_time_series/anomalies_iqr_' + self.column_to_analyze)



