import numpy as np
from scipy.stats import chi2


def chisquare_boundary(p, dimension):
    boundary = chi2.ppf(p, dimension) # Percent point function
    return boundary


class FFIDCAD:

    def __init__(self, feature_1, feature_2, lam=0.99, esponente=16, dofs=2):
        
        self.couple = [feature_1, feature_2]
        self.p = (1 - pow(10, -esponente))
        self.lam = lam
        self.S = np.identity(2)
        self.I = np.identity(2)
        self.bound = 0
        self.dofs = dofs
        self.mu = 0
        self.counter = 2
        self.feature_1_mean = 0
        self.feature_2_mean = 0
        self.feature_1_std = 0
        self.feature_2_std = 0
        self.indipendenza = 0

    def fit(self, dataset):
        
        self.bound = chisquare_boundary(self.p, self.dofs)
        x1 = np.array([dataset[self.couple[0]][0], dataset[self.couple[1]][0]])
        x2 = np.array([dataset[self.couple[0]][1], dataset[self.couple[1]][1]])
        self.mu = (x1 + x2) / 2
        self.mu = self.mu.reshape(2,1)

    def predict_anomaly(self, value1, value2):
        #print(self.counter)
        #print(self.mu)
        #print(self.S[0][1])
        self.counter = self.counter + 1
        self.indipendenza = self.indipendenza + self.S[0][1]
        x = np.array([value1, value2])
        x = x.reshape(2, 1)
        value = np.dot(np.dot((x - self.mu).T, self.S), (x - self.mu))
        """
        if value <= self.bound:
             a = np.dot(np.dot((x - self.mu), (x - self.mu).T), self.S)
             b = (pow(self.counter, 2) / self.counter) + np.dot(np.dot((x - self.mu).T, self.S), (x - self.mu))
             self.S = np.dot((self.counter * self.S / (self.counter - 1)), (self.I - a / b))
             self.mu = self.mu * self.lam + (1 - self.lam) * x
        else:
            return True
        """
        a = np.dot(np.dot((x - self.mu), (x - self.mu).T), self.S)
        b = (pow(self.counter, 2) / self.counter) + np.dot(np.dot((x - self.mu).T, self.S), (x - self.mu))
        self.S = np.dot((self.counter * self.S / (self.counter - 1)), (self.I - a / b))
        self.mu = self.mu * self.lam + (1 - self.lam) * x

        # print(self.S)
        if value > self.bound:
            return True

    def stampa_indipendenza(self, len):
        print(self.indipendenza / len)































