import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import sklearn.datasets

class KMeansOnDigits:
    def __init__(self, n_clusters, random_state):
        self.n_clusters = n_clusters
        self.random_state = random_state
    
    def load_dataset(self):
        self.digits = sklearn.datasets.load_digits()
        
    def predict(self):
        self.clusters = KMeans(random_state=self.random_state, n_clusters=self.n_clusters).fit_predict(self.digits.data)
        
    def get_labels(self):
        self.labels = np.zeros_like(self.clusters)
        for i in range(self.n_clusters):
            mask = (self.clusters == i)
            self.labels[mask] = mode(self.digits.target[mask])[0]
            
    def calc_accuracy(self):
        self.accuracy=accuracy_score(self.digits.target, self.labels)
        
    def confusion_matrix(self):
        self.mat = confusion_matrix(self.digits.target, self.labels)
        #sns.heatmap(self.mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels = self.digits.target_names, yticklabels = self.digits.target_names)
        #plt.xlabel('true label')
        #plt.ylabel('predicted label')
