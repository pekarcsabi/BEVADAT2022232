# %%
import numpy as np
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import math

#csv_path = "diabetes.csv"

# %%
class KNNClassifier:
    counter = 0
    
    def __init__(self, k:int, test_split_ratio :float):
        self.k = k
        self.test_split_ratio = test_split_ratio

    @property
    def k_neihbors(self):
        return self.k

    @staticmethod
    def load_csv(csv_path:str)->Tuple[pd.DataFrame, pd.Series]:
        dataset= pd.read_csv(csv_path)
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        x,y = dataset.iloc[:,:-1],dataset.iloc[:,-1]
        return x,y
    
    @classmethod
    def train_test_split(self, features:pd.DataFrame, labels:pd.Series):
        test_size = int(len(features) *  self.test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"

        self.x_train, self.y_train = features.iloc[:train_size,:].reset_index(drop=True), labels.iloc[:train_size].reset_index(drop=True)
        self.x_test, self.y_test = features.iloc[train_size:train_size + test_size,:].reset_index(drop=True), labels.iloc[train_size:train_size + test_size].reset_index(drop=True)
    
    @classmethod
    def euclidean(self, element_of_x:pd.Series) -> pd.Series:
        return pd.np.sqrt(((self.x_train - element_of_x)**2).sum(axis=1))
    
    @classmethod
    def predict(self, x_test:pd.DataFrame):
        labels_pred = []
        for index, x_test_element in x_test.iterrows():
            distances = self.euclidean(x_test_element)
            distances = pd.DataFrame({'distances': distances, 'labels': self.y_train})
            distances.sort_values(by='distances', inplace=True)
            label_pred = mode(distances.iloc[:self.k,1],axis=0).mode[0]
            labels_pred.append(label_pred)
        self.y_preds = pd.Series(labels_pred)
            
    @classmethod
    def accuaracy(self) -> float:
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive / len(self.y_test) * 100
    
    @classmethod
    def plot_confusion_matrix(self):
        return confusion_matrix(self.y_test, self.y_preds)

# %%



