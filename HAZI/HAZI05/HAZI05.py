# %%
import numpy as np
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

csv_path = "diabetes.csv"

# %%
class KNNClassifier:
    counter = 0
    k: int
    test_split_ratio: float
    
    def __init__(self, k:int, test_split_ratio :float):
        KNNClassifier.k = k
        KNNClassifier.test_split_ratio = test_split_ratio
        KNNClassifier.counter += 1

    @staticmethod
    def load_csv(csv_path:str)->Tuple[pd.Series, pd.Series]:
        dataset= pd.read_csv(csv_path)
        dataset = dataset.sample(frac=1, random_state=42)
        x,y = dataset.iloc[:,:-1],dataset.iloc[:,-1]
        return x,y
    
    @classmethod
    def train_test_split(self, features:pd.DataFrame, labels:pd.DataFrame):
        test_size = int(len(features)*  self.test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"

        self.x_train, self.y_train = features.iloc[:train_size,:], labels.iloc[:train_size]
        self.x_test, self.y_test = features.iloc[train_size:train_size + test_size,:], labels.iloc[train_size:train_size + test_size]
    
    @classmethod
    def euclidean(self, element_of_x:pd.DataFrame) -> pd.Series:
        return (((self.x_train - element_of_x) ** 2).sum(axis=1)) ** 1/2
    
    @classmethod
    def predict(self, x_test:pd.DataFrame):
        self.y_spreds = []
        for index, x_test_element in x_test.iterrows():
            distances = self.euclidean(x_test_element)
            distances_and_labels = pd.concat([distances, self.y_train], axis=1).sort_values(by=0, ascending=True)
            label_pred = distances_and_labels.iloc[:self.k, 1].mode().values[0]
            self.y_spreds.append(label_pred)
            self.y_spreds = pd.DataFrame(self.y_spreds, columns=['y_pred'], index=x_test.index)
            
    @classmethod
    def accuaracy(self) -> float:
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive / len(self.y_test) * 100
    
    @property
    def k_neihbors(self):
        return self.k

# %%



