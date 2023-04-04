import numpy as np
import pandas as pd

class NJCleaner():

    def __init__(self, csv_path:str):
        self.data = pd.read_csv(csv_path, skiprows=1, header=None)

    def order_by_scheduled_time(self) -> pd.DataFrame:
        return self.data.sort_values('scheduled_time')

