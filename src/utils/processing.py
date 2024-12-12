import pandas as pd

from typing import List, Tuple

from numpy import ndarray
from pandas.core.frame import DataFrame
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self, file_path: str, segmentation_features: List[str]):
        self.file_path = file_path
        self.segmentation_features = segmentation_features
        self.df = None
        self.scaled_data = None
        self.scaler = None

    def load_data(self) -> DataFrame:
        try:
            data = pd.ExcelFile(self.file_path)
            self.df = data.parse(data.sheet_names[0])
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return None
        return self.df

    def preprocess_data(self) -> Tuple[DataFrame, ndarray, StandardScaler]:
        if self.df is None:
            print("Data not loaded. Please load data first.")
            return None, None, None

        df_cleaned = self.df.dropna(subset=self.segmentation_features)
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(
            df_cleaned[self.segmentation_features]
        )
        return df_cleaned, self.scaled_data, self.scaler

