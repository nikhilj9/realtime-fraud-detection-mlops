from typing import Dict
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encode categorical column using smoothed fraud rate."""
    
    def __init__(self, column: str, smoothing: float = 10.0):
        self.column = column
        self.smoothing = smoothing
        self.encoding_map: Dict[str, float] = {}
        self.global_mean: float = 0.0
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        df = X[[self.column]].copy()
        df["_target"] = y.values
        
        self.global_mean = y.mean()
        
        agg = df.groupby(self.column)["_target"].agg(["mean", "count"])
        smoothed = (agg["count"] * agg["mean"] + self.smoothing * self.global_mean) / (agg["count"] + self.smoothing)
        
        self.encoding_map = smoothed.to_dict()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[f"{self.column}_encoded"] = X[self.column].map(self.encoding_map).fillna(self.global_mean)
        return X