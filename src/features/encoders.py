from typing import Dict
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Robust Target Encoder – works with pandas 2.3+ and Categorical dtypes."""
    
    def __init__(self, column: str, smoothing: float = 10.0):
        self.column = column
        self.smoothing = smoothing
        self.encoding_map: Dict[str, float] = {}
        self.global_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        df = X[[self.column]].copy()
        df["_target"] = y.values
        self.global_mean = y.mean()

        # CRITICAL: observed=True → silences warning + faster + correct behavior
        agg = df.groupby(self.column, observed=True)["_target"].agg(["mean", "count"])
        
        smoothed = (
            agg["count"] * agg["mean"] + self.smoothing * self.global_mean
        ) / (agg["count"] + self.smoothing)
        
        self.encoding_map = smoothed.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # THIS IS THE BULLETPROOF LINE
        # .map() on Categorical returns Categorical → .astype(float) forces numeric
        encoded = (
            X[self.column]
            .map(self.encoding_map)
            .astype("float64")          # ← breaks the Categorical spell
            .fillna(self.global_mean)   # ← now safe
        )
        
        X[f"{self.column}_encoded"] = encoded.astype("float32")
        return X