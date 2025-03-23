from typing import List, Dict, Any
import pandas as pd
import numpy as np
from .constants import *

def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    quality_report = {
        "total_rows": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentages": (df.isnull().sum() / len(df) * 100).to_dict(),
        "duplicates": df.duplicated().sum(),
        "numerical_stats": df[NUMERICAL_COLS].describe().to_dict(),
        "categorical_counts": {
            col: df[col].value_counts().to_dict()
            for col in CATEGORICAL_COLS
        }
    }
    return quality_report

def validate_predictions(predictions: np.ndarray) -> bool:
    try:
        # Check for negative values
        if (predictions < 0).any():
            raise ValueError("Negative predictions found")
            
        # Check for invalid values
        if not np.isfinite(predictions).all():
            raise ValueError("Invalid prediction values found")
            
        return True
        
    except Exception as e:
        logger.error(f"Prediction validation failed: {str(e)}")
        return False