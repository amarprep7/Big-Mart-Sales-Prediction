import pandas as pd
import numpy as np
from typing import List, Dict, Union
import logging
from pathlib import Path
from .constants import *

logger = logging.getLogger(__name__)

def setup_directories() -> None:
    directories = [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    allow_nulls: bool = False
) -> bool:

    try:
        # Check columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check for nulls if not allowed
        if not allow_nulls and df[required_columns].isnull().any().any():
            raise ValueError("Null values found in required columns")
            
        return True
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return False

def load_data(filepath: str) -> pd.DataFrame:

    try:
        data = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {str(e)}")
        raise

def save_data(
    df: pd.DataFrame,
    filepath: str,
    index: bool = False
) -> None:

    try:
        df.to_csv(filepath, index=index)
        logger.info(f"Successfully saved data to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {str(e)}")
        raise

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:

    from sklearn.metrics import mean_squared_error, r2_score
    
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred)
    }
    return metrics

def create_submission(
    predictions: np.ndarray,
    test_ids: pd.DataFrame,
    filepath: str
) -> None:

    submission = test_ids[ID_COLS].copy()
    submission[TARGET_COL] = predictions
    save_data(submission, filepath)