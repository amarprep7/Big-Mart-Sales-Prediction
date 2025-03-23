from dataclasses import dataclass

@dataclass
class ModelConfig:
    random_state: int = 42
    n_estimators: int = 100
    n_features: int = 6
    cv_folds: int = 5
    current_year: int = 2025

@dataclass
class DataConfig:
    train_path: str = "data/raw/train_v9rqX0R.csv"
    test_path: str = "data/raw/test_AbJTz2l.csv"
    submission_path: str = "data/processed/submission.csv"

@dataclass
class FeatureConfig:
    outlet_size_mapping = {
        'Grocery Store': 'Small',
        'Supermarket Type1': 'Small',
        'Supermarket Type2': 'Medium',
        'Supermarket Type3': 'Medium'
    }
    
    fat_content_mapping = {
        'Low Fat': 'Low Fat',
        'LF': 'Low Fat',
        'low fat': 'Low Fat',
        'Regular': 'Regular',
        'reg': 'Regular'
    }