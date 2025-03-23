import pandas as pd
import logging
from pathlib import Path
from config.logging_config import setup_logging
from config.model_config import DataConfig
from data.clean_dataset import preprocess_data
from models.train_model import train_top_features_model

def main():
    logger = setup_logging()
    logger.info("Starting sales prediction pipeline")
    
    # Create directories if they don't exist
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess training data
    train_data = pd.read_csv(DataConfig.train_path)
    processed_train = preprocess_data(train_data)
    
    X = processed_train.drop('Item_Outlet_Sales', axis=1)
    y = processed_train['Item_Outlet_Sales']
    
    # Train model
    final_model, features, feature_importance, mean_r2_score = train_top_features_model(X, y)
    
    # Process test data and make predictions
    test_data = pd.read_csv(DataConfig.test_path)
    processed_test = preprocess_data(test_data)
    test_features = processed_test[features]
    predictions = final_model.predict(test_features)
    
    # Create submission
    submission = test_data[['Item_Identifier', 'Outlet_Identifier']].copy()
    submission['Item_Outlet_Sales'] = predictions
    submission.to_csv(DataConfig.submission_path, index=False)
    
    logger.info(f"Submission saved to {DataConfig.submission_path}")
    logger.info(f"Top features: {', '.join(features)}")

if __name__ == "__main__":
    main()