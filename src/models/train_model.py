import logging
from xgboost import XGBRFRegressor
import pandas as pd
from sklearn.model_selection import cross_val_score
from config.model_config import ModelConfig

logger = logging.getLogger(__name__)

def train_top_features_model(X, y):
    """
    Train model with feature selection and cross-validation
    """
    logger.info("Training model with top features selection...")
    
    xg_all = XGBRFRegressor(
        n_estimators=ModelConfig.n_estimators, 
        random_state=ModelConfig.random_state
    )
    xg_all.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xg_all.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top n features
    top_features = feature_importance.head(ModelConfig.n_features)
    X_selected = X[top_features['feature'].tolist()]
    
    # Cross-validation
    model = XGBRFRegressor(
        n_estimators=ModelConfig.n_estimators, 
        random_state=ModelConfig.random_state
    )
    cv_scores = cross_val_score(
        model, 
        X_selected, 
        y, 
        cv=ModelConfig.cv_folds, 
        scoring='r2',
        n_jobs=-1
    )
    
    # Train final model
    final_model = XGBRFRegressor(
        n_estimators=ModelConfig.n_estimators, 
        random_state=ModelConfig.random_state
    )
    final_model.fit(X_selected, y)
    
    mean_r2_score = cv_scores.mean()
    logger.info(f"Mean R2 score: {mean_r2_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return final_model, top_features['feature'].tolist(), feature_importance, mean_r2_score