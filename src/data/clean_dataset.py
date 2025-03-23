import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from config.model_config import FeatureConfig, ModelConfig

def preprocess_data(data):
    """
    Preprocess the input data with feature engineering and encoding
    """
    data = data.copy()
    
    # Handle missing values
    data['Item_Weight'] = data['Item_Weight'].interpolate(method="linear")
    data['Item_Visibility'] = data['Item_Visibility'].replace(0, np.nan).interpolate(method='linear')
    
    # Fill missing Outlet_Size based on Outlet_Type
    for outlet_type, size in FeatureConfig.outlet_size_mapping.items():
        data.loc[(data['Outlet_Type'] == outlet_type) & data['Outlet_Size'].isnull(), 'Outlet_Size'] = size
    
    # Standardize Item_Fat_Content
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(FeatureConfig.fat_content_mapping)
    
    # Feature engineering
    data['Item_Identifier'] = data['Item_Identifier'].apply(lambda x: x[:2])
    data['Outlet_Establishment_Year'] = ModelConfig.current_year - data['Outlet_Establishment_Year']
    
    data['Price_per_Weight'] = data['Item_MRP'] / data['Item_Weight']
    data['Store_Age_Size'] = data['Outlet_Establishment_Year'] * data['Outlet_Size']
    data['Visibility_MRP'] = data['Item_Visibility'] * data['Item_MRP']
    
    # Encode categorical variables
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        oe = OrdinalEncoder()
        data[col] = oe.fit_transform(data[[col]])
    
    return data