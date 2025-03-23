# Data related constants
RANDOM_SEED = 42
CURRENT_YEAR = 2025

# File paths
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
MODEL_DIR = "models"

# Column names
TARGET_COL = "Item_Outlet_Sales"
ID_COLS = ["Item_Identifier", "Outlet_Identifier"]
NUMERICAL_COLS = [
    "Item_Weight",
    "Item_Visibility",
    "Item_MRP",
    "Outlet_Establishment_Year"
]
CATEGORICAL_COLS = [
    "Item_Fat_Content",
    "Item_Type",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type"
]

# Feature engineering constants
MAPPINGS = {
    "fat_content": {
        "Low Fat": "Low Fat",
        "LF": "Low Fat",
        "low fat": "Low Fat",
        "Regular": "Regular",
        "reg": "Regular"
    },
    "outlet_size": {
        "Grocery Store": "Small",
        "Supermarket Type1": "Small",
        "Supermarket Type2": "Medium",
        "Supermarket Type3": "Medium"
    }
}