import pandas as pd

def load_data(path: str = "data/main_dataset.csv"):
    try:
        data = pd.read_csv(path, sep=';')  
        print(f"Successfully loaded {data.shape[0]} rows and {data.shape[1]} columns")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
   