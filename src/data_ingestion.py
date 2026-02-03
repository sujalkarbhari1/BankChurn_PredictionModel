from config import Config

# Import Data Manipulation Libraries
import pandas as pd
import numpy as np

def data_ingestion():
    # Read the CSV file into a DataFrame
    df = pd.read_csv(Config.filepath)
    return df