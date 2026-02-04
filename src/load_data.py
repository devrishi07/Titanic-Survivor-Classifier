from pathlib import Path
import pandas as pd

# project root = one level above src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

def load_train():
    return pd.read_csv(DATA_DIR / "train.csv")

def load_test():
    return pd.read_csv(DATA_DIR / "test.csv")
    
