import pandas as pd
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "data", "laptops.csv")

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Cleaning to numeric
    df["RAM"] = pd.to_numeric(df["RAM"], errors="coerce")
    df["Storage"] = pd.to_numeric(df["Storage"], errors="coerce")
    df["Screen"] = pd.to_numeric(df["Screen"], errors="coerce")
    df["Final Price"] = pd.to_numeric(df["Final Price"], errors="coerce")

    return df

def recommend(df, usage=None, budget=None, brand=None, screen_size=None, preference=None, ram=None, storage=None, touchscreen=None):
    results = df.copy()

    # BRAND FILTER
    if brand:
        results = results[results["Brand"].str.contains(brand, case=False, na=False)]

    # SCREEN SIZE EXACT
    if screen_size:
        size = float(screen_size)
        results = results[results["Screen"].between(size - 0.3, size + 0.3)]

    # SCREEN SIZE PREFERENCE
    if preference:
        if preference == "big":
            results = results[results["Screen"] >= 15]
        elif preference == "medium":
            results = results[results["Screen"].between(14, 15)]
        elif preference == "small":
            results = results[results["Screen"] <= 14]

    # RAM FILTER
    if ram:
        results = results[results["RAM"] >= ram]
    
    # STORAGE FILTER
    if storage:
        results = results[results["Storage"] >= storage]

    # TOUCHSCREEN FILTER
    if touchscreen is not None:
        if touchscreen:
            results = results[results["Touch"].str.lower() == "yes"]
        else:
            results = results[results["Touch"].str.lower() == "no"]

    # USAGE FILTER
    if usage == "gaming":
        results = results[results["GPU"].notna()]
        results = results[results["RAM"] >= 8]

    elif usage == "coding":
        results = results[results["CPU"].str.contains("i5|i7|Ryzen", case=False, na=False)]
        results = results[results["RAM"] >= 8]

    elif usage == "editing":
        results = results[results["RAM"] >= 16]
        

    # BUDGET
    if budget:
        budget_usd = budget / 16000
        results = results[results["Final Price"] <= budget_usd]

    return results[["Brand", "Model", "GPU", "CPU", "RAM", "Storage", "Screen", "Touch", "Final Price"]].head(5)

# For testing only
if __name__ == "__main__":
    df = load_data()
    print(recommend(df, usage="gaming", budget=1500))
