import pandas as pd
try:
    df = pd.read_csv('data/train_labels.csv')
    with open('debug_cols.txt', 'w') as f:
        for col in df.columns:
            f.write(f"'{col}'\n")
    print("Columns written to debug_cols.txt")
except Exception as e:
    print(f"Error: {e}")
