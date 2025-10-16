#imports
from pathlib import Path
import pandas as pd



csv_path = Path("dataset/creditcard.csv")

df = pd.read_csv(csv_path)

print(df.shape)
print(df.head())
print(df.info())
print("Distribución de la clase:")
print(df['Class'].value_counts())
print("Proporciones:")
print(df['Class'].value_counts(normalize=True))
