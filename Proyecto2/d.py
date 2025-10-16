#imports
from pathlib import Path
import pandas as pd
import os
from zipfile import ZipFile

# Si no existe el dataset, lo descarga
os.mkdir("dataset") if not os.path.exists("dataset") else None
if not os.path.exists("creditcard.csv"):
    os.system("kaggle datasets download -d mlg-ulb/creditcardfraud -p ./dataset")

    # Descomprimir
    with ZipFile("./dataset/creditcardfraud.zip", "r") as zip_ref:
        zip_ref.extractall("./dataset")
    
    rmfile = Path("dataset/creditcardfraud.zip")
    if rmfile.exists():
        rmfile.unlink()
        
print("Dataset descargado correctamente")

csv_path = Path("dataset/creditcard.csv")

df = pd.read_csv(csv_path)

print(df.shape)
print(df.head())
print(df.info())
print("Distribución de la clase:")
print(df['Class'].value_counts())
print("Proporciones:")
print(df['Class'].value_counts(normalize=True))
