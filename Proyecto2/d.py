from pathlib import Path
import pandas as pd
import os
from zipfile import ZipFile

os.mkdir("dataset") if not os.path.exists("dataset") else None
if not os.path.exists("creditcard.csv"):
    os.system("kaggle datasets download -d mlg-ulb/creditcardfraud -p ./dataset")

    with ZipFile("./dataset/creditcardfraud.zip", "r") as zip_ref:
        zip_ref.extractall("./dataset")
    
    rmfile = Path("dataset/creditcardfraud.zip")
    if rmfile.exists():
        rmfile.unlink()

print("Dataset descargado correctamente")