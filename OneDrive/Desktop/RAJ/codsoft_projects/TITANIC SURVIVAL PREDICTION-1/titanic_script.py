import os
import pandas as pd

os.system("kaggle datasets download -d yasserh/titanic-dataset --unzip")
df = pd.read_csv("Titanic-Dataset.csv")
survived_count = df['Survived'].value_counts()

print(f"Survived: {survived_count[1]} passengers")
print(f"Did not survive: {survived_count[0]} passengers")
