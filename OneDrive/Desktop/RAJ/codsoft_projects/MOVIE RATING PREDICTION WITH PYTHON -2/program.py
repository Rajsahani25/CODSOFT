
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


zip_path = r"C:\Users\Acer25\OneDrive\Desktop\RAJ\codsoft_projects\MOVIE RATING PREDICTION WITH PYTHON -2\archive.zip"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    print("Files inside ZIP:", zip_ref.namelist())
    with zip_ref.open('IMDb Movies India.csv') as f:
        df = pd.read_csv(f, encoding='latin1')  
print("Columns in dataset:", df.columns.tolist())

df.columns = df.columns.str.strip()

if 'IMDB Rating' not in df.columns:
    for col in df.columns:
        if 'imdb' in col.lower() and 'rating' in col.lower():
            print(f"Detected IMDB Rating column: {col}")
            df.rename(columns={col: 'IMDB Rating'}, inplace=True)
            break

df = df.dropna(subset=['Rating'])

for col in ['Director', 'Actor 1', 'Actor 2', 'Genre']:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')
    else:
        print(f"Warning: Column '{col}' not found in data!")

features = df[['Genre', 'Director', 'Actor 1', 'Actor 2']]
target = df['Rating']

features_encoded = pd.get_dummies(features, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    features_encoded, target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel('Actual IMDB Ratings')
plt.ylabel('Predicted IMDB Ratings')
plt.title('Actual vs Predicted Ratings')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.grid(True)
plt.show()
